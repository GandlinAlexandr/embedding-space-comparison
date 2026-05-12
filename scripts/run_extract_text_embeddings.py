from __future__ import annotations

import argparse
import gc
import importlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple, Sequence

import numpy as np
from tqdm import tqdm

from configs.text_benchmark_configs import (
    TEXT_EMBEDDING_MODEL_BY_ID,
    TEXT_EMBEDDING_CPU_MODEL_IDS,
    TEXT_EMBEDDING_CPU_MODEL_IDS_CSV,
    TEXT_EMBEDDING_MODEL_IDS,
    TEXT_EMBEDDING_MODEL_IDS_CSV,
    TEXT_EMBEDDING_SNOWFLAKE_MODEL_IDS,
    TEXT_EMBEDDING_SNOWFLAKE_MODEL_IDS_CSV,
    TEXT_EMBEDDING_TEXT20_MODEL_IDS,
    TEXT_EMBEDDING_TEXT20_MODEL_IDS_CSV,
    TextEmbeddingModelSpec,
)


def _read_jsonl(path: Path) -> tuple[List[str], np.ndarray]:
    texts: List[str] = []
    labels: List[int] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, 1):
            raw = raw.strip()
            if not raw:
                continue
            obj = json.loads(raw)
            if "text" not in obj or "label" not in obj:
                raise ValueError(f"{path}:{line_no}: ожидались ключи 'text' и 'label'")
            texts.append(str(obj["text"]))
            labels.append(int(obj["label"]))
    if not texts:
        raise ValueError(f"В файле нет строк: {path}")
    return texts, np.asarray(labels, dtype=np.int64)


def _dataset_key(dataset: str, split: str) -> str:
    return f"{dataset}_{split}"


def _resolve_input_jsonl(args: argparse.Namespace) -> Path:
    if args.input_jsonl:
        return Path(args.input_jsonl)
    canonical_path = Path("data") / args.dataset / f"{args.split}.jsonl"
    if canonical_path.exists():
        return canonical_path

    legacy_path = (
        Path("data") / "text_classification" / args.dataset / f"{args.split}.jsonl"
    )
    if legacy_path.exists():
        print(
            "[ПРЕДУПРЕЖДЕНИЕ] Используется старый путь к текстовому набору. "
            "Лучше перенести данные в: "
            f"{canonical_path}"
        )
        return legacy_path

    return canonical_path


def _resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir:
        return Path(args.output_dir)
    return Path("data") / "embeddings" / _dataset_key(args.dataset, args.split)


def _parse_models(raw: str) -> List[str]:
    raw = str(raw).strip()
    if raw == "primary":
        return list(TEXT_EMBEDDING_MODEL_IDS)
    if raw == "cpu":
        return list(TEXT_EMBEDDING_CPU_MODEL_IDS)
    if raw == "snowflake":
        return list(TEXT_EMBEDDING_SNOWFLAKE_MODEL_IDS)
    if raw == "text20":
        return list(TEXT_EMBEDDING_TEXT20_MODEL_IDS)
    model_ids = [x.strip() for x in raw.split(",") if x.strip()]
    unknown = [x for x in model_ids if x not in TEXT_EMBEDDING_MODEL_BY_ID]
    if unknown:
        raise ValueError(
            f"Неизвестные идентификаторы текстовых моделей: {unknown}. "
            f"Доступно: {TEXT_EMBEDDING_MODEL_IDS}"
        )
    return model_ids


def _batched(items: Sequence[str], batch_size: int) -> Iterable[List[str]]:
    for start in range(0, len(items), batch_size):
        yield list(items[start : start + batch_size])


def _prefixed_texts(texts: Sequence[str], prefix: str) -> List[str]:
    if not prefix:
        return list(texts)
    return [prefix + text for text in texts]


def _import_transformers_and_torch():
    try:
        transformers = importlib.import_module("transformers")
        torch = importlib.import_module("torch")
        AutoTokenizer = getattr(transformers, "AutoTokenizer")
        AutoModel = getattr(transformers, "AutoModel")
        return AutoTokenizer, AutoModel, torch
    except Exception as exc:
        raise RuntimeError(
            "Не удалось импортировать transformers/torch для извлечения "
            "текстовых эмбеддингов. "
            f"Исходная ошибка: {type(exc).__name__}: {exc}"
        ) from exc


def _cache_dir_arg(raw: str) -> str | None:
    raw = str(raw).strip()
    return raw or None


def _check_runtime_requirements() -> None:
    missing: List[str] = []
    try:
        _import_transformers_and_torch()
    except Exception as exc:
        missing.append("transformers/torch")

    if missing:
        raise RuntimeError(
            "Не выполнены требования для извлечения текстовых эмбеддингов: "
            + ", ".join(missing)
            + f". Подробности: {exc}"
        )


def _configure_torch_threads(torch: Any, num_threads: int) -> None:
    if int(num_threads) <= 0:
        return
    torch.set_num_threads(int(num_threads))
    try:
        torch.set_num_interop_threads(max(1, min(4, int(num_threads))))
    except RuntimeError:
        # PyTorch allows changing interop threads only before parallel work starts.
        pass


def _mean_pool(last_hidden_state: Any, attention_mask: Any, torch: Any) -> Any:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


def _pool_outputs(outputs: Any, attention_mask: Any, torch: Any, pooling: str) -> Any:
    if pooling == "mean":
        return _mean_pool(outputs.last_hidden_state, attention_mask, torch)
    if pooling == "cls":
        return outputs.last_hidden_state[:, 0]
    raise ValueError(f"Неподдерживаемый режим pooling: {pooling}")


def _encode_model(
    spec: TextEmbeddingModelSpec,
    texts: Sequence[str],
    *,
    batch_size: int,
    device: str,
    normalize_embeddings: bool,
    model_cache_dir: str | None,
    num_threads: int,
) -> np.ndarray:
    AutoTokenizer, AutoModel, torch = _import_transformers_and_torch()
    _configure_torch_threads(torch, int(num_threads))
    runtime_device = torch.device(
        device if device == "cpu" or torch.cuda.is_available() else "cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        spec.hf_name,
        cache_dir=model_cache_dir,
        trust_remote_code=bool(spec.trust_remote_code),
    )
    model = AutoModel.from_pretrained(
        spec.hf_name,
        use_safetensors=True,
        cache_dir=model_cache_dir,
        trust_remote_code=bool(spec.trust_remote_code),
    ).to(runtime_device)
    model.eval()

    encoded_texts = _prefixed_texts(texts, spec.prompt_prefix)

    chunks: List[np.ndarray] = []
    with torch.no_grad():
        total_batches = (len(encoded_texts) + int(batch_size) - 1) // int(batch_size)
        for batch in tqdm(
            _batched(encoded_texts, int(batch_size)),
            total=total_batches,
            desc=f"Батчи {spec.model_id}",
            unit="batch",
        ):
            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            inputs = {k: v.to(runtime_device) for k, v in inputs.items()}
            outputs = model(**inputs)
            pooled = _pool_outputs(
                outputs, inputs["attention_mask"], torch, spec.pooling
            )
            if normalize_embeddings:
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            chunks.append(pooled.detach().cpu().numpy().astype(np.float32))

    out = np.concatenate(chunks, axis=0)
    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def _download_model_assets(
    specs: Sequence[TextEmbeddingModelSpec],
    *,
    model_cache_dir: str | None,
) -> None:
    AutoTokenizer, AutoModel, torch = _import_transformers_and_torch()
    if model_cache_dir:
        Path(model_cache_dir).mkdir(parents=True, exist_ok=True)

    print(f"Моделей к скачиванию: {len(specs)}")
    print(f"Кэш моделей: {model_cache_dir or 'кэш HuggingFace по умолчанию'}")

    for spec in tqdm(specs, desc="Модели", unit="model"):
        print(f"\n=== Модель: {spec.model_id} ===")
        print(f"HF: {spec.hf_name}")
        print(f"Семейство: {spec.family}")
        if spec.prompt_prefix:
            print(f"Префикс текста: {spec.prompt_prefix!r}")
        print(f"Pooling: {spec.pooling}")
        print(f"Доверять удалённому коду: {bool(spec.trust_remote_code)}")

        tokenizer = AutoTokenizer.from_pretrained(
            spec.hf_name,
            cache_dir=model_cache_dir,
            trust_remote_code=bool(spec.trust_remote_code),
        )
        model = AutoModel.from_pretrained(
            spec.hf_name,
            use_safetensors=True,
            cache_dir=model_cache_dir,
            trust_remote_code=bool(spec.trust_remote_code),
        )
        print(
            "Загружено: "
            f"tokenizer={tokenizer.__class__.__name__}, "
            f"model={model.__class__.__name__}"
        )
        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Извлечь эмбеддинги трансформерных моделей из локальных JSONL-файлов "
            "и сохранить их в формате .npy, принятом в проекте."
        )
    )
    parser.add_argument(
        "--dataset",
        default="",
        help="Ключ набора данных для стандартных путей, например banking77.",
    )
    parser.add_argument(
        "--split", default="", help="Ключ разбиения, например train/test/val."
    )
    parser.add_argument(
        "--input_jsonl",
        default="",
        help="Явный путь к локальному JSONL. По умолчанию: data/<dataset>/<split>.jsonl.",
    )
    parser.add_argument(
        "--output_dir",
        default="",
        help="Папка вывода. По умолчанию: data/embeddings/<dataset>_<split>.",
    )
    parser.add_argument(
        "--models",
        default="primary",
        help=(
            "Идентификаторы текстовых моделей через запятую или 'primary'. "
            f"Primary: {TEXT_EMBEDDING_MODEL_IDS_CSV}. "
            f"CPU: {TEXT_EMBEDDING_CPU_MODEL_IDS_CSV}. "
            f"Snowflake: {TEXT_EMBEDDING_SNOWFLAKE_MODEL_IDS_CSV}. "
            f"Text20: {TEXT_EMBEDDING_TEXT20_MODEL_IDS_CSV}"
        ),
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--num_threads",
        type=int,
        default=0,
        help="Число CPU-потоков для torch. 0 оставляет значение torch по умолчанию.",
    )
    parser.add_argument(
        "--num_interop_threads",
        type=int,
        default=0,
        help=(
            "Число interop-потоков CPU для torch. 0 использует небольшое "
            "значение, выведенное из --num_threads."
        ),
    )
    parser.add_argument(
        "--model_cache_dir",
        default="data/.hf_model_cache",
        help="Папка кэша HuggingFace для файлов модели и токенайзера.",
    )
    parser.add_argument(
        "--download_models_only",
        action="store_true",
        help=(
            "Скачать файлы выбранных моделей/токенайзеров и выйти; "
            "наборы данных не читаются."
        ),
    )
    parser.add_argument(
        "--normalize_embeddings",
        action="store_true",
        help="Сохранять L2-нормированные эмбеддинги предложений.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Пересчитывать эмбеддинги модели, даже если выходной файл уже существует.",
    )
    parser.add_argument(
        "--check_requirements",
        action="store_true",
        help=(
            "Только проверить зависимости Python и выйти; "
            "данные и модели не скачиваются."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.check_requirements:
        _check_runtime_requirements()
        print("Требования для извлечения текстовых эмбеддингов: OK")
        return

    model_ids = _parse_models(args.models)
    model_specs = [TEXT_EMBEDDING_MODEL_BY_ID[model_id] for model_id in model_ids]
    model_cache_dir = _cache_dir_arg(args.model_cache_dir)
    AutoTokenizer, AutoModel, torch = _import_transformers_and_torch()
    if int(args.num_threads) > 0:
        torch.set_num_threads(int(args.num_threads))
    if int(args.num_interop_threads) > 0:
        try:
            torch.set_num_interop_threads(int(args.num_interop_threads))
        except RuntimeError:
            pass
    elif int(args.num_threads) > 0:
        try:
            torch.set_num_interop_threads(max(1, min(4, int(args.num_threads))))
        except RuntimeError:
            pass

    if args.download_models_only:
        _download_model_assets(model_specs, model_cache_dir=model_cache_dir)
        print("\nГотово: модели и токенайзеры скачаны.")
        return

    if not args.dataset or not args.split:
        raise ValueError(
            "Для извлечения эмбеддингов нужно указать --dataset и --split."
        )

    input_jsonl = _resolve_input_jsonl(args)
    if not input_jsonl.exists():
        raise FileNotFoundError(f"Входной JSONL не найден: {input_jsonl}")

    output_dir = _resolve_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)

    texts, labels = _read_jsonl(input_jsonl)
    labels_path = output_dir / "labels.npy"
    np.save(labels_path, labels)

    saved_models: List[Dict[str, Any]] = []
    classes, counts = np.unique(labels, return_counts=True)

    print(f"Датасет: {args.dataset}")
    print(f"Сплит: {args.split}")
    print(f"Вход: {input_jsonl}")
    print(f"Выход: {output_dir}")
    print(f"Строк: {labels.shape[0]}")
    print(f"Классов: {classes.size}")
    print(f"Мин/макс объектов на класс: {int(counts.min())} / {int(counts.max())}")
    print(f"Моделей: {len(model_specs)}")
    print(f"Размер батча: {int(args.batch_size)}")
    print(f"Запрошенное устройство: {args.device}")
    print(f"Запрошено CPU-потоков torch: {int(args.num_threads)}")
    print(f"Фактически CPU-потоков torch: {torch.get_num_threads()}")
    print(f"Фактически interop-потоков torch: {torch.get_num_interop_threads()}")
    print(f"Кэш моделей: {model_cache_dir or 'кэш HuggingFace по умолчанию'}")

    for model_idx, spec in enumerate(tqdm(model_specs, desc="Модели", unit="model"), 1):
        out_path = output_dir / f"{spec.model_id}.npy"
        if out_path.exists() and not args.overwrite:
            print(f"\n=== Модель {model_idx}/{len(model_specs)}: {spec.model_id} ===")
            print(f"Пропуск: файл уже существует: {out_path}")
            saved_models.append(
                {
                    "model_id": spec.model_id,
                    "hf_name": spec.hf_name,
                    "family": spec.family,
                    "prompt_prefix": spec.prompt_prefix,
                    "pooling": spec.pooling,
                    "trust_remote_code": bool(spec.trust_remote_code),
                    "path": str(out_path),
                    "skipped_existing": True,
                }
            )
            continue

        print(f"\n=== Модель {model_idx}/{len(model_specs)}: {spec.model_id} ===")
        print(f"HF: {spec.hf_name}")
        print(f"Семейство: {spec.family}")
        if spec.prompt_prefix:
            print(f"Префикс текста: {spec.prompt_prefix!r}")
        print(f"Pooling: {spec.pooling}")
        print(f"Доверять удалённому коду: {bool(spec.trust_remote_code)}")
        print(f"Файл: {out_path}")
        effective_batch_size = int(args.batch_size)
        print(f"Фактический размер батча: {effective_batch_size}")
        embeddings = _encode_model(
            spec,
            texts,
            batch_size=effective_batch_size,
            device=str(args.device),
            normalize_embeddings=bool(args.normalize_embeddings),
            model_cache_dir=model_cache_dir,
            num_threads=int(args.num_threads),
        )
        if embeddings.shape[0] != labels.shape[0]:
            raise RuntimeError(
                f"Число строк эмбеддингов не совпадает с числом меток для "
                f"{spec.model_id}: {embeddings.shape[0]} против {labels.shape[0]}"
            )
        np.save(out_path, embeddings.astype(np.float32))
        print(f"Сохранено: {out_path} | shape={embeddings.shape}")
        saved_models.append(
            {
                "model_id": spec.model_id,
                "hf_name": spec.hf_name,
                "family": spec.family,
                "prompt_prefix": spec.prompt_prefix,
                "pooling": spec.pooling,
                "trust_remote_code": bool(spec.trust_remote_code),
                "path": str(out_path),
                "shape": list(map(int, embeddings.shape)),
                "batch_size": int(effective_batch_size),
            }
        )

    manifest = {
        "schema_version": 1,
        "kind": "extracted_embeddings",
        "dataset": str(args.dataset),
        "split": str(args.split),
        "dataset_key": _dataset_key(args.dataset, args.split),
        "sampled": False,
        "sample_size": int(labels.shape[0]),
        "full_size": int(labels.shape[0]),
        "seed": None,
        "subset_strategy": "",
        "input_jsonl": str(input_jsonl),
        "output_dir": str(output_dir.resolve()),
        "labels_path": str(labels_path.resolve()),
        "n_rows": int(labels.shape[0]),
        "n_classes": int(classes.size),
        "class_counts": {str(int(c)): int(n) for c, n in zip(classes, counts)},
        "normalize_embeddings": bool(args.normalize_embeddings),
        "batch_size": int(args.batch_size),
        "models": [m["model_id"] for m in saved_models],
        "text_model_details": saved_models,
    }
    (output_dir / "embeddings_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Сохранён манифест: {output_dir / 'embeddings_manifest.json'}")


if __name__ == "__main__":
    main()
