<a name="readme-top"></a>


[license-shield]: https://img.shields.io/github/license/GandlinAlexandr/embedding-space-comparison.svg?style=for-the-badge
[license-url]: https://github.com/GandlinAlexandr/embedding-space-comparison/blob/main/LICENSE

[![MIT][license-shield]][license-url]
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org/)
[!['Black'](https://img.shields.io/badge/code_style-black-black?style=for-the-badge)](https://github.com/psf/black)

<h1 align="center">Геометрические методы сравнения пространств эмбеддингов</h1>
  
  <p align="center">
    Проект в рамках ВКР магистерской программы "Искусственный Интеллект"
  </p>

<details>
  <summary>Содержание</summary>
  <ol>
    <li><a href="#описание-проекта">Описание проекта</a></li>
    <li><a href="#технологии">Технологии</a></li>
    <li><a href="#содержание-проекта">Содержание проекта</a></li>
      <ul><li><a href="#примеры-запуска">Примеры запуска</a></li></ul>
    <li><a href="#лицензия">Лицензия</a></li>
    <li><a href="#контакты">Контакты</a></li>
  </ol>
</details>


# Описание проекта

Данный проект предназначен для разработки метрики геометрического сравнения эмбеддингов нейросетей, основанной на ранге матрицы отображения.

<p align="right">(<a href="#readme-top">Вернуться к началу</a>)</p>

# Технологии

Для реализации проекта используются следующие технологии:

* [![Colab][Colab]][Colab-url]
* [![Python][Python.org]][Python-url]
  * [![Matplotlib][Matplotlib.org]][Matplotlib-url]
  * [![Numpy][Numpy.org]][Numpy-url]
  * [![Pandas][Рandas.pydata.org]][Pandas-url]
  * [![Pytorch][Pytorch]][Pytorch-url]
  * [![scikit-learn][scikit-learn]][scikit-learn-url]
  * [![SciPy][SciPy]][SciPy-url]


<p align="right">(<a href="#readme-top">Вернуться к началу</a>)</p>

# Содержание проекта

## Примеры запуска

Ниже приведены актуальные примеры запуска через `bash`. Все изменяемые значения вынесены в переменные: набор данных, модели, папки, список `k`, список метрик и режим геометрии. В самих командах ничего заменять не нужно.

### Основные скрипты

| Скрипт | Назначение |
| --- | --- |
| `scripts.run_extract_embeddings` | Сохраняет эмбеддинги моделей для изображений. |
| `scripts.run_extract_text_embeddings` | Сохраняет эмбеддинги моделей для текстов. |
| `scripts.run_compute_downstream_scores` | Обучает простой классификатор на эмбеддингах и сохраняет качество моделей. |
| `scripts.run_compute_local_map_store` | Считает локальные линейные отображения между парами моделей. |
| `scripts.run_compute_metrics_from_local_maps` | Строит матрицы метрик по уже сохранённым локальным отображениям. |
| `scripts.run_compute_embedding_metrics` | Старый способ расчёта парных метрик напрямую из эмбеддингов. |
| `scripts.run_evaluate_metrics` | Сравнивает парные метрики с разницей качества моделей. |
| `scripts.run_compute_single_metrics` | Считает одиночные метрики качества эмбеддингов. |
| `scripts.run_evaluate_single_metrics` | Сравнивает одиночные метрики с качеством моделей. |
| `scripts.run_diagnose_local_map` | Строит диагностические графики по локальным отображениям. |
| `scripts.plot_metric_summary`, `scripts.plot_single_metric_summary`, `scripts.plot_single_metric_diagnostics`, `scripts.plot_pairwise_error_heatmaps` | Строят итоговые графики. |
| `scripts.plot_centering_comparison_diagnostics` | Сравнивает режимы центрирования по готовым файлам артефактов. |

<details>
  <summary><b>▶ Полный расчёт через сохранённые локальные отображения</b></summary>

1. Задать рабочую папку проекта и основные пути:

   ```bash
   PROJECT_DIR="$HOME/YarTwoProject"
   cd "$PROJECT_DIR"

   DATA_ROOT="$PROJECT_DIR/data"
   DATASET="food101"
   TRAIN_SPLIT="train"
   TEST_SPLIT="test"
   TRAIN_DATASET_KEY="${DATASET}_${TRAIN_SPLIT}"
   TEST_DATASET_KEY="${DATASET}_${TEST_SPLIT}"

   TRAIN_EMB_DIR="$DATA_ROOT/embeddings/$TRAIN_DATASET_KEY"
   TEST_EMB_DIR="$DATA_ROOT/embeddings/$TEST_DATASET_KEY"
   DOWNSTREAM_JSON="$DATA_ROOT/downstream/${DATASET}_mlp.json"
   LOCAL_MAPS_ROOT="$DATA_ROOT/local_maps"
   EXPERIMENT_ROOT="$DATA_ROOT/experiments/${DATASET}_centered_v2_knn"
   ```

2. Задать модели и общие параметры расчёта:

   ```bash
   MODELS="resnet18,resnet34,resnet50,resnet50_v2,resnet101,resnet101_v2,wide_resnet50_2,wide_resnet50_2_v2,wide_resnet101_2,wide_resnet101_2_v2,vgg11,vgg13,vgg16,vgg19,vit_b_16,vit_b_16_swag_e2e,vit_b_16_swag_linear,vit_b_32,vit_l_16,vit_l_16_swag_linear,vit_l_32"
   SEED="42"
   BATCH_SIZE="128"
   DEVICE="auto"
   BACKEND="cuda"
   N_CENTERS="200"
   MAP_DTYPE="float32"
   ```

3. Задать режим геометрии и окрестности для сравнения обычного kNN с адаптивным kNN:

   ```bash
   LOCAL_GEOMETRY_MODE="centered_offsets_v2"
   K_LIST="5,10,20,40,80"
   RFF_K_LIST=""
   EPS_PERCENTILES=""
   WEIGHTED_EPS_PERCENTILES=""
   WEIGHTED_EPS_SCALE="3.0"
   ```

4. Если нужен полный набор окрестностей, задать kNN, RFF и epsilon-варианты:

   ```bash
   LOCAL_GEOMETRY_MODE="absolute_coords_v0"
   EXPERIMENT_ROOT="$DATA_ROOT/experiments/${DATASET}_absolute_all"
   K_LIST="5,10,20,40,80"
   RFF_K_LIST="10"
   EPS_PERCENTILES="5,10,20"
   WEIGHTED_EPS_PERCENTILES="5,10,20"
   WEIGHTED_EPS_SCALE="3.0"
   ```

5. Задать параметры отчётов и графиков:

   ```bash
   PLOTS_EXT="png,svg"
   PLOTS_MODE="alltasks"
   EVAL_PROTOCOL="delta_signed"
   ```

6. Сохранить эмбеддинги обучающей части набора данных:

   ```bash
   python -m scripts.run_extract_embeddings \
     --dataset "$DATASET" \
     --data_root "$DATA_ROOT" \
     --split "$TRAIN_SPLIT" \
     --output_dir "$TRAIN_EMB_DIR" \
     --models "$MODELS" \
     --batch_size "$BATCH_SIZE"
   ```

7. Сохранить эмбеддинги тестовой части набора данных:

   ```bash
   python -m scripts.run_extract_embeddings \
     --dataset "$DATASET" \
     --data_root "$DATA_ROOT" \
     --split "$TEST_SPLIT" \
     --output_dir "$TEST_EMB_DIR" \
     --models "$MODELS" \
     --batch_size "$BATCH_SIZE"
   ```

8. Обучить простой классификатор на обучающих эмбеддингах, проверить его на тестовых эмбеддингах и сохранить качество моделей:

   ```bash
   python -m scripts.run_compute_downstream_scores \
     --train_embeddings_dir "$TRAIN_EMB_DIR" \
     --test_embeddings_dir "$TEST_EMB_DIR" \
     --dataset "$DATASET" \
     --data_root "$DATA_ROOT" \
     --out_json "$DOWNSTREAM_JSON" \
     --task_name "${DATASET}_mlp"
   ```

9. Посчитать локальные отображения между парами моделей. Без `--store_maps` сохраняются спектры и служебные величины, а не полные матрицы отображений:

   ```bash
   python -m scripts.run_compute_local_map_store \
     --embeddings_dir "$TEST_EMB_DIR" \
     --out_root "$LOCAL_MAPS_ROOT" \
     --dataset_key "$TEST_DATASET_KEY" \
     --models "$MODELS" \
     --k_list "$K_LIST" \
     --rff_k_list "$RFF_K_LIST" \
     --eps_percentiles "$EPS_PERCENTILES" \
     --weighted_eps_percentiles "$WEIGHTED_EPS_PERCENTILES" \
     --weighted_eps_scale "$WEIGHTED_EPS_SCALE" \
     --n_centers "$N_CENTERS" \
     --seed "$SEED" \
     --local_geometry_mode "$LOCAL_GEOMETRY_MODE" \
     --backend "$BACKEND" \
     --map_dtype "$MAP_DTYPE" \
     --incremental
   ```

10. Задать папку с готовыми локальными отображениями. Значение `STORE_ID` берётся из имени папки внутри `data/local_maps/<ключ_набора>/`:

   ```bash
   STORE_ID="45734736e2cf"
   MAPS_DIR="$LOCAL_MAPS_ROOT/$TEST_DATASET_KEY/$STORE_ID"
   METRICS_EXPERIMENT_DIR="$EXPERIMENT_ROOT/store_metrics"
   ```

11. Посчитать фиксированные kNN-метрики и адаптивную kNN-метрику по готовым локальным отображениям:

   ```bash
   SELECTORS="fixed_k,adaptive"
   FIXED_KS="$K_LIST"
   AGGREGATIONS="rankme,stable_rank,nesum,pseudo_condition_number,alpha_req,spectral_entropy,hard_rank,tail_spectrum_log_ratio"
   PAIR_AGG="antisym"

   python -m scripts.run_compute_metrics_from_local_maps \
     --maps_dir "$MAPS_DIR" \
     --dataset_key "$TEST_DATASET_KEY" \
     --models "$MODELS" \
     --selectors "$SELECTORS" \
     --fixed_ks "$FIXED_KS" \
     --k_list "$K_LIST" \
     --aggregations "$AGGREGATIONS" \
     --pair_agg "$PAIR_AGG" \
     --experiment_dir "$METRICS_EXPERIMENT_DIR" \
     --downstream_json "$DOWNSTREAM_JSON" \
     --eval_protocol "$EVAL_PROTOCOL" \
     --local_geometry_mode "$LOCAL_GEOMETRY_MODE" \
     --plots_mode "$PLOTS_MODE" \
     --plots_ext "$PLOTS_EXT" \
     --incremental
   ```

12. Посчитать явно перечисленные метрики, включая RFF и epsilon-варианты:

   ```bash
   INCLUDE_METRICS="lin_k5_rankme_antisym,lin_k10_rankme_antisym,lin_k20_rankme_antisym,lin_k40_rankme_antisym,lin_k80_rankme_antisym,adaptive_k5_10_20_40_80_rankme_antisym,rff_k10_rankme_antisym,lin_eps_5_rankme_antisym,lin_eps_10_rankme_antisym,lin_eps_20_rankme_antisym,w_eps_5_rankme_antisym,w_eps_10_rankme_antisym,w_eps_20_rankme_antisym"

   python -m scripts.run_compute_metrics_from_local_maps \
     --maps_dir "$MAPS_DIR" \
     --dataset_key "$TEST_DATASET_KEY" \
     --models "$MODELS" \
     --include "$INCLUDE_METRICS" \
     --experiment_dir "$METRICS_EXPERIMENT_DIR" \
     --downstream_json "$DOWNSTREAM_JSON" \
     --eval_protocol "$EVAL_PROTOCOL" \
     --local_geometry_mode "$LOCAL_GEOMETRY_MODE" \
     --plots_mode "$PLOTS_MODE" \
     --plots_ext "$PLOTS_EXT" \
     --incremental
   ```

Примечания:

* Для сравнения фиксированных kNN-метрик с адаптивной kNN-метрикой используйте один и тот же `LOCAL_GEOMETRY_MODE`.
* Для текущего варианта с центрированием используется `centered_offsets_v2`.
* Для старого варианта без центрирования используется `absolute_coords_v0`.
* `--incremental` продолжает расчёт и не пересчитывает уже готовые результаты.
* Без `--store_maps` сохраняются спектры и служебные значения. Это основной режим для больших запусков.

</details>

<details>
  <summary><b>▶ Диагностика и вспомогательные запуски</b></summary>

1. Задать основные пути к уже посчитанному эксперименту:

   ```bash
   PROJECT_DIR="$HOME/YarTwoProject"
   cd "$PROJECT_DIR"

   DATA_ROOT="$PROJECT_DIR/data"
   DATASET="food101"
   TEST_DATASET_KEY="${DATASET}_test"
   EXPERIMENT_ROOT="$DATA_ROOT/experiments/${DATASET}_centered_v2_knn"
   METRICS_EXPERIMENT_DIR="$EXPERIMENT_ROOT/store_metrics"
   DOWNSTREAM_JSON="$DATA_ROOT/downstream/${DATASET}_mlp.json"
   PLOTS_EXT="png,svg"
   PLOTS_MODE="alltasks"
   ```

2. Посчитать только одну часть исходных моделей для локальных отображений. Это удобно для параллельного ручного запуска нескольких одинаковых команд с разными значениями `SOURCE_SHARD_INDEX`:

   ```bash
   LOCAL_MAPS_ROOT="$DATA_ROOT/local_maps"
   TEST_EMB_DIR="$DATA_ROOT/embeddings/$TEST_DATASET_KEY"
   MODELS="primary"
   LOCAL_GEOMETRY_MODE="centered_offsets_v2"
   K_LIST="5,10,20,40,80"
   RFF_K_LIST=""
   EPS_PERCENTILES=""
   WEIGHTED_EPS_PERCENTILES=""
   WEIGHTED_EPS_SCALE="3.0"
   SOURCE_SHARD_INDEX="0"
   SOURCE_SHARD_COUNT="21"

   python -m scripts.run_compute_local_map_store \
     --embeddings_dir "$TEST_EMB_DIR" \
     --out_root "$LOCAL_MAPS_ROOT" \
     --dataset_key "$TEST_DATASET_KEY" \
     --models "$MODELS" \
     --k_list "$K_LIST" \
     --rff_k_list "$RFF_K_LIST" \
     --eps_percentiles "$EPS_PERCENTILES" \
     --weighted_eps_percentiles "$WEIGHTED_EPS_PERCENTILES" \
     --weighted_eps_scale "$WEIGHTED_EPS_SCALE" \
     --n_centers "200" \
     --seed "42" \
     --local_geometry_mode "$LOCAL_GEOMETRY_MODE" \
     --backend "cuda" \
     --map_dtype "float32" \
     --source_shard_index "$SOURCE_SHARD_INDEX" \
     --source_shard_count "$SOURCE_SHARD_COUNT" \
     --incremental
   ```

3. Посчитать одиночные метрики качества эмбеддингов для каждой модели:

   ```bash
   TEST_EMB_DIR="$DATA_ROOT/embeddings/$TEST_DATASET_KEY"
   SINGLE_METRICS_ROOT="$DATA_ROOT/single_metrics"
   SINGLE_METRICS="rankme stable_rank nesum pseudo_condition_number alpha_req"
   MODELS="primary"

   python -m scripts.run_compute_single_metrics \
     --embeddings_dir "$TEST_EMB_DIR" \
     --dataset_key "$TEST_DATASET_KEY" \
     --out_root "$SINGLE_METRICS_ROOT" \
     --models "$MODELS" \
     --metrics $SINGLE_METRICS \
     --device "auto"
   ```

4. Сравнить одиночные метрики с качеством моделей:

   ```bash
   SINGLE_METRICS_ROOT="$DATA_ROOT/single_metrics"
   SINGLE_EXPERIMENT_DIR="$EXPERIMENT_ROOT/single_metrics_eval"

   python -m scripts.run_evaluate_single_metrics \
     --dataset_key "$TEST_DATASET_KEY" \
     --single_metrics_root "$SINGLE_METRICS_ROOT" \
     --experiment_dir "$SINGLE_EXPERIMENT_DIR" \
     --downstream_json "$DOWNSTREAM_JSON" \
     --plots_mode "$PLOTS_MODE" \
     --plots_ext "$PLOTS_EXT" \
     --protocol "signed"
   ```

5. Построить общие диагностические графики по всем артефактам метрик:

   ```bash
   ARTIFACTS_DIR="$METRICS_EXPERIMENT_DIR/metric_matrices/$TEST_DATASET_KEY/artifacts"
   DIAGNOSTICS_DIR="$METRICS_EXPERIMENT_DIR/diagnostics/$TEST_DATASET_KEY"

   python -m scripts.run_diagnose_local_map \
     --artifacts_dir "$ARTIFACTS_DIR" \
     --out_dir "$DIAGNOSTICS_DIR/summary" \
     --degenerate_threshold "0.01" \
     --plots_ext "$PLOTS_EXT"
   ```

6. Построить диагностические графики для одной пары моделей:

   ```bash
   ARTIFACTS_DIR="$METRICS_EXPERIMENT_DIR/metric_matrices/$TEST_DATASET_KEY/artifacts"
   DIAGNOSTICS_DIR="$METRICS_EXPERIMENT_DIR/diagnostics/$TEST_DATASET_KEY"
   ARTIFACT_NAME="lin_k10_antisym_artifacts.npz"
   MODEL_A="resnet50"
   MODEL_B="vit_b_16"

   python -m scripts.run_diagnose_local_map \
     --artifacts_path "$ARTIFACTS_DIR/$ARTIFACT_NAME" \
     --model_a "$MODEL_A" \
     --model_b "$MODEL_B" \
     --out_dir "$DIAGNOSTICS_DIR/pair_${MODEL_A}_${MODEL_B}" \
     --degenerate_threshold "0.01" \
     --plots_ext "$PLOTS_EXT"
   ```

7. Сравнить режимы центрирования по уже готовым артефактам. Новый расчёт локальных отображений эта команда не запускает:

   ```bash
   ABS_EXPERIMENT_DIR="$DATA_ROOT/experiments/${DATASET}_absolute_coords_v0/store_metrics"
   CV2_EXPERIMENT_DIR="$DATA_ROOT/experiments/${DATASET}_centered_v2_knn/store_metrics"
   CENTERING_OUT_DIR="$DATA_ROOT/experiments/${DATASET}_centering_comparison"
   CENTERING_ARTIFACT_NAME="lin_k10_antisym_artifacts.npz"

   python -m scripts.plot_centering_comparison_diagnostics \
     --experiment "absolute_coords_v0=$ABS_EXPERIMENT_DIR" \
     --experiment "centered_offsets_v2=$CV2_EXPERIMENT_DIR" \
     --dataset_key "$TEST_DATASET_KEY" \
     --artifact_name "$CENTERING_ARTIFACT_NAME" \
     --out_dir "$CENTERING_OUT_DIR" \
     --plots_ext "$PLOTS_EXT"
   ```

</details>

<details>
  <summary><b>▶ Расчёт без сохранённых локальных отображений</b></summary>

1. Задать рабочую папку проекта и основные пути:

   ```bash
   PROJECT_DIR="$HOME/YarTwoProject"
   cd "$PROJECT_DIR"

   DATA_ROOT="$PROJECT_DIR/data"
   DATASET="cifar10"
   TEST_DATASET_KEY="${DATASET}_test"
   TEST_EMB_DIR="$DATA_ROOT/embeddings/$TEST_DATASET_KEY"
   DOWNSTREAM_JSON="$DATA_ROOT/downstream/${DATASET}_mlp.json"
   EXPERIMENT_ROOT="$DATA_ROOT/experiments/${DATASET}_direct_metrics"
   ```

2. Задать список метрик и параметры расчёта:

   ```bash
   INCLUDE_METRICS="directed_k10,lin_k10_antisym,lin_k5_antisym,lin_k20_antisym,lin_k40_antisym,lin_eps_5_antisym,lin_eps_10_antisym,lin_eps_20_antisym,multiscale_mean_antisym,rff_k10_antisym"
   SEED="42"
   LOCAL_GEOMETRY_MODE="centered_offsets_v2"
   METRICS_DIR="$EXPERIMENT_ROOT/metric_matrices/$TEST_DATASET_KEY"
   REPORTS_DIR="$EXPERIMENT_ROOT/reports"
   PLOTS_DIR="$EXPERIMENT_ROOT/plots/$TEST_DATASET_KEY"
   ```

3. Посчитать парные метрики напрямую по эмбеддингам:

   ```bash
   python -m scripts.run_compute_embedding_metrics \
     --embeddings_dir "$TEST_EMB_DIR" \
     --out_dir "$METRICS_DIR" \
     --include "$INCLUDE_METRICS" \
     --seed "$SEED" \
     --local_geometry_mode "$LOCAL_GEOMETRY_MODE" \
     --incremental
   ```

4. Оценить метрики по качеству моделей:

   ```bash
   python -m scripts.run_evaluate_metrics \
     --metrics_dir "$METRICS_DIR" \
     --downstream_json "$DOWNSTREAM_JSON" \
     --out_csv "$REPORTS_DIR/${DATASET}_eval_signed.csv" \
     --eval_protocol "delta_signed" \
     --plots_dir "$PLOTS_DIR/scatter" \
     --plots_mode "alltasks" \
     --plots_ext "png,svg"
   ```

5. Построить итоговый график сравнения метрик:

   ```bash
   python -m scripts.plot_metric_summary \
     --eval_csv "$REPORTS_DIR/${DATASET}_eval_signed.csv" \
     --out_dir "$EXPERIMENT_ROOT" \
     --dataset "$DATASET" \
     --protocol "Δacc" \
     --out_name "metrics_summary_antisym.png"
   ```

</details>
<p align="right">(<a href="#readme-top">Вернуться к началу</a>)</p>

# Лицензия

Распространяется по лицензии MIT. Дополнительную информацию см. в файле [`LICENSE`][license-url].

<p align="right">(<a href="#readme-top">Вернуться к началу</a>)</p>

# Контакты

Ссылка на проект: [https://github.com/GandlinAlexandr/embedding-space-comparison](https://github.com/GandlinAlexandr/embedding-space-comparison)

<p align="right">(<a href="#readme-top">Вернуться к началу</a>)</p>

<!-- Раздел ссылок на сайты и миниатюры -->

[Python-url]: https://python.org/
[Python.org]: https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=blue

[Pandas-url]: https://pandas.pydata.org/
[Рandas.pydata.org]: https://img.shields.io/badge/Pandas-2C2D72?style=for-the-badge&logo=pandas&logoColor=white

[Numpy-url]: https://numpy.org/
[Numpy.org]: https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white

[Colab-url]: https://colab.research.google.com/
[Colab]: https://img.shields.io/badge/Colab-F9AB00?style=for-the-badge&logo=googlecolab&color=525252

[scikit-learn-url]: https://scikit-learn.org/
[scikit-learn]: https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white

[Matplotlib-url]: https://matplotlib.org/
[Matplotlib.org]: https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black

[Pytorch-url]: https://pytorch.org/
[Pytorch]: https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white

[SciPy-url]: https://scipy.org/
[SciPy]: https://img.shields.io/badge/SciPy-654FF0?logo=SciPy&logoColor=white&style=for-the-badge
