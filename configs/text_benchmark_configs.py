from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class TextEmbeddingModelSpec:
    model_id: str
    hf_name: str
    family: str
    prompt_prefix: str = ""
    pooling: str = "mean"
    trust_remote_code: bool = False


TEXT_EMBEDDING_MODEL_SPECS: List[TextEmbeddingModelSpec] = [
    TextEmbeddingModelSpec(
        model_id="st_all_minilm_l6_v2",
        hf_name="sentence-transformers/all-MiniLM-L6-v2",
        family="sbert",
    ),
    TextEmbeddingModelSpec(
        model_id="st_all_minilm_l12_v2",
        hf_name="sentence-transformers/all-MiniLM-L12-v2",
        family="sbert",
    ),
    TextEmbeddingModelSpec(
        model_id="st_all_mpnet_base_v2",
        hf_name="sentence-transformers/all-mpnet-base-v2",
        family="sbert",
    ),
    TextEmbeddingModelSpec(
        model_id="st_paraphrase_minilm_l6_v2",
        hf_name="sentence-transformers/paraphrase-MiniLM-L6-v2",
        family="sbert",
    ),
    TextEmbeddingModelSpec(
        model_id="st_paraphrase_mpnet_base_v2",
        hf_name="sentence-transformers/paraphrase-mpnet-base-v2",
        family="sbert",
    ),
    TextEmbeddingModelSpec(
        model_id="e5_small_v2",
        hf_name="intfloat/e5-small-v2",
        family="e5",
        prompt_prefix="query: ",
    ),
    TextEmbeddingModelSpec(
        model_id="e5_base_v2",
        hf_name="intfloat/e5-base-v2",
        family="e5",
        prompt_prefix="query: ",
    ),
    TextEmbeddingModelSpec(
        model_id="e5_large_v2",
        hf_name="intfloat/e5-large-v2",
        family="e5",
        prompt_prefix="query: ",
    ),
    TextEmbeddingModelSpec(
        model_id="multilingual_e5_small",
        hf_name="intfloat/multilingual-e5-small",
        family="e5",
        prompt_prefix="query: ",
    ),
    TextEmbeddingModelSpec(
        model_id="multilingual_e5_base",
        hf_name="intfloat/multilingual-e5-base",
        family="e5",
        prompt_prefix="query: ",
    ),
    TextEmbeddingModelSpec(
        model_id="bge_small_en_v15",
        hf_name="BAAI/bge-small-en-v1.5",
        family="bge",
    ),
    TextEmbeddingModelSpec(
        model_id="bge_base_en_v15",
        hf_name="BAAI/bge-base-en-v1.5",
        family="bge",
    ),
    TextEmbeddingModelSpec(
        model_id="bge_large_en_v15",
        hf_name="BAAI/bge-large-en-v1.5",
        family="bge",
    ),
    TextEmbeddingModelSpec(
        model_id="bge_m3",
        hf_name="BAAI/bge-m3",
        family="bge",
    ),
    TextEmbeddingModelSpec(
        model_id="gte_small",
        hf_name="thenlper/gte-small",
        family="gte",
    ),
    TextEmbeddingModelSpec(
        model_id="gte_base",
        hf_name="thenlper/gte-base",
        family="gte",
    ),
    TextEmbeddingModelSpec(
        model_id="gte_large",
        hf_name="thenlper/gte-large",
        family="gte",
    ),
    TextEmbeddingModelSpec(
        model_id="mxbai_embed_large_v1",
        hf_name="mixedbread-ai/mxbai-embed-large-v1",
        family="mxbai",
    ),
    TextEmbeddingModelSpec(
        model_id="st_multi_qa_minilm_l6_cos_v1",
        hf_name="sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
        family="sbert",
    ),
    TextEmbeddingModelSpec(
        model_id="e5_small",
        hf_name="intfloat/e5-small",
        family="e5",
        prompt_prefix="query: ",
    ),
    TextEmbeddingModelSpec(
        model_id="bge_small_en",
        hf_name="BAAI/bge-small-en",
        family="bge",
    ),
    TextEmbeddingModelSpec(
        model_id="bge_base_en",
        hf_name="BAAI/bge-base-en",
        family="bge",
    ),
    TextEmbeddingModelSpec(
        model_id="snowflake_arctic_embed_xs",
        hf_name="Snowflake/snowflake-arctic-embed-xs",
        family="snowflake",
        pooling="cls",
    ),
    TextEmbeddingModelSpec(
        model_id="snowflake_arctic_embed_s",
        hf_name="Snowflake/snowflake-arctic-embed-s",
        family="snowflake",
        pooling="cls",
    ),
    TextEmbeddingModelSpec(
        model_id="snowflake_arctic_embed_m",
        hf_name="Snowflake/snowflake-arctic-embed-m",
        family="snowflake",
        pooling="cls",
    ),
    TextEmbeddingModelSpec(
        model_id="snowflake_arctic_embed_m_v15",
        hf_name="Snowflake/snowflake-arctic-embed-m-v1.5",
        family="snowflake",
        pooling="cls",
    ),
    TextEmbeddingModelSpec(
        model_id="snowflake_arctic_embed_l",
        hf_name="Snowflake/snowflake-arctic-embed-l",
        family="snowflake",
        pooling="cls",
    ),
]


TEXT_EMBEDDING_MODEL_BY_ID: Dict[str, TextEmbeddingModelSpec] = {
    spec.model_id: spec for spec in TEXT_EMBEDDING_MODEL_SPECS
}


TEXT_EMBEDDING_MODEL_IDS: List[str] = [
    spec.model_id for spec in TEXT_EMBEDDING_MODEL_SPECS
]


TEXT_EMBEDDING_MODEL_IDS_CSV: str = ",".join(TEXT_EMBEDDING_MODEL_IDS)


TEXT_EMBEDDING_CPU_MODEL_IDS: List[str] = [
    "st_all_minilm_l6_v2",
    "st_all_minilm_l12_v2",
    "st_all_mpnet_base_v2",
    "st_paraphrase_minilm_l6_v2",
    "st_paraphrase_mpnet_base_v2",
    "st_multi_qa_minilm_l6_cos_v1",
    "e5_small_v2",
    "e5_base_v2",
    "e5_large_v2",
    "multilingual_e5_small",
    "multilingual_e5_base",
    "e5_small",
    "bge_small_en_v15",
    "bge_base_en_v15",
    "bge_large_en_v15",
    "bge_m3",
    "bge_small_en",
    "bge_base_en",
]


TEXT_EMBEDDING_CPU_MODEL_IDS_CSV: str = ",".join(TEXT_EMBEDDING_CPU_MODEL_IDS)


TEXT_EMBEDDING_SNOWFLAKE_MODEL_IDS: List[str] = [
    "snowflake_arctic_embed_xs",
    "snowflake_arctic_embed_s",
    "snowflake_arctic_embed_m",
    "snowflake_arctic_embed_m_v15",
    "snowflake_arctic_embed_l",
]


TEXT_EMBEDDING_TEXT20_MODEL_IDS: List[str] = [
    "st_all_minilm_l6_v2",
    "st_all_minilm_l12_v2",
    "st_all_mpnet_base_v2",
    "st_paraphrase_minilm_l6_v2",
    "st_paraphrase_mpnet_base_v2",
    "e5_small_v2",
    "e5_base_v2",
    "e5_large_v2",
    "multilingual_e5_small",
    "multilingual_e5_base",
    "bge_small_en_v15",
    "bge_base_en_v15",
    "bge_large_en_v15",
    "bge_m3",
    "bge_base_en",
    *TEXT_EMBEDDING_SNOWFLAKE_MODEL_IDS,
]


TEXT_EMBEDDING_SNOWFLAKE_MODEL_IDS_CSV: str = ",".join(
    TEXT_EMBEDDING_SNOWFLAKE_MODEL_IDS
)


TEXT_EMBEDDING_TEXT20_MODEL_IDS_CSV: str = ",".join(TEXT_EMBEDDING_TEXT20_MODEL_IDS)


TEXT_EMBEDDING_FAMILY_MAP: Dict[str, str] = {
    spec.model_id: spec.family for spec in TEXT_EMBEDDING_MODEL_SPECS
}
