"""Every tunable lives here, nowhere else."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql+asyncpg://prorag:prorag@localhost:5432/prorag"

    db_pool_size: int = 10
    db_max_overflow: int = 20
    db_pool_timeout: float = 30.0
    db_connect_timeout: float = 10.0

    embed_model: str = "text-embedding-3-small"
    embed_dim: int = 1536
    reasoning_enabled: bool = False  # OpenRouter reasoning models: think before answering
    answer_model: str = "gpt-4o-mini"
    planner_model: str = "gpt-4o-mini"  # unused until Phase 2's planner exists

    blob_dir: str = "./blobs"

    # Ingestion limits — ceilings so one pathological upload can't exhaust the
    # process. Tune per deployment; these suit a single-box side project.
    max_upload_bytes: int = 100 * 1024 * 1024  # 100 MB
    max_chunks_per_document: int = 20_000
    embed_batch_size: int = 64  # texts per embedding request
    embed_batch_concurrency: int = 4  # batches in flight at once
    embed_timeout_seconds: float = 600.0  # whole embed phase, all batches

    chunk_target_tokens: int = 700
    chunk_overlap_tokens: int = 100

    search_top_k: int = 5

    # Phase 2: hybrid + rerank
    rerank_enabled: bool = True
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    rerank_top_n: int = 40  # fused hits sent into the reranker
    crop_min_docs: int = 3
    crop_max_docs: int = 12
    crop_score_gap: float = 0.15  # dynamic floor = max(top_score - gap, crop_score_floor)
    crop_score_floor: float = 0.0
    crop_token_budget: int = 6000

    # Phase 3: structured retrieval arm
    structured_weight: float = 1.2  # RRF arm weight (§4.3) — vector 1.0, fts 1.0, structured 1.2

    # Phase 5: operations
    auth_enabled: bool = False  # off by default so the local UI keeps working without keys
    daily_cost_cap_usd: float = 5.0
    eval_timeout_seconds: float = 900.0  # whole golden-set run; ~50 questions × LLM latency
    # Must stay well under both db_pool_timeout and the orchestrator's own probe
    # timeout, or a saturated pool reads as a hang instead of a 503.
    readyz_timeout_seconds: float = 5.0
    fallback_price_per_1k_usd: float = 0.002  # used only when litellm has no price entry for a model


settings = Settings()
