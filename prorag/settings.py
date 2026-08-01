"""Every tunable lives here, nowhere else."""

from pydantic import field_validator
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
    daily_cost_cap_usd: float = 5.0  # install-wide hard cap (unchanged, #9's resolution)

    # #21: per-user budgets on top of the install-wide cap above. Soft at 1x
    # (request still runs, response carries a warning), hard at the multiplier
    # (429). Overridable per user via users.daily_cap_usd_override (0010).
    user_daily_cap_usd: float = 1.0
    user_hard_cap_multiplier: float = 2.0

    # Phase 6: sessions + OIDC (#19)
    session_cookie_secure: bool = True  # set false in .env for local http:// dev
    session_ttl_days: int = 7
    oidc_issuer: str | None = None
    oidc_client_id: str | None = None
    oidc_client_secret: str | None = None
    oidc_redirect_path: str = "/auth/oidc/callback"
    eval_timeout_seconds: float = 900.0  # whole golden-set run; ~50 questions × LLM latency
    # Must stay well under both db_pool_timeout and the orchestrator's own probe
    # timeout, or a saturated pool reads as a hang instead of a 503.
    readyz_timeout_seconds: float = 5.0
    fallback_price_per_1k_usd: float = 0.002  # used only when litellm has no price entry for a model

    # #23: connector polling scheduler, on top of #22's sync engine and #15's
    # cadence resolution (15 min incremental, mandatory nightly sweep).
    scheduler_enabled: bool = True
    connector_poll_seconds: int = 900
    connector_sweep_hours: int = 24

    # ---- sanity checks (#20) ---------------------------------------------------
    # Deliberately narrow: obviously-nonsensical values that would otherwise fail
    # confusingly deep inside a request instead of at boot. Not exhaustive —
    # EMBED_DIM-vs-index mismatch etc. is `prorag doctor`'s job, not startup's.

    @field_validator("daily_cost_cap_usd")
    @classmethod
    def _daily_cost_cap_must_be_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"daily_cost_cap_usd must be > 0, got {v}")
        return v

    @field_validator("session_ttl_days")
    @classmethod
    def _session_ttl_must_be_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError(f"session_ttl_days must be > 0, got {v}")
        return v

    @field_validator("blob_dir")
    @classmethod
    def _blob_dir_must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("blob_dir must not be empty")
        return v


settings = Settings()
