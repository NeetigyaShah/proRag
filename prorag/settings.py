"""Every tunable lives here, nowhere else.

Configuration layer: all tunables as pydantic-settings fields, loaded once at
import time from .env / environment variables (module-level `settings`), then
read at boot and per-request across the app — DB, models, ingest limits,
retrieval, budgets, auth, and the scheduler.
"""

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All tunables, read from .env / environment at import (see the
    module-level `settings` instance every module imports)."""

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
    # Prefill agent (query refinement): a tiny free OpenRouter model that
    # cleans typos and expands the user's prompt *before* planning, so hybrid
    # retrieval matches intent. It never adds facts — only rewrites what's
    # already there — and any failure falls back to the raw prompt.
    prefill_enabled: bool = True
    prefill_model: str = "openrouter/google/gemma-4-26b-a4b-it:free"
    prefill_max_tokens: int = 160
    prefill_timeout: float = 8.0
    # OpenRouter key for the direct /embeddings call (llm.py). Lives here so a
    # .env-only setup works from any shell — pydantic-settings reads .env into
    # this field but does NOT export it to os.environ, and llm.py used to read
    # os.environ directly (fine when the key was a real env var, a KeyError
    # when it was only in .env).
    openrouter_api_key: str | None = None

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
    # "api" = OpenRouter's hosted cross-encoder rerank endpoint
    # (https://openrouter.ai/api/v1/rerank — real cross-encoder scores,
    # ~$0.001/call, ~1.5s for 40 chunks); "local" = sentence-transformers
    # cross-encoder on this machine. API is the default: the free LLM scorer
    # it replaced was quota-capped (50 req/day) and scored unreliably.
    rerank_backend: str = "api"
    rerank_model: str = "BAAI/bge-reranker-v2-m3"  # local backend only
    # OpenRouter rerank model (backend "api"). Confirmed live with the
    # project key: cohere/rerank-v3.5.
    rerank_api_model: str = "cohere/rerank-v3.5"
    rerank_api_timeout: float = 20.0
    # When the API call fails (network, provider outage), degrade to the
    # local cross-encoder instead of silently skipping the rerank.
    rerank_api_fallback_to_local: bool = True
    rerank_top_n: int = 40  # fused hits sent into the reranker
    crop_min_docs: int = 3
    crop_max_docs: int = 12
    crop_max_chunks_per_doc: int = 3  # sections of one PDF all answerable
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

    # #4, #24: access-rule preview/confirm/auto-admission. Cosine similarity
    # a chunk's embedding must clear (against the rule's nl_query embedding)
    # for its document to count as a rule match. Tunable: raise it to make
    # rules stricter (fewer false-positive grants), lower it to widen recall.
    rule_similarity_floor: float = 0.35

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
