"""Pydantic request/response schemas."""

import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class IngestResponse(BaseModel):
    doc_id: uuid.UUID
    # Every value that can reach here: ingest/router.py writes processing/ready/
    # failed, and models.py's column default is "pending" (reachable by any
    # insert that doesn't set status). This is a RESPONSE model, so a value
    # outside the set is a 500 — hence grounding it in the code rather than in
    # what the happy path happens to produce. Deliberately NOT applied to
    # Source.kind: those are prose/table_summary/table_window/row, and pinning
    # the wrong set would break every table citation.
    status: Literal["pending", "processing", "ready", "failed"]
    duplicate_of: uuid.UUID | None = None


class Source(BaseModel):
    n: int
    doc_id: uuid.UUID
    page: int | None
    file_url: str
    snippet: str
    score: float
    title: str | None = None
    kind: str | None = None
    bbox: list[float] | None = None


class ChatRequest(BaseModel):
    # Strip before length validation (Pydantic runs it in that order), so "   "
    # is rejected rather than embedded, sent to the LLM and billed. The web UI
    # already trims, but the API is the trust boundary — a direct POST doesn't.
    # It also lets " docs " satisfy the collection pattern instead of 422ing.
    model_config = ConfigDict(str_strip_whitespace=True)

    # Bounded so a giant paste can't blow up the prompt (and its cost) or sit in
    # memory; 8k chars is far beyond any real question.
    message: str = Field(min_length=1, max_length=8000)
    collection: str | None = Field(default=None, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    chat_id: uuid.UUID | None = None


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]
    message_id: uuid.UUID | None = None
    # Set when the caller is over their per-user soft cap but under the hard
    # one (#9's resolution, #21) — the answer still ran, this is advisory.
    budget_warning: str | None = None


class FeedbackRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    message_id: uuid.UUID
    rating: Literal["up", "down"]
    comment: str | None = Field(default=None, max_length=2000)


class EvalRunResponse(BaseModel):
    run_id: int
    aggregate: dict


class EvalRunDetail(EvalRunResponse):
    created_at: datetime
    questions: list[dict]


class ConnectorCreate(BaseModel):
    type: Literal["s3"]
    name: str = Field(min_length=1, max_length=255)
    # endpoint_url/bucket/prefix/access key id/secret, plus an optional
    # `collection` for where ingested docs land. Stored as plain JSONB v1
    # (prorag/models.py's Connector docstring notes the env-ref-indirection
    # upgrade path for the secret) — admin-only surface, so no redaction here.
    config: dict = Field(default_factory=dict)
    enabled: bool = True


class ConnectorUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=255)
    config: dict | None = None
    enabled: bool | None = None


class ConnectorOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    type: str
    name: str
    config: dict
    enabled: bool
    last_sync_at: datetime | None = None
    last_full_sweep_at: datetime | None = None
    last_error: str | None = None
    created_at: datetime


class SyncReport(BaseModel):
    new: int
    changed: int
    deleted: int
    skipped: int
    errors: int


class AccessRuleCreate(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    nl_query: str = Field(min_length=1, max_length=2000)
    group_id: uuid.UUID


class AccessRuleUpdate(BaseModel):
    # v1 ships confirm-once (admin/router.py refuses this once state is
    # 'confirmed') — see AccessRule's docstring in models.py.
    name: str | None = Field(default=None, min_length=1, max_length=255)
    nl_query: str | None = Field(default=None, min_length=1, max_length=2000)
    group_id: uuid.UUID | None = None


class AccessRuleOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    nl_query: str
    group_id: uuid.UUID
    state: Literal["draft", "confirmed"]
    created_at: datetime
    confirmed_at: datetime | None = None


class GroupCreate(BaseModel):
    name: str = Field(min_length=1, max_length=255)


class GroupOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    source: str
    external_id: str | None = None


class UserPatch(BaseModel):
    is_admin: bool | None = None
    disabled_at: datetime | None = None
    daily_cap_usd_override: float | None = None
