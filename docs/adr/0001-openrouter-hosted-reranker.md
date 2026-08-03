# ADR 0001: OpenRouter hosted cross-encoder rerank

- Status: accepted (2026-08-04)
- Deciders: Neetigya Shah

## Context

The reranker scored (query, chunk) pairs with a free OpenRouter LLM
(`gemma-4-26b-a4b-it:free`): one JSON of 0–10 scores per call. Free tier is
capped at ~50 req/day, timed out (the 8s stall), and produced flat,
unreliable scores that the crop treated as gospel.

## Decision

Call OpenRouter's hosted rerank endpoint (`POST /api/v1/rerank`) with
`cohere/rerank-v3.5` — a real cross-encoder, ~$0.001/call, ~1.5s for 40
chunks — via a direct httpx call (litellm cannot route this endpoint).
`RERANK_BACKEND=api` is the default; the local sentence-transformers
cross-encoder remains as a fallback when the API fails.

## Consequences

- Scores are real cross-encoder relevance, 0..1, comparable across queries.
- Cost is negligible; usage rows are not tracked for rerank (parity with the
  old path, which never recorded them either).
- The free LLM scorer and its JSON parsing were deleted.
