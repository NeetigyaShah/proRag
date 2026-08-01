"""Golden-set eval runner (§8 Phase 6): loads `golden.jsonl`, runs the full
retrieve→answer path per question, and scores it two ways:

1. Cheap deterministic metrics, no ragas needed — always computed:
   - retrieval hit-rate: did an expected doc/title survive the crop?
   - answer keyword coverage: fraction of expected_answer_contains substrings
     present in the answer (case-insensitive)
   - citation validity: every [Sn] the model wrote resolves to a real,
     in-range source
2. ragas metrics (faithfulness, answer_relevancy, context_precision) — an
   OPTIONAL import. If ragas isn't installed, `ragas_available()` is False
   and the aggregate simply omits those keys with a logged warning, per
   §8 Phase 6 ("degrade gracefully").

Everything except run_eval() (which needs a DB session + live LLM) is a pure
function, unit-tested in tests/test_eval.py.
"""

import json
import logging
from pathlib import Path

from prorag.chat.citations import extract_cited_indices, normalize_citations
from prorag.models import EvalRun

logger = logging.getLogger(__name__)

DEFAULT_GOLDEN_PATH = Path(__file__).parent / "golden.jsonl"


def load_golden(path: str | Path = DEFAULT_GOLDEN_PATH) -> list[dict]:
    """Reads the checked-in golden set. One JSON object per line:
    {question, expected_answer_contains[], expected_doc_ids[]?,
     expected_source_titles[]?, notes}."""
    entries = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def retrieval_hit_rate(entry: dict, hits: list[dict]) -> float:
    """1.0 if any expected doc_id/title survived the crop into `hits`, else
    0.0. An entry with no expectations set at all (both lists empty) is
    excluded from scoring by returning None-like — callers should check
    has_retrieval_expectation() first; kept simple here (returns 1.0, i.e.
    vacuously satisfied) so aggregation never divides oddly."""
    expected_ids = {str(d) for d in entry.get("expected_doc_ids") or []}
    expected_titles = {t.lower() for t in entry.get("expected_source_titles") or []}
    if not expected_ids and not expected_titles:
        return 1.0

    for h in hits:
        if str(h.get("doc_id")) in expected_ids:
            return 1.0
        title = (h.get("title") or "").lower()
        if title and any(t in title for t in expected_titles):
            return 1.0
    return 0.0


def keyword_coverage(answer: str, expected_answer_contains: list[str]) -> float:
    """Fraction of expected substrings present in the answer, case-insensitive.
    Empty expectation list scores 1.0 (nothing to miss)."""
    if not expected_answer_contains:
        return 1.0
    lowered = answer.lower()
    hits = sum(1 for kw in expected_answer_contains if kw.lower() in lowered)
    return hits / len(expected_answer_contains)


def citation_validity(answer: str, num_sources: int) -> float:
    """1.0 if every [Sn] in the (already-normalized) answer resolves to a
    source that was actually in context; 1.0 on an answer with no citations
    at all (nothing invalid to find)."""
    normalized = normalize_citations(answer)
    cited = extract_cited_indices(normalized)
    if not cited:
        return 1.0
    valid = sum(1 for n in cited if 1 <= n <= num_sources)
    return valid / len(cited)


def aggregate_metrics(per_question: list[dict]) -> dict:
    """Mean of each deterministic metric across questions. Empty input -> zeros."""
    if not per_question:
        return {"hit_rate": 0.0, "keyword_coverage": 0.0, "citation_validity": 0.0, "n_questions": 0}
    n = len(per_question)
    return {
        "hit_rate": sum(q["hit_rate"] for q in per_question) / n,
        "keyword_coverage": sum(q["keyword_coverage"] for q in per_question) / n,
        "citation_validity": sum(q["citation_validity"] for q in per_question) / n,
        "n_questions": n,
    }


def ragas_available() -> bool:
    try:
        import ragas  # noqa: F401
    except ImportError:
        return False
    return True


async def compute_ragas_metrics(rows: list[dict]) -> dict | None:
    """rows: [{question, answer, contexts: [str], ground_truth?}]. Returns
    {faithfulness, answer_relevancy, context_precision} or None (with a
    logged warning) if ragas isn't installed — an optional import per §8
    Phase 6, never a hard dependency."""
    if not ragas_available():
        logger.warning("ragas not installed — skipping faithfulness/answer_relevancy/context_precision")
        return None

    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import answer_relevancy, context_precision, faithfulness

    dataset = Dataset.from_list(
        [
            {
                "question": r["question"],
                "answer": r["answer"],
                "contexts": r["contexts"],
                "ground_truth": r.get("ground_truth", ""),
            }
            for r in rows
        ]
    )
    result = evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_precision])
    return {k: float(v) for k, v in result.items()}


async def run_eval(session, golden_path: str | Path = DEFAULT_GOLDEN_PATH) -> dict:
    """Runs the full retrieve→answer path for every golden entry, scores it,
    persists an `eval_runs` row (per-question JSONB + aggregate), and returns
    the aggregate dict. Imports the chat retrieval path lazily to avoid a
    circular import (chat.router -> eval.runner would be the wrong direction)."""
    from prorag.chat.router import SYSTEM_PROMPT, build_prompt, retrieve
    from prorag.llm import answer

    golden = load_golden(golden_path)
    per_question: list[dict] = []
    ragas_rows: list[dict] = []

    for entry in golden:
        # user=None (default): the eval runner is #3's service-user, scored
        # against the full corpus, not a real principal's filtered view (#18).
        hits = await retrieve(session, entry["question"])
        user_prompt = build_prompt(entry["question"], hits)
        raw_answer = await answer(SYSTEM_PROMPT, user_prompt, session=session)

        per_question.append(
            {
                "question": entry["question"],
                "answer": raw_answer,
                "hit_rate": retrieval_hit_rate(entry, hits),
                "keyword_coverage": keyword_coverage(raw_answer, entry.get("expected_answer_contains") or []),
                "citation_validity": citation_validity(raw_answer, len(hits)),
            }
        )
        ragas_rows.append({"question": entry["question"], "answer": raw_answer, "contexts": [h["text"] for h in hits]})

    aggregate = aggregate_metrics(per_question)

    ragas_scores = await compute_ragas_metrics(ragas_rows)
    if ragas_scores is not None:
        aggregate.update(ragas_scores)
    else:
        aggregate["ragas"] = "skipped (ragas not installed)"

    run = EvalRun(questions=per_question, aggregate=aggregate)
    session.add(run)
    await session.commit()
    await session.refresh(run)

    return {"run_id": run.id, "aggregate": aggregate}
