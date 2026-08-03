"""Parameter sweep harness (§8 Phase 6): runs the golden eval set once per
combination of retrieval-tunables, prints a ranked table. No code edits per
run — everything is a `settings` attribute, overridden in-process for the
duration of one combination and restored after.

Chunk `target_tokens` is deliberately NOT swept here: chunk size is an
ingest-time decision (re-chunking means re-ingesting every document), not a
per-request knob like the others — see ARCHITECTURE.md §8 Phase 6.

Usage (needs a live Postgres + LLM keys, same as the app):
    uv run python scripts/sweep.py
    uv run python scripts/sweep.py --golden path/to/your_golden.jsonl
"""

import argparse
import asyncio
import itertools
from pathlib import Path

from prorag.db import SessionLocal
from prorag.eval.runner import DEFAULT_GOLDEN_PATH, run_eval
from prorag.settings import settings

# One value list per settings attribute to sweep. Add/remove keys freely —
# generate_combinations() is a plain cartesian product over whatever's here.
SWEEP_GRID: dict[str, list] = {
    "rerank_top_n": [20, 40],
    "crop_score_floor": [0.01, 0.02, 0.05],
    "crop_max_docs": [8, 12],
    "structured_weight": [1.0, 1.2],
}


def generate_combinations(grid: dict[str, list]) -> list[dict]:
    """Cartesian product of a {setting_name: [values]} grid -> one dict per
    combination, e.g. {"rerank_top_n": 40, "crop_score_gap": 0.15, ...}."""
    if not grid:
        return [{}]
    keys = list(grid)
    return [dict(zip(keys, values, strict=True)) for values in itertools.product(*(grid[k] for k in keys))]


async def _run_one(golden_path: Path, overrides: dict) -> dict:
    originals = {k: getattr(settings, k) for k in overrides}
    for k, v in overrides.items():
        setattr(settings, k, v)
    try:
        async with SessionLocal() as session:
            result = await run_eval(session, golden_path=golden_path)
    finally:
        for k, v in originals.items():
            setattr(settings, k, v)
    return result["aggregate"]


def _score(aggregate: dict) -> float:
    """Ranking key: mean of the three always-present deterministic metrics.
    Ragas scores (when present) aren't comparable in scale, so they're printed
    but not folded into the ranking key."""
    return (aggregate["hit_rate"] + aggregate["keyword_coverage"] + aggregate["citation_validity"]) / 3


async def main_async(golden_path: Path) -> None:
    combos = generate_combinations(SWEEP_GRID)
    rows = []
    for combo in combos:
        aggregate = await _run_one(golden_path, combo)
        rows.append((combo, aggregate))

    rows.sort(key=lambda r: _score(r[1]), reverse=True)

    print(f"{'score':>7}  {'hit_rate':>9}  {'coverage':>9}  {'citation':>9}  combo")
    for combo, aggregate in rows:
        print(
            f"{_score(aggregate):7.3f}  {aggregate['hit_rate']:9.3f}  "
            f"{aggregate['keyword_coverage']:9.3f}  {aggregate['citation_validity']:9.3f}  {combo}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN_PATH)
    args = parser.parse_args()
    asyncio.run(main_async(args.golden))


if __name__ == "__main__":
    main()
