# ADR 0002: Flatness guard and floor-only crop

- Status: accepted (2026-08-04)
- Deciders: Neetigya Shah

## Context

For "tell me about neetigy shahs skills", the reranker scored the top six
chunks within 0.005 of each other and demoted the correct chunk (the resume
skills section, #2 in the fused order) to #5, letting irrelevant book pages
(p.289 = RL/MCMC) win the citation lottery. Cohere's scores are also erratic
run-to-run: one run spikes a single chunk to 0.37, another run is flat at
0.05. The crop's dynamic gap (`top_score - 0.15`) then either admits junk
(floor 0.0) or — with a spike — starves every relevant chunk below 0.13.

## Decision

Two mechanisms:

1. **Flatness guard** (`rerank_flat_spread = 0.03`): if the reranker's five
   best scores span less than the spread, its ordering is noise — keep the
   pre-rerank fused order. The real score multiset is still attached, but
   assigned in descending order down the fused order: the crop and the UI
   both sort by score, so the scores must encode the fused order or the
   sort would silently undo it. Measured on the top-5 *by score*, so an
   outlier (junk doc at 0.010) can't mask a flat top.
2. **Floor-only crop**: the dynamic spike gap is removed. The crop keeps
   everything above a hard `crop_score_floor = 0.02`, bounded by the
   existing caps (12 docs, 3 chunks/doc, 6000 tokens). A spike can no longer
   starve legitimate chunks, and junk scores never enter context.

## Consequences

- Person queries answer from the person's documents again (fused order is
  correct in the failing cases).
- Context quality is bounded by an absolute relevance bar, not by one
  unreliable top score.
- A genuinely irrelevant-but-common score band (0.02–0.04) can still enter
  context if many chunks score there — accepted; the answer model's
  citation discipline is the last line of defense.
