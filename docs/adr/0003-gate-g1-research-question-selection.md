# ADR-0003: Gate G1 — primary and backup research question selection

- **Status**: accepted (Gate G1 executed by @SakkarinKt via rubric review + merge of this PR,
  2026-07-20)
- **Deciders**: @SakkarinKt (rubric review + selection), Claude (proposal + pre-filled evidence)

## Decision

1. **Primary question: proposal `0001`**
   (`docs/proposals/0001-direct-nonstationarity-measurement.md`) — isolating and directly measuring
   co-learning non-stationarity via the freeze intervention, compared across world-model sharing
   topologies. Rubric verdict: **6/6 pass** (per-item evidence in the Gate-G1 PR body):
   one-sentence falsifiable hypothesis; novelty grounded in `docs/proposals/gap-analysis.md` §2–§3
   over an n=8 empirical lit map; laptop-sized (≤200K steps/arm/seed, ≈40 overnight CPU runs);
   seeds + variance stated (3-seed instrument milestone, ≥5-seed full sweep); arms treated equally
   via fixed env/backbone/compute — with explicit per-arm hyperparameter-tuning equalization to be
   enforced at the experiment-run co-sign template before any sweep; falsifiable kill criteria,
   including the two-sided instrument gate.
2. **Backup question: `0001`'s built-in comms-content pivot** (its kill criteria, second bullet):
   if post-freeze error growth is statistically indistinguishable across all four topology arms,
   the independent variable becomes communication *content* — compressed self-intention (plan)
   versus raw latent state — holding topology fixed at Arm B (peer comms). Anchors:
   `plans-not-percepts-2025`, `codreamer-2024`, gap-analysis §1.3. The backup reuses the entire
   apparatus (environment, backbone, freeze instrument, metric); only the independent variable
   changes.
3. **Recorded deviation from plan**: Phase 1 targeted 3–5 proposals; exactly one was drafted.
   Accepted because the instrument-first design makes primary and backup share one apparatus (the
   backup is a de-risked pivot, not a second bet); the L2 Arm-A carve-out has already validated
   feasibility of that apparatus; and Phase 2 is scheduled for weeks 3–7 with week 3 starting now —
   drafting alternatives the selected question doesn't need would trade schedule for breadth. The
   undrafted second candidate (policy-execution scope as the independent variable, gap-analysis §5)
   stays on file, unranked.

## Consequences

- Phase 2 opens: `PLAN.html`, `loop/GOAL.md` (v2 — L2 Phase-2 vertical-slice scope), and
  `README.md` are updated in the same PR.
- Proposal `0001`'s status moves `draft` → `selected-primary`, and its kill-criteria pivot is
  annotated as the designated G1 backup.
- If primary **and** backup both die by their kill criteria, Phase 1 reopens to draft the
  gap-analysis §5 candidate and G1 re-runs — no silent question swap.
