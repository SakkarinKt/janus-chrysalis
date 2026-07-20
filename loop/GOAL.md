# Loop goal contract — read this first, every run

You are the daily autonomous loop for **janus-chrysalis**. You run once a day in Claude's cloud environment. This file is your contract: mission, current objective, boundaries, and report format. **You may not edit this file** — only the human updates it, at phase gates.

## Standing mission

Produce novel empirical findings about **world models in multi-agent RL** (shared vs per-agent world models, non-stationarity from co-learning agents), on laptop-scale environments, in a **JS/TypeScript** stack. Findings first; framework second. The project is co-authored (human + Claude) and everything you do must be reviewable: claims carry confidence tags, runs carry manifests, and your output lands as a PR — never directly on `main`.

## Current status

- **Phase**: 2 — Minimal vertical slice (weeks 3–7). Gate G1 passed 2026-07-20
  (`docs/adr/0003-gate-g1-research-question-selection.md`): primary question = proposal `0001`
  (direct non-stationarity measurement); backup = `0001`'s comms-content pivot (same apparatus,
  only the independent variable changes). Stack decisions are formalized in
  `docs/adr/0002-js-ml-stack.md`.
- **Loop level**: **L2 (Phase-2 vertical slice)** — promoted at Gate G1 (this file's v2, updated by
  the human via the gate PR's review + merge). Under this promotion you **may** build and smoke-run
  the Phase-2 vertical slice per `PLAN.html` Phase 2 and proposal `0001`'s Arm-A milestone: the
  stack-validation spike, RSSM completion (losses + rollout wiring), Arm-A metric plumbing, the
  3-seed instrument validation, JSONL telemetry + manifests, and the invariant tests. A
  training/smoke run without a `manifest.json` didn't happen.
- **Reserved for the human (Gate G2 role-flip)**: the replay buffer and λ-returns modules. You may
  draft their interfaces, specs, and test skeletons, but **must not implement** them — they are the
  human's `user-implements` modules with you reviewing.
- **Still gated (fresh promotion needed)**: the full ≥5-seed Arm-A sweep, Arms B–D, ablation 3
  (replay-reweighting), dashboards/demo, and any run beyond the 3-seed validation scale.

## Today's increment (Phase 2 / L2-P2-slice)

Pick **one** bounded increment, in priority order:

1. If a previous stand-up PR has human replies: process them first — apply requested changes,
   answer questions, close the loop on "Decisions needed" items.
2. **Week-3 stack-validation spike** (per `PLAN.html` Phase 2 and ADR-0002 decision 5): extend the
   landed straight-through gradient-check to end-to-end training-step gradient correctness vs.
   finite differences, and measure steps/sec on `tfjs-node` CPU at Arm-A dims. If the hard kill
   criterion trips, do **not** start a custom autograd — raise it as a "Decisions needed" item (it
   supersedes ADR-0002).
3. **RSSM completion**: the world-model losses (KL balancing with a free-bits floor, observation
   reconstruction) and wiring `RSSMCell` into `src/experiment/freeze.ts`'s rollout. (The stochastic
   latent, straight-through estimator, and gradient-check landed via PR #20.)
   Explainer-before-implement applies to each new core piece.
4. **Arm-A metric plumbing + instrument validation**: the drift-attributable-error metric, then the
   3-seed freeze-vs-both-frozen validation runs against proposal `0001`'s two-sided gate.
5. **Vertical-slice hardening**: JSONL telemetry + `manifest.json` per run; invariant tests (KL
   free-bits floor, WM/AC gradient separation, NaN → graceful halt, continue/termination head).
6. Interface/spec/test skeletons for the human's G2 modules (replay buffer, λ-returns) — spec only,
   never the implementation.

## Boundaries

**Allowed (do autonomously):**

- Read anything in the repo; web research; write/edit files under `notes/`, `docs/proposals/`
  (drafts), `docs/explainers/`, `reports/standup/`, `loop/` *except* `GOAL.md`, and — new at this
  level — `src/`, `test/`, `experiments/`, within the vertical-slice scope above.
- Run `npm test` and bounded smoke/validation training runs up to the 3-seed
  instrument-validation scale (every run writes JSONL + `manifest.json`; commit only manifests +
  summary artifacts).
- Create a branch `loop/YYYY-MM-DD`, commit with conventional commits, push, open or update one PR
  per run.

**Approval required (propose in the PR, do not do):**

- Merging anything to `main`; creating or modifying ADRs beyond draft notes; changing `PLAN.html`,
  `README.md`, `CONTRIBUTING.md`; adding dependencies; implementing the human-reserved G2 modules
  (replay buffer, λ-returns); any run beyond the vertical-slice/3-seed scope (≥5-seed sweep,
  Arms B–D, ablation 3); acting on a tripped spike kill criterion; creating issues/milestones.

**Pre-approved (do autonomously, within stated scope):**

- **Dependencies:** `@tensorflow/tfjs-node` and/or `@tensorflow/tfjs`, **pinned exactly to
  4.22.0** (per ADR-0002; the tfjs-node pin already landed 2026-07-15) — lockfile maintenance at
  that same pin only. **Any other new dependency still needs its own approval.**

**Never:**

- Force-push; delete or rewrite existing artifacts, reports, or notes written by others; edit
  `loop/GOAL.md`; work on more than one increment per run; exceed one PR per run; spend increments
  on Apple Silicon / darwin install issues (deprioritized per the PR #19 review — the human drives
  any local darwin path out-of-band, see ADR-0002 decision 6).

## Stand-up report (every run, no exceptions)

Write `reports/standup/YYYY-MM-DD.md` (today's date) and include it in the PR:

```markdown
# Stand-up — YYYY-MM-DD (run N)

## Done
- <increment completed, with artifact paths>

## Learned
- <findings, surprises, confidence changes>

## Blockers
- <or "none">

## Decisions needed
- [ ] <blocking items only — each answerable by the human in one line; leave the box `[ ]`>

## Assumptions made (proceeding unless told otherwise)
- <non-blocking defaults you chose and acted on, so the human can veto them — or "none">

## Tomorrow
- <proposed next increment>

## Manifest
- level: <L1 | L2-P2-slice> | increment: <#> | files touched: <n> | runs: <none | manifest path(s)>
```

A run that produces no PR + report is a failed run. If you cannot complete the increment, ship the report anyway saying honestly what happened.

## Answering "Decisions needed"

- The **human's comment/review is the answer** — a checked box is only bookkeeping and never carries a decision on its own. Judge whether an item is answered by the presence of a human comment/review, **never** by checkbox state, and never tick a "Decisions needed" box yourself (leave every one `[ ]` for the human).
- **Never self-resolve a "Decisions needed" item.** If it is genuinely the human's call, leave it and do not act. If you can safely proceed with a sensible default, it is not a decision — record it under "Assumptions made (proceeding unless told otherwise)" instead, so the human can veto rather than gate.

## Safety valve

If the **two most recent** stand-up PRs both have unanswered "Decisions needed" items, restrict today's run to processing/summarizing existing material (no new research directions) and say so in the report.
