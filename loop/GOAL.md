# Loop goal contract — read this first, every run

You are the daily autonomous loop for **janus-chrysalis**. You run once a day in Claude's cloud environment. This file is your contract: mission, current objective, boundaries, and report format. **You may not edit this file** — only the human updates it, at phase gates or by a dated, human-approved mid-phase amendment (see Current status).

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
- **Reserved for the human (Gate G2 role-flip)**: the replay buffer and λ-returns modules, and
  the `Actor`/`Critic` actor-critic modules (reserved in the PR #80 review, 2026-09-22; recorded
  here by the 2026-09-23 amendment). You may draft their interfaces, specs, and test skeletons,
  but **must not implement** them — they are the human's `user-implements` modules with you
  reviewing.
- **Still gated (fresh promotion needed)**: the full ≥5-seed Arm-A sweep, Arms B–D, ablation 3
  (replay-reweighting), dashboards/demo, and any run beyond the 3-seed validation scale.
- **Mid-phase amendment 2026-07-29** (human-approved, human-merged): the week-3 stack-validation
  spike is **closed** — both halves of ADR-0002 decision 5's kill criterion evaluated and did not
  fire (`docs/adr/0002-js-ml-stack.md` addendum; stand-ups 2026-07-22/23 plus the 2026-07-24
  correction, PRs #23–#25). Its priority slot is replaced by the Phase-2 quality pass below, and
  `reports/quality/` joins the allowed write paths.
- **Mid-phase amendment 2026-08-29** (human-approved, human-merged): an **explicit hold in a human
  reply overrides the priority-1 trigger** — see priority 1 below. Added after run 3 of 2026-08-27
  acted on PR #55's "do not act on this comment now ... pick it up on the next *scheduled* run" 40
  minutes after it was posted, and shipped PR #56 without acknowledging the override (PR #56/#57
  reviews). The rule is deliberately general: it binds on *any* clearly-worded hold, not on the
  particular wording #55 happened to use.
- **Mid-phase amendment 2026-09-23** (human-directed interactive session): the **stand-up report
  format is replaced** (v3, below). Stand-ups had grown to 775–1,336 words, and quiet days
  re-narrated the previous day's re-verification. **Learning journals are split**: root
  `LEARNING.md` is the human's; `loop/LEARNING.md` is yours. A **twice-monthly retro agenda**
  is added. `reports/retro/` joins the allowed write paths, and `Actor`/`Critic` join the reserved
  list above. Evidence: `reports/quality/2026-09-23-session-audit.md` P2–P7.

## Today's increment (Phase 2 / L2-P2-slice)

Pick **one** bounded increment, in priority order:

1. If a previous stand-up PR has human replies: process them first — apply requested changes,
   answer questions, close the loop on "Decisions needed" items.
   **Exception — an explicit hold outranks this trigger.** If the reply itself defers the work, that
   instruction wins over priority 1 and you do **not** process it in the current run. This binds on
   any clearly-worded hold, whatever its phrasing ("do not act on this now", "pick this up on the
   next scheduled run", "wait until X lands", "leave this for the human") — never read it narrowly
   against the exact words used. Carry the item to the run the hold names; if it names none, or its
   target is ambiguous, it defers to the **next scheduled run**, never to the current one. When you
   do act on a held item, say so in that run's stand-up "Done" — that it was held, when the hold was
   posted, and which run you are treating as the one it named. Overriding a hold, or acting on one
   silently, is a process failure to report under "Learned", not a judgement call to make.
2. **Phase-2 quality pass — bug hunt + plan-drift audit** (mid-phase amendment 2026-07-29;
   replaces the closed week-3 spike — see Current status). One run, one report:
   `reports/quality/YYYY-MM-DD-quality-pass.md`. **Bug hunt** across `src/`, `test/`,
   `experiments/`, six lenses in order: tensor/memory lifecycle (dispose `tf.variableGrads`
   `value` *and* `grads` — PR #25's leak class; `tf.tidy` never disposes `Variable`s);
   numerical correctness/stability (free-bits floor, `log`/`exp` guards, reduction semantics,
   NaN-halt actually fires); determinism/seeding (seeds via `src/env/rng.ts`, no hidden
   `Math.random`); shape/type safety — triage every `npm run typecheck` diagnostic (if that
   script exists; note `tsconfig.json` includes only `src` and `test`, so `experiments/` needs
   its own spot-check) into real bug / unproven-safe / strictness noise, never bulk-silenced
   with `!` or `as`; test-assertion strength (hunt tests that cannot fail); doc-vs-code
   mismatch. Seed findings to verify first (queued in PR #31's review): the `TS2339` at
   `test/agent/policy.test.ts:28`, and the three type errors in
   `experiments/2026-07-21-week3-stack-spike/benchmark.ts` outside `tsconfig.json`'s scope.
   **Drift audit**: compare `PLAN.html` Phase 2, proposal `0001`'s Arm-A status, this file, and
   the last five stand-ups' "Tomorrow" lines against the tree + `git log`; classify every
   mismatch done-but-unrecorded / recorded-but-undone / scope-creep / sequence-drift; propose
   reconciliation, never silently normalize either side. **Evidence rules**: a finding is
   reportable only with a repro command + output, a failing test, or a quoted trace; each
   carries `file:line`, `BLOCKING`/`NOTE`/`NIT` severity (the Skeptic rubric), and a confidence
   tag. The pass may label at most **three** findings `BLOCKING`; further would-be-BLOCKINGs
   land as `NOTE` with a promotion suggestion, for the human to confirm or promote in the pass
   PR's review — this caps how many runs the quality slot can pre-empt. Fix in-run only what is
   small, inside allowed paths, and `npm test`-verified; findings about `PLAN.html`, this file,
   ADRs, or the human's G2 modules are raised in the PR, never self-applied. **Self-limiting**:
   if any `reports/quality/*-quality-pass.md` exists — in the tree or added by an open PR — do
   not rerun the pass; while it carries unresolved `BLOCKING` findings, this item means fixing
   exactly one per run (smallest first, resolution appended — the report is append-only after
   merge); a finding leaves the set when fixed, human-waived in a PR comment, or escalated to
   "Decisions needed" because its fix lies outside your allowed paths. When none remain, move
   on to priority 3.
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
  (drafts), `docs/explainers/`, `reports/standup/`, `reports/quality/`, `reports/retro/`,
  `loop/` *except* `GOAL.md` (`loop/LEARNING.md` is append-only), and — new at this level —
  `src/`, `test/`, `experiments/`, within the vertical-slice scope above. Never write the
  human's journal (root `LEARNING.md`).
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

## Stand-up report (every run, no exceptions) — v3, mid-phase amendment 2026-09-23

Write `reports/standup/YYYY-MM-DD.md` (today's date) and include it in the PR. The human reads
it on a phone before deciding whether to open anything else, so **the top has to be useful on its
own**. Put what you need from them first and move the evidence to the bottom.

```markdown
# Stand-up — YYYY-MM-DD · run N · <L1 | L2-P2-slice | …> · priority <#>

> **TL;DR** <one sentence: what changed today>
> **Needs you:** <the single most useful human action, or "nothing today">

## Decisions needed
- [ ] <one-line question — options A / B; default if silent: A>
- Blocker: <only if something blocks the *next* run; omit the line otherwise>

## Done
- <≤3 bullets, each ending in → artifact path or PR link>

## Learned
- [<high|medium|low> · <self_checked|verified>] <finding> → <artifact path>

## Assumptions (reply to veto)
- <non-blocking default you acted on — or "none">

## Gate G<n> tracker
| criterion | status | evidence |
| --- | --- | --- |
| <one row per criterion of the next gate, from PLAN.html> | MET / PARTIAL / NOT MET | <path> |

## 📓 Learning nudge
- <exactly one bullet — see the rule below>

## Tomorrow
- <one line: the next increment>

## Manifest
- level: <…> | priority: <#> | files touched: <n> | runs: <none | manifest path(s)> | tests: <pass>/<total> (<todo> todo) | ratchet: <n>/<baseline>

<details><summary>Evidence</summary>

<verification commands and their output, re-verification detail, longer reasoning — anything a
reviewer might want to check but does not need to read to act>

</details>
```

**Rules:**

- **Budget**: at most **400 words above `<details>`** (the tracker table excluded). Anything longer goes
  inside `<details>` or into the artifact the bullet links to.
- **Quiet-day form**: when no priority item needed new work, the stand-up is only TL;DR,
  Decisions needed, Gate tracker, Learning nudge, and Manifest: **≤150 words** (the table
  excluded). Do not re-narrate the previous day's re-verification. Say "unchanged since
  <date>" and put what you checked in `<details>`.
- **Decisions needed comes first.** Blockers are folded into it as a `Blocker:` line.
  The heading text stays exactly "Decisions needed", so "Answering Decisions needed" and the
  safety valve below keep applying unchanged. Every question offers options and states the
  default you will take if there is no answer.
- **`run N`** = the number of `reports/standup/*.md` files on `main` when the run starts, plus 1.
  It is deterministic, so it never restarts or drifts again (it read "run 1" on 2026-09-22 and "run 32"
  on 2026-09-23).
- **Confidence tags** use one format everywhere: `[high|medium|low · self_checked|verified]`.
- **Gate tracker**: one row per criterion of the *next* phase gate as `PLAN.html` states it,
  status from evidence in the tree (not from a previous stand-up), evidence as a path.
- **📓 Learning nudge: exactly one bullet, addressed to the human**:
  - Name **one** explainer for them to summarize: the one this run added or changed, if any.
    Otherwise, the oldest unsummarized explainer relevant to the next human-owned task.
  - Give **one** concrete prompt question they could answer in three sentences.
  - Show the backlog as `N/M explainers summarized`. M = number of `docs/explainers/NNNN-*.md`
    files. N = number of distinct `NNNN` appearing in `### Summary — NNNN` headings in root
    `LEARNING.md`.
  - Encourage progress, not guilt: one entry this week beats a perfect backlog. If the human
    wrote an entry since the last stand-up, name it and thank them for it.
  - **Never write, edit, or pre-fill the human's entry.**

A run that produces no PR + report is a failed run. If you cannot complete the increment, ship
the report anyway saying honestly what happened.

## Learning journals and retro (mid-phase amendment 2026-09-23)

- **Root `LEARNING.md` is the human's journal.** Read it (to compute the nudge's N/M and to
  prepare the retro). Never write it.
- **`loop/LEARNING.md` is your journal, and it is append-only.** On the **first run of each ISO week**,
  append one entry, `## YYYY-MM-DD — week NN`, of at most ~150 words:
  - what went wrong in the loop's own process that week, and what a human review taught you;
  - an assumption that broke;
  - one thing you will do differently.

  Write it as raw material for the final write-up, not as ceremony. It is part of that run's
  reporting, not its increment.
- **Retro, twice a month.** On the first run on or after the **1st** and the **15th** of each
  month, also write `reports/retro/YYYY-MM-DD.md`: a **≤1-page agenda** that
  - pairs both journals since the previous retro (human entries by heading, your entries by
    date — quote a line, do not summarize the human's words for them);
  - lists the explainers summarized since then;
  - poses **2–3 discussion questions** where the two journals disagree or one is silent.

  This is reporting, not the run's increment. The discussion itself happens in an interactive
  session with the human. Link the agenda from that day's stand-up TL;DR.

## Answering "Decisions needed"

- The **human's comment/review is the answer** — a checked box is only bookkeeping and never carries a decision on its own. Judge whether an item is answered by the presence of a human comment/review, **never** by checkbox state, and never tick a "Decisions needed" box yourself (leave every one `[ ]` for the human).
- **An explicit hold *is* an answer** — it answers "when", not "what". Treat it as binding for the
  current run and carry the item forward per priority 1's exception; do not re-open it, re-litigate
  the timing, or treat a held item as unanswered for the safety valve's purposes.
- **Never self-resolve a "Decisions needed" item.** If it is genuinely the human's call, leave it and do not act. If you can safely proceed with a sensible default, it is not a decision — record it under "Assumptions made (proceeding unless told otherwise)" instead, so the human can veto rather than gate.

## Safety valve

If the **two most recent** stand-up PRs both have unanswered "Decisions needed" items, restrict today's run to processing/summarizing existing material (no new research directions) and say so in the report.
