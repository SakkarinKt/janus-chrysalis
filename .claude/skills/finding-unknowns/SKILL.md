---
name: finding-unknowns
description: >
  Phase-quality pass for janus-chrysalis — hunt for verified bugs in the TS
  world-model stack (tensor lifecycle, numerics, determinism, shape/typecheck
  backlog, weak tests, doc-vs-code) and audit drift between the repo and
  PLAN.html / proposal 0001 / loop/GOAL.md. Use when asked for a quality pass,
  bug hunt, or plan-drift audit, or when GOAL.md's quality-pass increment is
  next.
---

# Finding unknowns — the Phase-2 quality pass

Turn unknown unknowns into severity-tagged knowns. Two workstreams, one run, one report:
**Part A** hunts for real defects; **Part B** audits drift between what the plan says and what the
tree contains. This file is a plain checklist — it assumes nothing beyond reading files and running
the repo's own npm scripts, so the daily loop can follow it without a Skill tool. **Scheduling is
not defined here**: when (and whether) to run the pass, and what "self-limiting" means, lives in
`loop/GOAL.md`, Today's-increment priority 2. This file defines only how.

## Ground rules

- A finding is reportable only if **verified**: a reproduction command with its output, a failing
  test, or a quoted code trace. Anything less must be phrased as a suspicion and tagged
  `[low, self_checked]` — suspicions may not be `BLOCKING`.
- Severity uses `CONTRIBUTING.md`'s Skeptic rubric: **BLOCKING** (wrong results, crash, or an
  existing claim invalidated) / **NOTE** (should fix; conclusions survive) / **NIT** (style).
- Every finding cites `file:line` (and an artifact path if it came from a run), and carries a
  confidence tag `[high|medium|low, self_checked|verified]`.
- Fix in-run only what is small, inside the loop's allowed write paths, and verifiable by
  `npm test` in the same run. Never widen a fix into a refactor. Everything else is reported.
- Findings about `PLAN.html`, `loop/GOAL.md`, ADRs, or the human-reserved G2 modules are
  **raised, never self-applied**: propose exact wording in the PR, or file a "Decisions needed"
  item.
- This pass changes no boundaries: no new dependencies, no training runs beyond the scope
  `loop/GOAL.md` already grants.
- One run only. If time runs out, ship the report with Coverage marking unexamined lenses
  honestly; do not schedule a continuation yourself — the human can request one in the PR.

## Part A — bug-hunt lenses (in this order)

1. **Tensor and memory lifecycle.** Every `tf.*` op result is returned, disposed, or inside
   `tf.tidy`. `tf.variableGrads`: dispose `value` **and** every tensor in `grads` (PR #25's leak
   class). `tf.tidy` never disposes `tf.Variable`s, even ones created inside it (documented in
   `test/model/rssm.test.ts`) — check nothing relies on the opposite. Verify suspicions with a
   `tf.memory().numTensors` before/after check in a throwaway script; do not commit the script.
2. **Numerical correctness and stability.** KL terms floored (free bits); `log`/`exp`/softmax
   guarded against 0/NaN/Inf; loss reductions match their stated semantics (mean vs sum over
   batch/time); the NaN-graceful-halt path actually triggers when fed a NaN.
3. **Determinism and seeding.** Every stochastic op takes an explicit seed from `src/env/rng.ts`
   or an RSSM seed parameter; no hidden `Math.random`; the same seed twice gives identical output.
4. **Shape, index, and type safety.** If `package.json` has a `typecheck` script, run
   `npm run typecheck` and triage every in-repo diagnostic (43 as of PR #30 — a snapshot, not a
   quota) into: (a) real latent bug — fix or raise as a finding; (b) unproven-safe — propose a
   guard; (c) strictness noise from `noUncheckedIndexedAccess` — record proposed handling. Never
   bulk-silence with `!` non-null assertions or `as` casts.
5. **Test-assertion strength.** Hunt tests that cannot fail: tolerances wide enough to pass a
   broken implementation, shape-only assertions where values matter, loops that can run zero
   iterations, unawaited async assertions. Spot-check by mutation: flip one sign or constant in
   the code under test (scratch only, revert immediately) and confirm the suite goes red — if it
   stays green, that is a finding about the test.
6. **Doc-vs-code mismatch.** Statements in `README.md`, `docs/explainers/`, and `notes/`
   contradicted by the code as it now exists.

## Part B — plan-drift audit

Compare, line by line, against the actual tree and `git log`: `PLAN.html`'s Phase-2 bullets and
G2 criteria; proposal `0001`'s Arm-A milestone and status log; `loop/GOAL.md`'s Current status
and priority list; the last five stand-ups' "Tomorrow" promises. Classify every mismatch:

- **done-but-unrecorded** — work landed; plan or status still says pending
- **recorded-but-undone** — plan or status claims it; the tree lacks it
- **scope-creep** — work landed that no plan line asks for
- **sequence-drift** — order of work diverges from the plan's stated order

Drift findings get the same severity and confidence tags. Drift is not automatically bad: the
output is a proposed reconciliation (change the repo, or propose changing the plan) — never a
silent normalization of either side.

## Output contract

Write `reports/quality/YYYY-MM-DD-finding-unknowns.md`:

```markdown
# Finding-unknowns report — YYYY-MM-DD (Phase 2)

## Scope and method

<commit audited, lenses run, commands run with versions>

## Findings

### Q-001 — <title> [BLOCKING] [high, self_checked] — src/model/rssm.ts:123

Evidence: <repro command + output, or quoted trace>
Proposed action: <fix now | fix in follow-up run | raise>
Fixed in this run: <yes (commit) | no>

## Drift audit

<one entry per mismatch, classified with the four labels above>

## Clean lenses

<lenses that found nothing, each with a confidence tag — absence of findings is a claim too>

## Coverage

<what was not examined, and why>
```

- Findings are numbered `Q-NNN` and ordered BLOCKING, NOTE, NIT.
- Every BLOCKING finding also appears in the day's stand-up: fixed ones under "Done", unfixed
  ones under "Decisions needed" as a one-line question.
- A BLOCKING finding whose fix lies outside the loop's allowed paths (or inside a human-reserved
  module) is raised as a "Decisions needed" item and marked `Resolved: escalated YYYY-MM-DD` —
  it then stops occupying the priority queue.
- After the report's PR merges, the report is append-only: record a resolution by appending a
  dated `Resolved:` line to the finding's entry (citing the fixing commit/PR, the human's waiver
  comment, or the escalation), never by rewriting the finding.
