# Quality pass — 2026-08-03

Phase-2 quality pass (mid-phase amendment 2026-07-29, `loop/GOAL.md` priority 2). First landing of
this report — `reports/quality/` previously held only `.gitkeep` (PR #31). Six-lens bug hunt across
`src/`, `test/`, `experiments/`, plus a plan-drift audit. Baseline before this run: `npm test`
47/47, `npm run typecheck:ratchet` 43/43 (exit 0).

## Bug hunt

### Lens 1 — tensor/memory lifecycle

**[NOTE, self_checked, high confidence] `test/model/rssm.test.ts`: `tf.variableGrads`'s `value`
return dropped undisposed — PR #25's exact leak class.** Both gradient-check tests that call
`tf.variableGrads` directly (not through the `checkFiniteDifference` helper, which already tidies
correctly) destructured only `{ grads }`, silently leaking the forward-pass loss scalar (`value`)
once per test run.

- Repro: instrumented a standalone copy of the single-step gradient test's `forward()`/`variableGrads`
  call with `tf.memory().numTensors` before/after. Before the fix: 1 net undisposed tensor per call
  (excluding `grads`, which are consumed). After the fix: 0.
- Fixed in-run: `test/model/rssm.test.ts` — destructure `{ value, grads }` and call `value.dispose()`
  before using `grads`, at both call sites (former lines 337 and 405; the single-step and
  chained-across-timesteps tests). Verified: `npm test` 47/47 still passes; leak-check script above
  confirms 0 net tensors after the fix.
- Not BLOCKING: confined to the test process, which exits after the suite — no correctness impact,
  no CI failure. Flagged because it's precisely the leak class the last four PRs (#25, #27, #28, #29)
  were about, and it had escaped notice in the file's two main (non-helper) `variableGrads` calls.

**[NOTE, self_checked, medium confidence] `RSSMCell.step()`/`.prior()`/`.posterior()` carry no
documented `tf.tidy` contract.** `step()` builds a `tf.concat` recurrent-input tensor and
`oneHotActions()` builds two more (`tf.oneHot` then `.toFloat()`), none of which the method disposes
itself — callers are implicitly expected to wrap calls in `tf.tidy`. Every current call site does
this correctly (`benchmark.ts`'s three benchmarks, `repro.ts`, and the test helpers), so nothing is
leaking today, but the class's own doc comment (`src/model/rssm.ts:58-57`) never states this
contract. Priority 3 ("RSSM completion: ... wiring `RSSMCell` into `src/experiment/freeze.ts`'s
rollout") will be the first *production* (non-benchmark, non-test) call site — worth a one-line
doc note (and a deliberate `tf.tidy` wrap) when that lands, so the contract isn't rediscovered by
a leak. Not fixed here: touching `RSSMCell`'s doc comment right before priority 3 rewrites it seems
more likely to create merge friction than help; flagging for that increment's own
explainer-before-implement pass instead.

### Lens 2 — numerical correctness/stability

No findings. The free-bits floor, KL balancing, reduction-semantics, and NaN-halt code this lens
asks about don't exist yet — they're `loop/GOAL.md` priority 3/5 (RSSM losses, invariant tests),
still unimplemented. Nothing to audit prematurely.

### Lens 3 — determinism/seeding

**[BLOCKING, self_checked, high confidence] `sampleHard()` (`src/model/rssm.ts:196-201`) draws
from `tf.multinomial` without a seed — stochastic-latent sampling is not reproducible per the
project's `Rng`, and never routes through it.** `tf.multinomial(logits, numSamples, seed?,
normalized?)` accepts an optional numeric seed
(`node_modules/@tensorflow/tfjs-core/dist/ops/multinomial.d.ts:40`); `sampleHard` calls it with none,
so every draw uses tfjs's own internal (non-project-seeded) randomness source, not
`src/env/rng.ts`'s `Rng`. Every other source of randomness in the repo — environment spawn
positions, `RandomPolicy.act` — is threaded through an explicit `Rng` instance and is reproducible
per-seed (confirmed by existing tests: `rng.test.ts`, `gridworld.test.ts`'s "same seed + same
actions produce identical trajectories", `policy.test.ts`'s "driven by the passed-in rng, not
internal state"). The stochastic latent is the one place that contract is silently broken.

- Repro (executed directly, not from a test file):
  ```js
  function run() {
    const rssm = new RSSMCell({ deterministicSize: 4, latentCategoricals: 3, latentClasses: 5 });
    const det = rssm.step(rssm.initialState(1), [1]);
    return rssm.prior(det).sample.arraySync();
  }
  console.log(run(), run());
  ```
  Output: `[[1,0,0,0,0,0,1,0,0,0,0,0,0,1,0]]` vs `[[0,0,0,1,0,1,0,0,0,0,0,0,1,0,0]]` — two fresh
  `RSSMCell`s, identical config/state/action, different samples. (Every existing test either fixes
  `hard` explicitly via `fixedHard`/`priorHard`, or only checks shape/marginal properties that don't
  depend on which sample was drawn — so this doesn't show up as a failing test today.)
- Why BLOCKING, not NOTE: this doesn't block anything currently shipped (no real, non-`fixedHard`
  sampling path is exercised outside benchmarking), but proposal `0001`'s whole instrument-validation
  methodology is a **3-seed** comparison — "same seed reproduces the same run" is exactly the
  property this breaks, and priority 3 (RSSM completion) is the very next increment that will start
  calling `prior()`/`posterior()` for real, non-fixed-hard sampling. Left unseeded, every future
  training/rollout run becomes silently non-reproducible the moment it samples the stochastic latent,
  which is difficult to catch after the fact (nothing crashes; numbers just stop replicating).
- Not fixed in-run: the fix isn't small — it means threading a seed (derived from the project's
  `Rng`, e.g. `rng.nextInt(2**31)`) through `sampleHard` → `sampleStraightThrough` → `prior()` and
  `posterior()`, which changes `RSSMCell`'s public API and touches every non-`fixedHard` call site
  (several in `test/model/rssm.test.ts`). That's `RSSMCell` API surface, which priority 3 owns and
  is about to touch anyway — proposing here rather than pre-empting that increment's own design
  call on exactly how the seed threads through.

No hidden `Math.random` found anywhere in `src/`, `test/`, `experiments/`, or `scripts/` (checked
directly: `grep -rn "Math.random" src/ test/ experiments/ scripts/` — no matches). The two
`new Date()` calls (`experiments/*/benchmark.ts:260`, `experiments/*/repro.ts:244`) are both
run-id/date-stamping for `summary.json`, not simulation randomness — out of scope for this lens.

### Lens 4 — shape/type safety

Triaged all 43 baseline `npm run typecheck:ratchet` diagnostics, plus a spot-check of both
`experiments/` files (outside `tsconfig.json`'s `include`, per the ratchet script's own scope note).
Method: re-ran `tsc` with the same flags but `noUncheckedIndexedAccess: false` to isolate which
diagnostics come from that one strictness flag vs. everything else.

| Bucket | Count | Verdict |
| --- | --- | --- |
| `noUncheckedIndexedAccess` strictness noise (array/index-signature access TS believes could be `undefined`) | 40 of 43 baseline, + 2 of 3 in `benchmark.ts`, + all 7 in `repro.ts` | **unproven-safe, not a bug** — every instance is guarded by an established invariant TS can't see: `NUM_AGENTS`-fixed-length arrays (`gridworld.ts`, `freeze.ts`, their tests), config-bounded loop indices, or lookups into a `NamedTensorMap`/`registeredVariables` keyed by names the same function just derived from that object's own keys. Confirmed by isolation test above: with the flag off, all but 2 baseline errors and all `repro.ts` errors vanish. |
| `test/agent/policy.test.ts:28` (`TS2339`) | 1 | **real bug** — fixed in-run (see Lens 5). |
| `src/model/rssm.ts:174` (+ cascading 181/182 under full strictness) — `tf.customGrad`'s declared `CustomGradientFunc` type doesn't correctly model a 2-input custom-grad callback | 3 of 43 baseline | **library typing limitation**, not a repo bug — confirmed present even with `noUncheckedIndexedAccess` off. Already worked around by the single, narrow `as (logits: tf.Tensor, hard: tf.Tensor) => tf.Tensor` cast at `rssm.ts:187` (not a bulk `as any` suppression), and the actual runtime gradient correctness is independently verified by the finite-difference tests in `rssm.test.ts`. No action needed. |
| `experiments/2026-07-21-week3-stack-spike/benchmark.ts:261` (`TS7006`, implicit `any`) | 1 of 3 in that file | **[NOTE, self_checked, medium confidence]** `existingRuns.filter((r) => r.date === date)` — `existing.runs` comes from `JSON.parse(readFileSync(...))`, typed `any`; the `Array.isArray(existing.runs) ? existing.runs : []` ternary doesn't narrow it, so `existingRuns` (and `r`) stay implicitly `any` — a real type-safety gap in run-history parsing (nothing validates `summary.json`'s shape). Not fixed in-run: this file's `summary.json` is append-only by explicit prior decision (PR #24 review, "overwriting it twice... destroyed the evidence behind numbers already cited") and the file sits outside `tsconfig.json`'s scope — a type-safety fix here is speculative hardening, not a confirmed-reachable bug, and better bundled with whatever next touches this experiment than done standalone. |

**New scope note**: `experiments/2026-07-25-ste-chained-finite-difference-trap/repro.ts` (added
after PR #30 wrote the ratchet script's scope note) has 7 of its own `noUncheckedIndexedAccess`-class
diagnostics, previously unaudited — `loop/GOAL.md`'s seed findings only named `benchmark.ts`'s
three. All 7 fall in the same "unproven-safe" bucket confirmed above (mostly `grads[variable.name]`
/ fixed-length array indexing). No fix needed, but recommend the "spot-check `experiments/`" habit
mentioned in the ratchet script's comment use `npx tsc --noEmit experiments/**/*.ts` (both files)
rather than continuing to special-case `benchmark.ts` by name.

### Lens 5 — test-assertion strength

One real fix, one near-tautological assertion left as a NIT:

- **[fixed in-run] `test/agent/policy.test.ts:28`** — `assert.equal(policy.update, undefined)` on a
  `policy: RandomPolicy` fails to type-check (`TS2339`, the seed finding named in `loop/GOAL.md`):
  `RandomPolicy` itself declares no `update` property — only the `Policy` interface's *optional*
  signature does — so accessing `.update` on the concrete class type doesn't type-check even though
  it's semantically exactly what the test means to assert (per `src/agent/policy.ts:23`'s own doc
  comment: "Absent for policies with nothing to learn"). Fixed by typing `policy` as `Policy` at the
  declaration (`const policy: Policy = new RandomPolicy();`) rather than casting or widening the
  class itself — the assertion's meaning doesn't change, only its declared type. Verified:
  `npm test` 47/47; `npm run typecheck:ratchet` in-repo count 43 → 42.
- **[NIT, self_checked, low confidence] `test/env/gridworld.test.ts:109`** —
  `assert.ok(last !== undefined)` in "collision penalty applies when agents share a cell". `last` is
  assigned unconditionally on every one of the 20 preceding loop iterations, so this assertion is
  true regardless of whether collision-penalty logic is correct — it only checks that the loop ran.
  The test's real check (`assert.deepEqual(a0, a1)`, confirming the agents actually collided) is
  unaffected and does the load-bearing work. Not fixed in-run: removing a passing assertion or
  replacing it with something more meaningful (e.g. asserting on `last.reward`) is a judgment call
  about the test author's intent that a bug-hunt pass shouldn't make unilaterally.

No other tests in `rng.test.ts`, `freeze.test.ts`, `rssm.test.ts`, or the rest of `gridworld.test.ts`
read as unable to fail — spot-checked every `assert.ok(...)` call site in the suite (11 total) for
this report; the other 10 all gate on a condition that a real regression would flip.

### Lens 6 — doc-vs-code mismatch

**[NOTE, self_checked, high confidence, fixed in-run] `src/env/types.ts:23` called `viewRadius` a
"Chebyshev-distance radius"; the actual masking logic is Manhattan distance.**
`CooperativeGridWorld.relativeEntry()` (`src/env/gridworld.ts:125`) gates visibility on
`this.distance(self, other) > this.config.viewRadius`, and `distance()` (`gridworld.ts:94-96`) is
`Math.abs(dx) + Math.abs(dy)` — Manhattan (L1), not Chebyshev (L∞ = `max(|dx|, |dy|)`). This isn't
an ambiguous call: `docs/explainers/0001-cooperative-gridworld-env.md:17` explicitly documents
"**Manhattan distance, not Euclidean**, for both reward and visibility" as a deliberate design
choice, and `test/env/gridworld.test.ts`'s own visibility test builds an explicit `manhattan()`
helper to check against — both the design doc and the tests agree with the code; only the
`GridWorldConfig` field comment used the wrong word. Fixed in-run: corrected the comment to
"Manhattan-distance radius" (pure doc fix, no behavior change). Verified: `npm test` 47/47
unaffected (doc-only change).

No other doc-vs-code mismatches found: `docs/explainers/0002-freeze-mechanism.md` and
`docs/explainers/0003-rssm-world-model-cell.md` were checked against their corresponding
`src/experiment/freeze.ts` and `src/model/rssm.ts` implementations — both explainers are current
(0003 was last updated 2026-07-20 for the stochastic-latent landing and its description of
`step()`/`prior()`/`posterior()`/`straightThroughEstimator()` matches the code). The
`tf.stopGradient` → `tf.customGrad` correction flagged inline in `rssm.ts:154-161`'s own doc comment
("worth a correction pass on that note") was checked against `notes/adr-0002-js-ml-stack.md` — the
correction already landed there (§7, 2026-07-20 entry, "supersedes... not edited in place"); the
`rssm.ts` comment is itself just slightly stale in *pointing at* an already-fixed note, not
describing a live inconsistency. Not worth a fix.

## Drift audit

Compared `PLAN.html` Phase 2, proposal `0001`'s Arm-A status log, `loop/GOAL.md`'s "Current status",
and the last five stand-ups' "Tomorrow" lines (2026-07-29 through 2026-08-02) against the tree and
`git log --oneline` (37 commits total, `bootstrap` through PR #35).

**No mismatches found.** Specifically checked:

- **PLAN.html vs. tree**: Phase 2's bullet list (RSSM cell, JSONL/manifest, quality pass) matches
  what exists (`src/model/rssm.ts` has the cell struct/forward-pass/stochastic-latent per PR
  #17/#20; no JSONL/manifest code yet, consistent with priority 5 being unstarted; the quality-pass
  bullet was added by PR #31 alongside this run's actual pass). The week-3 spike line is correctly
  marked "✓ closed 2026-07-23."
- **Proposal `0001`'s Arm-A status log vs. `git log`**: dated entries run 2026-07-14 through
  2026-07-25 and match the commit history for that span (env PR #10, freeze PR #12/#13, cell PR #17,
  stochastic latent PR #19/#20, week-3 spike PRs #22-#24, STE chaining root-cause PR #25/#26). No
  entry exists for 2026-07-29 through 2026-08-02 (the GOAL-amendment PR #31 and the four
  typecheck-ratchet-hardening PRs #32-#35) — checked whether this is a gap: it isn't. Those five PRs
  are CI-tooling and process-contract changes, not Arm-A-milestone progress; the proposal's update
  log is specifically an Arm-A status log (its own framing: "environment... landed", "cell's
  struct/forward-pass sub-increment landed", etc.), so tooling work between milestone updates is
  correctly out of its scope, not a silently-dropped entry.
- **`loop/GOAL.md`'s "Current status" vs. reserved-for-human scope**: no `src/`/`test/` file
  touches replay-buffer or λ-returns concepts (`grep -rln -i "replay\|lambda.return" src/ test/`
  returns only `gridworld.ts`'s one incidental comment about *replay ratio*, an unrelated
  training-throughput concept from the benchmark doc comments — not the reserved replay buffer
  module). The reservation is intact.
- **Last five stand-ups' "Tomorrow" lines**: 2026-07-29 through 2026-08-01 each said "priority 2
  (quality pass) next" but got redirected to priority-1 review-reply processing on PR #33/#34/#35's
  reviews each following day — a live, visible priority-1-preempts-priority-2 pattern, not drift
  (priority 1 explicitly outranks priority 2 in `loop/GOAL.md`'s own ordering, and each redirection
  was to a human review reply, exactly what priority 1 exists for). 2026-08-02's stand-up (PR #35)
  was the first with no further review redirection, and this run is that promised quality pass —
  the chain resolves cleanly, no open loose end.

## Findings summary

| # | Severity | File:line | Status |
| --- | --- | --- | --- |
| 1 | BLOCKING | `src/model/rssm.ts:196-201` (`sampleHard`) | Open — unseeded `tf.multinomial`, proposed fix above, not applied (API-surface change, belongs to priority 3) |
| 2 | NOTE | `test/model/rssm.test.ts` (two `variableGrads` call sites) | Fixed in-run |
| 3 | NOTE | `src/model/rssm.ts` class doc (no `tf.tidy` contract stated) | Open — recommend priority 3 documents it when wiring `freeze.ts` |
| 4 | NOTE | `experiments/2026-07-21-week3-stack-spike/benchmark.ts:261` | Open — low-stakes, deferred to next touch of that file |
| 5 | NOTE | `test/agent/policy.test.ts:28` | Fixed in-run |
| 6 | NOTE | `src/env/types.ts:23` | Fixed in-run |
| 7 | NIT | `test/env/gridworld.test.ts:109` | Open — left for human judgment call |

1 of 3 allowed BLOCKING slots used. Nothing escalated to "Decisions needed" — finding #1 is raised
in this report per `loop/GOAL.md`'s instruction ("findings about... are raised in the PR, never
self-applied") but doesn't need a same-day yes/no from the human; it's a proposed next-increment
input for whoever picks up priority 3 or the next quality-pass run's "fix exactly one" cycle.

## Files touched this run

- `test/agent/policy.test.ts` (finding #5 fix)
- `test/model/rssm.test.ts` (finding #2 fix)
- `src/env/types.ts` (finding #6 fix)
- `reports/quality/2026-08-03-quality-pass.md` (this report)
- `reports/standup/2026-08-03.md`

Verified after all three fixes together: `npm test` → 47/47 pass; `npm run typecheck:ratchet` →
in-repo count 43 → 42, baseline (43) unchanged, exit 0 (script itself notes "Baseline could be
lowered to 42" — not acted on here; `scripts/` isn't in this increment's allowed-paths list, and
the ratchet gate is unaffected either way at the current count).
