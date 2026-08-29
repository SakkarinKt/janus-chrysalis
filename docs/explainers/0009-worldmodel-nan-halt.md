# Explainer: `WorldModel.step()`'s NaN → graceful halt invariant

`src/model/worldModel.ts`, `test/model/worldModel.test.ts` — `loop/GOAL.md` priority 5
("vertical-slice hardening... invariant tests: KL free-bits floor, WM/AC gradient separation,
NaN → graceful halt, continue/termination head"). Written explainer-before-implement per
`loop/GOAL.md`'s "Explainer-before-implement applies to each new core piece."

## Why this run picked this sub-item, not the one the human's PR #57 review named

PR #57's review (@SakkarinKt) said "Next: priority 3, RSSM completion — KL balancing with a
free-bits floor, observation reconstruction, and wiring RSSMCell into
`src/experiment/freeze.ts`'s rollout." Checked against `git log` before starting: all three of
those landed already, as three explicit sub-increments — PR #38 (KL-balanced loss, `docs/explainers/0004`),
PR #39 (rollout wiring, `docs/explainers/0005`), PR #40 (reconstruction loss, `docs/explainers/0006`).
Priority 4 (Arm-A metric plumbing + instrument validation) is also done (PR #42, PR #45). Redoing
either would mean touching already-reviewed, already-merged design decisions without a reason to
— out of scope for "one bounded increment," and risking a regression in code nobody flagged as
broken. This is raised as a "Decisions needed" item in this run's stand-up rather than silently
reinterpreted, per `loop/GOAL.md`'s "never self-resolve" rule for anything that's genuinely the
human's call — but proceeding on *some* increment today still needs a decision, and "which one"
here is a sensible default (continue down `loop/GOAL.md`'s own priority-ordered list to the first
item with real remaining work), not a call about a `loop/GOAL.md`-reserved concern, so it's
recorded under "Assumptions made," not left blocking.

Priority 5's four invariants, checked the same way before picking:

- **KL free-bits floor** — already tested (`test/model/losses.test.ts`'s
  `klBalancedLoss: free-bits floor clips...` test, landed with PR #38).
- **WM/AC gradient separation** — not yet applicable. This project has no actor-critic policy;
  `src/agent/policy.ts` has `RandomPolicy` and `QLearningPolicy` (tabular, no shared network with
  the world model), so there's no WM/AC gradient boundary yet to test. Nothing to do here until an
  AC-style policy exists.
- **NaN → graceful halt** — nothing in `src/`, `test/`, or `experiments/` referenced `isNaN`,
  `Number.isNaN`, or any halt mechanism before this run (`grep -rn -i "halt\|isNaN" src/ test/
  experiments/` returned nothing). This is the one this entry closes.
- **Continue/termination head** — `PLAN.html`'s Phase 2 line lists it ("Continue/termination head
  from day one"), and `docs/explainers/0006`'s "what's deliberately not here yet" section already
  names it and points here: "The continue/termination head is explicitly listed under `loop/GOAL.md`
  priority 5... not here." Still not built — it's a new core piece (a network head, a loss term, a
  design decision about what "termination" means in a fixed-`horizon` env that currently has no
  early-termination condition at all, per `src/env/types.ts`'s `GridWorldConfig.horizon` doc
  comment) needing its own explainer-before-implement pass, not something to fold into this
  smaller, narrowly-scoped entry. Proposed as this run's "Tomorrow" line.

## What it is

**`WorldModelNaNError`** (`src/model/worldModel.ts`), new: an `Error` subclass carrying `loss`,
`reconstructionLoss`, and `klLoss` (the same three numbers `WorldModelStepResult` already
surfaces on a normal return) as plain fields — set in the constructor body, not TS parameter-
property shorthand (`constructor(readonly x: number)` isn't valid stripped JavaScript under this
project's type-stripping-only Node execution; confirmed by hitting `ERR_UNSUPPORTED_TYPESCRIPT_SYNTAX`
when first written that way, same constraint `src/env/types.ts`'s `Action` const-object comment
documents for `enum`).

**`WorldModel.step()`**, changed: after `lossValue`/`reconstructionLossValue`/`klLossValue` are
read back (forward() completed normally — this is not the existing catch block's "threw before
returning" case), a new check —
`!Number.isFinite(lossValue) || !Number.isFinite(reconstructionLossValue) || !Number.isFinite(klLossValue)`
— guards state commitment. On a non-finite value: `observationTensor`, `nextDeterministic`, and
`nextStochastic` (already `tf.keep()`'d out of the inner `tf.tidy` by `forward()`) are disposed,
`this.state` is left exactly as it was (never reassigned — same contract the existing catch block
already gives a genuine throw from `forward()`), and a `WorldModelNaNError` is thrown.

**Scope: non-finite, not literally `NaN`-only.** The check is `!Number.isFinite`, which also
catches `±Infinity`. `loop/GOAL.md`'s invariant name is "NaN → graceful halt," but `Infinity` is
the same failure class — a numerical breakdown that would otherwise silently propagate into every
later step the same way a `NaN` would (`Infinity - Infinity` eventually produces `NaN` downstream,
but not necessarily before it's already corrupted several steps' worth of state) — so narrowing
the guard to exclude it would leave a known gap for no benefit. Recorded under this run's
"Assumptions made," not treated as a scope decision needing the human's sign-off.

## Why this can't undo a `train: true` step's optimizer update — and why the halt still matters

`WorldModel.step()`'s existing structure (`docs/explainers/0005`) calls
`tf.variableGrads(forward, this.trainableVars)` and `this.optimizer.applyGradients(grads)` *inside*
the same `tf.tidy` that computes `lossValue` — by the time this function can read `lossValue` back
as a plain number and check whether it's finite, `applyGradients` has already run, for
`train: true`. There is no checkpoint-and-roll-back mechanism in this codebase, so a `train: true`
step whose loss goes non-finite leaves the weights however that one (likely NaN or Infinite)
gradient update left them — confirmed directly in `test/model/worldModel.test.ts`'s new
"...documented limitation..." test, which asserts `cell.trainableWeights()` contains at least one
`NaN` immediately after the caught throw.

What the halt *does* still buy, and the reason it's not therefore pointless: without it,
`this.state` would be reassigned to `next{Deterministic,Stochastic}` — computed from that same
non-finite forward pass — and every subsequent `step()` call, for the rest of the episode (and,
via `WorldModel.reset()`, every later episode too, since only the recurrent state resets, not the
weights) would keep producing more non-finite losses. Those don't fail loudly on their own —
`WorldModelStepResult.loss` is just a `number`, and a `NaN` or `Infinity` flowing into, say, the
drift-attributable-error metric (`docs/explainers/0007`) could read as an extreme (if bogus)
finding rather than the training collapse it actually is. The halt turns a silent, delayed,
hard-to-attribute failure into a loud, immediate, correctly-attributed one, even though it can't
prevent the one step that caused it. `train: false` (the frozen-agent evaluation path proposal
`0001` depends on) has no such limitation — no `applyGradients` call exists on that path at all, so
a `train: false` NaN-halt throw leaves weights bit-identical, verified in the other new test.

## Why a thrown error, not a return-value flag

`WorldModelStepResult` already has three numeric fields a caller could check
(`Number.isFinite(result.loss)`) without a new error type. A thrown error was chosen instead
because every existing caller of `WorldModel.step()` — `src/experiment/freeze.ts`'s `runEpisode`,
and this project's `experiments/*/run.ts` scripts — currently treats a normal return as
unconditionally consumable (recorded straight into `EpisodeStepRecord.worldModelLoss`, no
finite-check anywhere today). Silently returning a `NaN`-tagged result would depend on every
present and future caller remembering to check a flag that's easy to forget precisely because it's
rare; an uncaught throw instead stops the run at the point of failure by default, which matches
"halt" more literally than "flag and hope something downstream notices." No caller in this
codebase currently catches `WorldModelNaNError` — that's deliberate: `runEpisode` isn't touched by
this entry (kept small per `loop/GOAL.md`'s "may not exceed one PR-worth" framing for a single
increment), so today an episode with a non-finite world-model loss stops the whole process, which
is still strictly better than the silent-corruption status quo this entry replaces. Whether
`runEpisode` (or a caller above it) should catch this and end the episode gracefully instead of
crashing the process is a reasonable next step, not required by "NaN → graceful halt" as stated —
"halt" doesn't specify "and keep the rest of the process running."

## Test coverage

`test/model/worldModel.test.ts` gains two tests, split by `train` because the weight-corruption
behavior above differs by branch (a single test can't assert both "weights unchanged" and "weights
became NaN" against the same run):

- `train=false`: a NaN-valued observation (`[NaN, ...OBSERVATION.slice(1)]`) throws
  `WorldModelNaNError` with `Number.isNaN(err.loss)`/`err.reconstructionLoss` true; recurrent state
  and every cell weight are bit-identical before/after; zero net tensor growth (same
  `tf.memory().numTensors` convention every other leak check in this file already uses); a
  subsequent ordinary step returns a finite loss (full recovery).
- `train=true`: same throw/state/leak assertions, plus the documented-limitation assertion above
  (at least one cell weight is `NaN` after the throw) — this test does *not* assert recovery
  afterward, since the model is left genuinely unusable until re-initialized or (once it exists) a
  checkpoint is restored.

## What's deliberately not here yet

- **Continue/termination head.** Still open — see the priority-list section above. This entry only
  closes the "NaN → graceful halt" line of priority 5's invariant-test list.
- **`runEpisode`/experiment-script-level recovery.** Discussed above — an uncaught
  `WorldModelNaNError` currently crashes the whole process rather than ending just the offending
  episode. Left as a follow-up, not required by this invariant as `loop/GOAL.md` states it.
- **A "was this instrument-validation run NaN-free" check surfaced anywhere in
  `experiments/*/run.ts` or a manifest field.** `PLAN.html`'s Gate G3 line ("each variant trains
  NaN-free for 3 seeds") is a gate criterion for later, not something this entry wires into any
  run's manifest — it only makes non-finite losses loud instead of silent; whether that gets
  recorded automatically per run is future work.
