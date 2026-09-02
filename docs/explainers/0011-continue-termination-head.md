# Explainer: continue/termination head

`src/model/continueHead.ts`, `src/model/losses.ts`, `src/model/worldModel.ts`,
`src/experiment/freeze.ts` — `loop/GOAL.md` priority 5's fourth invariant ("continue/termination
head"), reassigned to priority 3 by PR #62's review (@SakkarinKt, 2026-09-01): "Next: the continue
head per (A), as priority 3's first sub-piece." Written explainer-before-implement per
`loop/GOAL.md`'s "Explainer-before-implement applies to each new core piece."

## The decision this closes (PR #61/#62)

`reports/standup/2026-09-01.md` raised three options for `PLAN.html:110`'s "Continue/termination
head from day one," since Arm-A's gridworld (`src/env/gridworld.ts:57,65-66`) only ever sets
`done = currentStep >= horizon` — a deterministic function of the step counter, not a real
early-termination condition to predict. PR #62's review picked a fourth option: **(A) plus the
drift recorded** — "Implement the head against the horizon boundary, and write down that its
target is horizon-deterministic scaffolding — gradient plumbing at zero comparability cost, with
(B) [a real hazard/early-success condition] left open rather than foreclosed." The review's
reasoning for ruling out (B) as this run's shape: `experiments/2026-08-27-power-aware-radius-
analysis/analyze.ts:34` filters on `>=27/38`, and every per-seed statistic in that analysis is a
count out of 38 post-freeze steps — a denominator that only exists because the horizon is fixed.
Early termination wouldn't just break comparability with the existing results table (already true
of (B)); it would make the binomial test the whole Arm-A instrument depends on ill-defined.

**What this sub-increment is, honestly:** gradient/shape plumbing for a network head whose target
is exactly recoverable from information `WorldModel.step()`'s caller already has (the transition's
`done` flag) — not a genuine prediction task. `RSSMCell`'s recurrent state has no explicit step
counter as an input feature, so if the head's logit is to track `done` at all, the GRU state would
have to implicitly encode elapsed-step-count from the fixed action/observation sequence it has
already seen — possible in principle (GRUs can learn counters), not verified or claimed here. This
sub-increment does not attempt to verify that; it wires the head, the loss, and the rollout
plumbing, and documents the target's degenerate character so nobody mistakes low continue-loss
later for "the model learned something about termination" without checking what it actually
learned.

## What it is

**`ContinueHead`** (`src/model/continueHead.ts`), new: one `tf.layers.dense` layer, `units: 1`,
linear (no activation — the raw logit is what `continueLoss` below consumes, matching
`tf.losses.sigmoidCrossEntropy`'s expected input), taking `concat([deterministic, stochastic], 1)`
and returning a `[batch, 1]` logit. Same shape and minimalism as `ObservationDecoder`
(`docs/explainers/0006`) and `RSSMCell`'s `priorDense`/`posteriorDense` — a third head reading off
`(h_t, z_t)` per DreamerV3's architecture split (dynamics core vs. decoder/reward/continuation
heads), which `docs/explainers/0006`'s "what's deliberately not here yet" already named as the
gap this entry closes.

**`continueLoss(logit, target)`** (`src/model/losses.ts`), new: `tf.losses.sigmoidCrossEntropy(target,
logit)` (batch-mean reduction, tfjs's default) — binary cross-entropy from a raw logit against a
`{0, 1}` target. Unlike `categoricalKL`'s hand-rolled `log(probs + LOG_PROB_EPSILON)`, this uses
tfjs's own primitive directly rather than composing `sigmoid` + `log` by hand: `sigmoidCrossEntropy`
is implemented internally with the numerically-stable
`max(x, 0) - x*z + log(1 + exp(-|x|))` form (the standard "logits, not probabilities" formulation),
so it doesn't hit the `0 * log(0) = NaN` underflow class `LOG_PROB_EPSILON` exists to guard against
— confirmed present and differentiable on this project's pinned `@tensorflow/tfjs-node@4.22.0` by
calling it directly (`node --input-type=module` smoke check), not assumed from the tfjs-core docs
alone, given this codebase's existing burn history with APIs that look present but behave
differently under this pin (`tf.stopGradient`, `tf.multinomial`'s `seed` param — both documented in
`src/model/rssm.ts`).

**Target definition:** `done ? 0 : 1` — 1 means "the episode continues past this transition," 0
means "this was the last step," matching `StepResult.done`'s own polarity
(`src/env/types.ts`) rather than inverting it into a "termination" framing.

**`WorldModel.step()`** (`src/model/worldModel.ts`), changed: takes a new required `done: boolean`
parameter (fifth positional, after `train`) — the transition's outcome, supplied by the caller
exactly the way `observation` already is. `forward()` now also calls
`this.continueHead.predict(deterministic, nextStochastic)` on the **post-transition** state (the
same `(deterministic, nextStochastic)` pair `decoder.decode()` already reconstructs from, and what
gets persisted as the next `RSSMState`) and adds `continueLoss(logit, target)` to the scalar
`tf.variableGrads` differentiates. `total = reconstructionLoss + klBalancedLoss(...).total +
continueLoss(...)`, unweighted (coefficient 1), matching `docs/explainers/0006`'s precedent for
adding a new DreamerV3 loss term without inventing a new coefficient. `WorldModelStepResult` gains
a `continueLoss: number` field alongside the existing `loss`/`reconstructionLoss`/`klLoss`, same
"expose the breakdown so callers/tests don't have to recompute it" reasoning as `docs/explainers/0006`.

**`WorldModelNaNError`** (`src/model/worldModel.ts`), changed: gains a `continueLoss: number` field,
and the non-finite check that throws it now also covers `continueLossValue` — same "any of the
loss's components going non-finite compounds into every later step" reasoning
`docs/explainers/0009` gives for the original three-way check, extended to the fourth term rather
than left as a gap this entry would otherwise silently introduce.

**Constructor wiring:** a third `deriveSeed(config.seed, 2)` salt (cell=0, decoder=1 already
existed) seeds `ContinueHead`'s kernel initializer when `config.seed` is given. The constructor's
existing throwaway warmup forward pass now also calls `continueHead.predict(...)` once, so its
layer is built before `trainableVars` (which now includes `continueHead.trainableWeights()`) is
captured — same reason the decoder was warmed up there (`docs/explainers/0006`).

**`src/experiment/freeze.ts`**, changed: `runEpisode`'s one production call site
(`worldModels?.[i]?.step(...)`) now passes `result.done` as the fifth argument. No other change —
`result.done` was already computed and already flowed into `EpisodeStepRecord.done`, just not into
the world model itself before this entry.

## Why the post-transition state, not the pre-transition state

Same reasoning `docs/explainers/0006` gives for reconstructing from the posterior's sample rather
than the prior's: `done` is a property of *this* transition (`currentStep >= horizon` *after*
`currentStep` was incremented, `src/env/gridworld.ts:62,65`), so the state that has "seen" this
transition — `nextDeterministic`/`nextStochastic`, computed inside the same `forward()` call — is
the state whose logit should be asked to predict it. Predicting from `prevState` instead would ask
the head to predict this step's outcome before the step's own action/observation had been folded
in, which isn't what `done` is a function of.

## Why this doesn't move at increment priority (still "priority 3, sub-piece 1")

Priority 3 in `loop/GOAL.md` reads "RSSM completion: the world-model losses (KL balancing... obser-
vation reconstruction) and wiring `RSSMCell` into `src/experiment/freeze.ts`'s rollout" — both
already done (PR #38–#40, per `docs/explainers/0009`'s priority-check). PR #62's review explicitly
renamed this item "priority 3's first sub-piece" rather than treating it as already-closed-out-of-
order under priority 5; this entry follows that review's placement rather than re-deriving its own
read of `loop/GOAL.md`'s ordering, since the review is the human decision this run is processing
under priority 1.

## What's deliberately not here yet

- **A real (non-degenerate) termination signal.** Option (B) from PR #61's "Decisions needed" —
  explicitly left open by PR #62's review, not foreclosed. Would need its own scoping (env change,
  baseline re-run plan or explicit arm fork) before any code lands, per the review's own framing.
- **`PLAN.html:110` wording.** The review asked for the wording change to be raised in this PR for
  the human to apply (`loop/GOAL.md`'s boundaries: `PLAN.html` needs approval, never a self-edit).
  Proposed text, for the human to apply or amend: *"Continue/termination head from day one
  (implemented against the fixed-horizon boundary in the absence of a real early-termination
  condition — gradient/shape scaffolding, not a genuine prediction task; see
  docs/explainers/0011)."*
- **Reward head.** DreamerV3's full `L_pred` also includes a reward-prediction term
  (`docs/explainers/0006`'s "what's deliberately not here yet" already named this alongside the
  continuation head). Still out of scope — this entry closes only the continuation half.
- **Using the continue head's prediction anywhere** (e.g. gating imagination rollouts, or feeding
  `drift-attributable-error`, `docs/explainers/0007`). Not part of this vertical slice; the head
  exists and trains, nothing downstream reads its output yet.

## Test coverage

`test/model/continueHead.test.ts`, new: `predict()` returns `[batch, 1]`; `trainableWeights()` is
non-empty after one `predict()` call; same seed reproduces identical initial weights, different
seeds diverge (mirrors `test/model/decoder.test.ts`'s three tests exactly).

`test/model/losses.test.ts` gains: `continueLoss` is (near-)zero for a confidently-correct logit
against its matching target and large for a confidently-wrong one (sanity-checks the label/logit
polarity actually matches `sigmoidCrossEntropy`'s expected argument order, not just that the
function runs); gradient flows into the logit (`tf.variableGrads` against a `tf.Variable` logit is
nonzero).

`test/model/worldModel.test.ts`: every existing `wm.step(...)` call site gains a `done` argument
(most pass `false` — mid-episode transitions — since none of the existing tests were exercising an
episode boundary); the "reconstructionLoss + klLoss equals loss" test is renamed and extended to
"reconstructionLoss + klLoss + continueLoss equals loss"; new tests: `train=true` moves at least
one `ContinueHead` weight, `train=false` moves none (mirrors the existing decoder-weight test); a
`done: true` step's continuation target is `0` and a `done: false` step's is `1`, checked
indirectly by driving `continueLoss` down over repeated identical-`done` training steps in each
direction (same "coarse does-it-learn check" convention as the existing reconstruction/KL-loss
descent tests) rather than reading the target tensor directly (private to `forward()`, not
exposed).
