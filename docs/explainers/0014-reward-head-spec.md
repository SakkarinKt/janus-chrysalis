# Explainer/spec: reward head (Dreamer-style reward prediction, docs-only)

`src/model/rewardHead.ts`, `src/model/losses.ts`, `src/model/worldModel.ts` — proposed only, no
`src/` changes in this increment. Closes the blocking-dependency gap
`docs/explainers/0013-actor-critic-design-spec.md` flagged twice (its "Blocking dependency this
spec does not resolve" section and open question 3): `computeLambdaReturns`'s `rewards` input
(`src/agent/lambdaReturns.ts`, `docs/explainers/0012`) has no source once an imagined rollout
leaves real environment steps, since `env.step()` is never called inside `imaginationTrainStep`.
Picked as this run's increment per the 2026-09-14 stand-up's "Tomorrow" line, itself chosen from
two options offered by the human's PR #74 review (2026-09-13, `@SakkarinKt`): the reward-head
spec "same treatment as `0012`/`0013` — interfaces, invariants, open questions, no `src/`
changes," over priority 5's remaining invariant tests (confirmed by that run's own grep that every
named invariant except WM/AC gradient separation already has a dedicated test file, and that one
is itself blocked on Gate G2 plus this reward head).

## Why this is a world-model head, not an agent module

DreamerV3 (arXiv:2301.04104) trains one reward-prediction head per world model, alongside the
decoder and continuation head, all three reading off the shared recurrent state `(h_t, z_t)` —
this project already has two of the three: `ObservationDecoder` (`docs/explainers/0006`) and
`ContinueHead` (`docs/explainers/0011`). `decoder.ts:20`'s doc comment has named this gap since
before `ContinueHead` was built: *"the decoder (and reward/continuation heads, not built yet)"* —
this entry closes the remaining half.

## Architecture — `RewardHead` (`src/model/rewardHead.ts`, proposed)

One `tf.layers.dense` layer, `units: 1`, linear (no activation) — identical shape and minimalism
convention to `ContinueHead`, reading `concat([deterministic, stochastic], 1)` and returning a
`[batch, 1]` scalar. Unlike `ContinueHead`'s logit (squashed downstream by `sigmoid`), this output
is the reward prediction itself, with no further transform — see "reward magnitude" below for why
no `symlog` transform is proposed despite `0013`'s parenthetical suggesting one.

```ts
export interface RewardHeadConfig {
  /** Same deriveSeed-per-component convention as WorldModelConfig — see below. */
  seed?: number;
}

export class RewardHead {
  constructor(config?: RewardHeadConfig);
  /** Scalar reward prediction, [batch, 1] — differentiable, no activation. */
  predict(deterministic: tf.Tensor2D, stochastic: tf.Tensor2D): tf.Tensor2D;
  trainableWeights(): tf.Variable[];
}
```

## Reward magnitude: correcting `0013`'s `symlog` note

`docs/explainers/0013`'s blocking-dependency section proposed the future reward head be "symlog-
transformed per DreamerV3, since the gridworld's reward already includes a `collisionPenalty` term
that isn't bounded to a friendly range for a bare linear head." Re-derived from `computeReward()`
(`src/env/gridworld.ts:145-157`) rather than carried forward as written, `self_checked, high
confidence`:

```
reward = -coverage / gridSize - (collision ? collisionPenalty : 0)
coverage = sum over numLandmarks landmarks of min(distance(agent, landmark) for agent in agents)
```

Every configuration this project currently runs uses `DEFAULT_CONFIG`'s `gridSize: 8`,
`numLandmarks: 2` — verified this run: `grep -rn "gridSize\|numLandmarks" experiments/*/run.ts`
shows every experiment either matches the default or derives `numLandmarks` from the same
`DEFAULT_CONFIG` at runtime, none overrides it to a different value. Max Manhattan distance in an
8×8 grid (cells `0..7`) is `14`, so `coverage ∈ [0, 28]`, and with `collisionPenalty: 0.5`,
`reward ∈ [-4.0, 0.0]` — a fixed, single-order-of-magnitude range, not the multi-order-of-magnitude
spread `symlog` exists to compress. This is the same reasoning `reconstructionLoss`'s own doc
comment already gives for skipping `symlog` on the decoder — quoted verbatim from
`src/model/losses.ts`: *"without that paper's `symlog` transform (this environment's `Observation`
is already normalized to a fixed small range, so there's no multi-order-of-magnitude spread for
`symlog` to compress ...)"*. That reasoning applies identically here; `0013`'s aside should be read
as superseded by this derivation, not as a still-open recommendation. **Proposed: a plain linear
head + `rewardLoss` (mean squared error, below), no `symlog`.** If a future arm or environment
change widens the reward's range, this decision should be revisited against that environment's
actual numbers, not reapplied by default just because DreamerV3 always uses `symlog`.

## Loss — `rewardLoss` (`src/model/losses.ts`, proposed)

```ts
export function rewardLoss(
  predicted: tf.Tensor2D,
  target: tf.Tensor2D,
  rewardScale: number,
): tf.Scalar {
  return tf.tidy(() => {
    const scale = tf.scalar(rewardScale);
    return tf.mean(tf.square(tf.sub(tf.div(predicted, scale), tf.div(target, scale)))) as tf.Scalar;
  });
}
```

Mean squared error between the predicted and actual scalar reward, each divided by `rewardScale`
first — the same simplification `reconstructionLoss` already uses (negative log-likelihood under
a unit-variance isotropic Gaussian, up to an additive constant and a `0.5` factor), specialized to
a `[batch, 1]` target instead of `[batch, observationSize]`, plus the normalization decided below.
Not implemented by calling `reconstructionLoss` with `observationSize: 1` — kept as a distinct
named function per this codebase's existing one-loss-per-head convention (`reconstructionLoss`,
`continueLoss`, now `rewardLoss`), so a future change to one head's loss shape doesn't silently
reach into another's.

### Loss magnitude vs. reconstruction/continue — decided this run (PR #74 review item 5)

Order-of-magnitude estimate, `self_checked, medium confidence` — reasoning only, since `RewardHead`
doesn't exist yet to produce a real number; the PR #74 review's own clarification is that this run
decides from the estimate, and a *measured* first-step loss breakdown is the implementation run's
job, checked against this decision rather than replacing it:

- `continueLoss`: sigmoid cross-entropy on a two-class target, O(0.7) at random init (`≈ -ln(0.5)`).
- `reconstructionLoss`: negative log-likelihood on `Observation`, already normalized to a small
  fixed range, O(0.1) at init (per `0011`'s own precedent quoted above).
- `rewardLoss` unweighted against a raw target in `[-4.0, 0.0]`, from a zero-initialized linear
  head (prediction ≈ 0 at init): squared error starts near `target²`, up to `16` at the range's
  extreme — O(1-16), one to two orders of magnitude above the other two terms. Coefficient 1
  (as first proposed above) would let reward-prediction error dominate the total world-model
  gradient early in training — the imbalance the review flagged.

**Decided: normalize by a config-derived `rewardScale`, don't accept unweighted.** Divide both
`predicted` and `target` by `rewardScale` before squaring — the same "already normalized to a
small range" strategy `reconstructionLoss` uses for `Observation`, applied to reward instead of
leaving the coefficient at 1 by default. `rewardScale` is derived from the same
`computeReward()` bound this doc's "Reward magnitude" section already worked out, generalized
from the specific `4.0` instance to the config it came from — the number a future config change
must be re-checked against, not a hardcoded constant:

```ts
// GridworldConfig fields; reproduces computeReward()'s worst case exactly, not just approximately —
// maxManhattanDistance in a gridSize×gridSize grid (indices 0..gridSize-1) is 2*(gridSize-1)
const rewardScale =
  config.numLandmarks * (2 * (config.gridSize - 1)) / config.gridSize + config.collisionPenalty;
// DEFAULT_CONFIG (gridSize: 8, numLandmarks: 2, collisionPenalty: 0.5): 2*(2*7/8) + 0.5 = 4.0 —
// matches this doc's earlier bound exactly, not merely approximately (that bound elided the
// `(gridSize-1)/gridSize` correction as a large-gridSize approximation; this form doesn't need to).
```

With this normalization, the reward-loss term starts within the same O(0.1-1) neighborhood as
`continueLoss`/`reconstructionLoss` rather than dominating them, at any `gridSize`/`numLandmarks`/
`collisionPenalty` this project's configs currently use or are likely to vary. `rewardScale` is
computed once at `WorldModel` construction from the `GridworldConfig` it's built with (not
per-call), same lifecycle as any other config-derived constant already closed over by `step()`.

### Measured — the "one to two orders above" premise didn't hold (PR #77 review, 2026-09-21)

`RewardHead`/`rewardLoss` landed via PR #77 (2026-09-20/21). That PR's merge review reproduced a
first-step loss breakdown and found it didn't match the estimate above: *"normalized `rewardLoss`
median 0.031 (max 0.146) vs `continueLoss` median 0.58 — O(0.01–0.1), not the O(0.1–1) asked for.
Raw MSE median 0.50 is the same order as `continueLoss`, so the 'one to two orders above' premise
... didn't hold; `0014` normalized by the −4.0 bound while typical rewards are −0.6 to −0.9."* The
review also noted PR #77's own reproduction attempt didn't reproduce even the reviewer's own
numbers, because the step `Rng` seed was unstated and the measurement script wasn't committed.
This addendum is this run's fix for that: every seed below is stated, and the script that produced
these numbers is committed at `scripts/measure-reward-loss-magnitude.ts`.

**Method**: 24 seed combos (seed `0..23`, each split via `deriveSeed` into independent
env-spawn/action-selection/model-init/step-`Rng` streams — see the script for the exact salts).
Per combo: a fresh `CooperativeGridWorld` at `DEFAULT_CONFIG` is reset, one action per agent is
drawn from the action-selection stream, `env.step()` is called once, and a fresh `WorldModel`
(`rewardScale` from `env.rewardScale`, i.e. `4.0`) runs one `step(..., train: false, ...)` on
agent 0's transition using the step-`Rng` stream — an untrained, first-real-transition
measurement, matching this doc's own framing ("a measured first-step loss breakdown is the
implementation run's job").

**Results** (`self_checked, medium confidence` — one measurement design, not swept across configs
or training progress; order-of-magnitude-consistent with, but not identical to, the PR #77
reviewer's own independent sweep, whose exact seed/action scheme wasn't preserved):

| quantity | median | max |
| --- | --- | --- |
| reward (raw, `[-4.0, 0.0]` bound) | −0.9375 | (min) −1.8750 |
| `continueLoss` | 0.6834 | 0.9193 |
| `rewardLoss` (normalized) | 0.0677 | 0.1792 |
| `rewardLoss` (raw, unnormalized MSE) | 1.0827 | 2.8668 |

This run's reward magnitudes (median −0.94, min −1.88) skew larger than the PR #77 review's
"typical rewards are −0.6 to −0.9" — expected, since every sample here is the *first* step after
`reset()` (freshly spawned, plausibly far from every landmark), not a mid-episode reward after
agents have had time to close in. That difference in method, not a disagreement in the underlying
claim, is the likely source of this run's raw-MSE median (1.08) running somewhat above the PR #77
reviewer's own reported 0.50 — both are nonetheless the same order of magnitude as `continueLoss`
(0.68), which is the load-bearing part of the finding.

**Conclusion, corrected from the original estimate above**: normalized `rewardLoss` runs
O(0.01–0.1) — one order of magnitude *below* `continueLoss`'s O(0.1–1), not matching it as
originally intended, and raw (unnormalized) MSE runs in the *same* order of magnitude as
`continueLoss` (1.08 vs. 0.68, within ~1.6×), not "one to two orders above" it. The original
estimate's error: it reasoned from the range's worst case (`target²` up to `16` at the `−4.0`
extreme), but actual — even first-step, freshly-reset — rewards run well inside that bound, so raw
squared error never approaches the extreme the estimate anchored on. **Decision, unchanged**: keep
the `rewardScale` normalization regardless of this correction. The PR #77 review's own reasoning
for keeping it stands: "Adam makes the head's own weights coefficient-insensitive; only the
shared-trunk weighting shifts" — normalization still keeps the term in a bounded, config-derived
range instead of an arbitrary raw one, independent of whether the original magnitude estimate that
motivated adding it turns out to have been off. `src/model/losses.ts`'s `rewardLoss` doc comment is
corrected to match this measurement (this run).

## Wiring into `WorldModel` (`src/model/worldModel.ts`, proposed) — same pattern as `0011`

- **Constructor**: `this.rewardHead = new RewardHead({ ...(config.seed !== undefined &&
  { seed: deriveSeed(config.seed, 3) }) })` — salt `3`, after cell (0) / decoder (1) / continueHead
  (2), same "one independent stream per component" reasoning `worldModel.ts:104-111` already
  documents. The constructor's throwaway warmup forward pass gains
  `this.rewardHead.predict(deterministic, posterior.sample)`, and `trainableVars` gains
  `...this.rewardHead.trainableWeights()`.
- **`step()`**: gains a `reward: number` parameter (the transition's `StepResult.reward`, supplied
  by the caller the same way `observation`/`done` already are — same reasoning `0011` gives for
  `done`). Predicted from the **post-transition** state (`nextDeterministic`/`nextStochastic`) —
  same reasoning `0011` gives for `continueHead` ("the state that has 'seen' this transition ... is
  the state whose logit should be asked to predict it"), which applies identically to reward: this
  transition's reward is a property of having taken this step, not of the state before it. Target:
  `tf.tensor2d([[reward]])`, same per-call tensor lifecycle as `continueTargetTensor`.
- **Loss total**: `reconstructionLoss + klBalancedLoss(...).total + continueLoss(...) +
  rewardLoss(predicted, target, rewardScale)` — the coefficient itself is 1 (`0011`'s own precedent
  for adding a DreamerV3 loss term without inventing a new coefficient, applied a second time), but
  unlike `0011`, the term isn't left at a raw, unweighted scale: `rewardScale`'s normalization
  (decided above) is what keeps coefficient 1 from being the same thing as "unweighted" in
  practice. What the normalization *is*, stated plainly rather than left implicit in "coefficient
  1": dividing both `predicted` and `target` by `rewardScale` before squaring is identical, in
  gradient, to applying coefficient `1/rewardScale²` to the raw (unnormalized) MSE —
  `1/4.0² = 1/16 = 0.0625` at `DEFAULT_CONFIG` — not literally coefficient 1 on anything (PR #74
  review, 2026-09-18). `rewardScale` is computed once from the constructor's `GridworldConfig` and
  closed over by `step()`, not recomputed per call.
- **`WorldModelStepResult`** gains `rewardLoss: number`. **`WorldModelNaNError`** gains a
  `rewardLoss: number` field, and the non-finite check that throws it now also covers it — same
  "any component going non-finite compounds into every later step" reasoning as the existing
  three-way (now four-way) check. **Cost worth budgeting for**: `WorldModelNaNError`'s constructor
  is positional and shorthand-free by design (`src/model/worldModel.ts:53-58`) — adding a 5th field
  (`rewardLoss`) is a signature change at *every* construction and test call site that builds this
  error, not just the one inside `step()`; an implementer should grep for `new WorldModelNaNError(`
  before starting, not discover the fan-out mid-change.
- **`src/experiment/freeze.ts`**: the one production call site (`worldModels?.[i]?.step(...)`)
  gains `result.reward` as a new argument, mirroring how `result.done` was threaded through in
  `0011`.
- **Shared reward, per-agent head.** `StepResult.reward` is "Shared (cooperative) scalar reward,
  identical for both agents" (`src/env/types.ts:56`), and Arm A gives each agent its own
  `WorldModel` (`src/experiment/freeze.ts`'s `(WorldModel | undefined)[]`) — so this wiring trains
  N independent `RewardHead`s, one per agent, on an identical target every step. That's consistent
  with every other per-agent head this project already has (`ObservationDecoder`, `ContinueHead`
  are per-agent too, on per-agent observations that happen to differ — reward is the one signal
  that's structurally identical across agents while the head predicting it isn't shared), so it's
  noted rather than treated as a defect. Cross-links `docs/explainers/0013`'s open question 5
  (per-agent vs. shared actor-critic) — the same question one layer up, also left open there.

## What this means for `docs/explainers/0013`'s actor-critic spec

Two corrections noted here for `0013`, one already carried, one still pending:

1. **Already carried** (PR #74 review's items 1-2, applied to `0013` this run): the
   training-procedure sketch now converts `rewardHead.predict(...)` to a plain number at the point
   of the call (`rewardsNumeric.push(rewardHead.predict(...).dataSync()[0])`), the same treatment
   `values`/`continues` now get explicitly — `computeLambdaReturns`'s `rewards` parameter is
   `number[]` per `0012`'s pinned contract. Recorded here so this entry doesn't keep describing a
   fix `0013` has already made as still-pending.
2. **The WM/AC gradient-separation test** (`0013`, "The WM/AC gradient-separation invariant, as a
   concrete test," steps 2/4) snapshots `cell`/`decoder`/`continueHead`'s `trainableWeights()` —
   written before this head existed. Once `RewardHead` lands, that snapshot needs a fourth set,
   `worldModel.rewardHead.trainableWeights()`, held to the same "must not move" assertion: reward
   prediction inside imagination must read `rewardHead.predict()` without including
   `rewardHead.trainableWeights()` in the actor/critic's `tf.variableGrads` `varList` — the same
   leak class the test's step 5 already guards `cell`/`decoder`/`continueHead` against, now with
   one more component that could leak into it.

## Test coverage (proposed, mirrors `docs/explainers/0011`'s continue-head coverage)

- `test/model/rewardHead.test.ts`, new: `predict()` returns `[batch, 1]`; `trainableWeights()`
  non-empty after one `predict()` call; same seed reproduces identical initial weights, different
  seeds diverge (mirrors `test/model/continueHead.test.ts`'s three tests exactly).
- `test/model/losses.test.ts` gains: `rewardLoss` is near-zero when `predicted === target` (any
  `rewardScale`); large for a confidently-wrong prediction; gradient flows into `predicted`
  (`tf.variableGrads` against a `tf.Variable` prediction is nonzero); a fixed absolute error scales
  down as `rewardScale` grows (catches a `rewardScale`/coefficient wiring bug that silently drops
  the normalization decided above).
- `test/model/worldModel.test.ts`: every existing `wm.step(...)` call site gains a `reward`
  argument; the "reconstructionLoss + klLoss + continueLoss equals loss" test extended to include
  `rewardLoss`; new tests: `train=true` moves at least one `RewardHead` weight, `train=false` moves
  none (mirrors the existing decoder/continueHead weight-movement tests).

## What's deliberately not here

- **No implementation.** Every type above is a proposed contract — `src/` is unchanged in this
  increment, matching `0012`/`0013`'s "spec only" treatment, per the human's PR #74 review picking
  this scope over an implementation-scope increment.
- **No `Actor`/`Critic`, no imagination wiring.** This closes `0013`'s blocking dependency in spec
  form only; building the actor-critic against it is still Gate G2 role-flip territory
  (`loop/GOAL.md` priority 6) plus a future loop increment for the non-reserved pieces
  (`Actor`/`Critic` classes themselves, per `0013`'s own architecture section).
- **No revisit of `0013`'s open questions 4–5** (entropy coefficient default/schedule, per-agent
  vs. shared actor-critic) — question 3 (reward head design) is what this entry answers; 4 and 5
  are unrelated to it and left exactly as `0013` stated them.
