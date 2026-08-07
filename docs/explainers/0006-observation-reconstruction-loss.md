# Explainer: observation reconstruction loss (`L_pred`)

`src/model/decoder.ts`, `src/model/losses.ts`, `src/model/worldModel.ts` — `loop/GOAL.md` priority
3's second sub-increment ("RSSM completion... observation reconstruction"), the last piece
`docs/explainers/0004-kl-balanced-world-model-loss.md` and `docs/explainers/0005-world-model-rollout-wiring.md`
both deliberately deferred. Written explainer-before-implement per `loop/GOAL.md`. Processes PR #39's
review (@SakkarinKt, 2026-08-06): "Next: priority 3 sub-increment 2 (observation reconstruction loss
/ L_pred)."

## Scope of this sub-increment

`docs/explainers/0004`'s "What's deliberately not here yet" named the full per-timestep loss as
`L_pred + betaDyn * L_dyn + betaRep * L_rep` (DreamerV3's form) and said only the KL-balanced pair
was built so far. This entry adds the missing `L_pred` term — a decoder network reconstructing the
observation from `(h_t, z_t)`, a reconstruction loss between that and the real observation, and
wiring both into `WorldModel.step()`'s total loss alongside the existing `klBalancedLoss`. No change
to `RSSMCell`, `klBalancedLoss`, or `src/experiment/freeze.ts`'s rollout-level wiring (already done,
PR #39) — this sub-increment only fills in the one term `WorldModel.step()`'s loss was missing.

## What it is

**`ObservationDecoder`** (`src/model/decoder.ts`), new: one `tf.layers.dense` layer, `units:
observationSize`, linear (no activation — this is a regression head, not a classifier), taking
`concat([deterministic, stochastic], 1)` and returning a `[batch, observationSize]` reconstruction.
Deliberately *not* a method on `RSSMCell`: DreamerV3's architecture treats the RSSM (deterministic
recurrence + prior/posterior over the stochastic latent) as the dynamics core, with the decoder,
reward head, and continuation head as separate networks reading off `(h_t, z_t)` — `RSSMCell`
already only builds the first of those. Kept as a single dense layer, not a deeper MLP, to match
`priorDense`/`posteriorDense`'s existing minimalism in `RSSMCell` — this environment's
`Observation` is already a small (~9-13 element, `src/env/types.ts`), pre-normalized, hand-engineered
feature vector, not raw pixels, so there's no established need for decoder depth yet; revisit only if
reconstruction quality turns out to need it.

**`reconstructionLoss(predicted, target)`** (`src/model/losses.ts`), new: `mean_batch( sum_features(
(predicted - target)^2 ) )` — sum of squared error per row, then batch mean, mirroring
`categoricalKL`'s existing "reduce over the per-example axes, then batch-mean" shape (`docs/explainers/0004`).
This is (up to an additive constant and a `0.5` factor that doesn't change what minimizes it) the
negative log-likelihood of `target` under a unit-variance isotropic Gaussian centered at `predicted`
— the standard simplification of DreamerV3's `L_pred = -log p(o_t | h_t, z_t)` for a continuous
observation vector, dropped by exactly the amount discussed below.

**`WorldModel.step()`** (`src/model/worldModel.ts`), changed: `forward()` now also calls
`this.decoder.decode(deterministic, nextStochastic)` — the *posterior's* sample (`nextStochastic`,
teacher-forced, same as what the KL loss's `repLoss` term trains toward and what the persisted
`RSSMState` carries forward), not the prior's — and adds `reconstructionLoss(predicted,
observationTensor)` to the scalar returned to `tf.variableGrads`/the eval path. `total = recon +
klBalancedLoss(...).total`, unweighted (coefficient 1 on `recon`), matching DreamerV3's own
`L_pred + betaDyn*L_dyn + betaRep*L_rep` form exactly — `betaDyn`/`betaRep` already discount the KL
terms relative to reconstruction; `L_pred` itself carries no separate coefficient in the paper's
loss, and none is introduced here. `WorldModelStepResult` gains `reconstructionLoss`/`klLoss` fields
(both plain numbers) alongside the existing `loss` total, so callers/tests can inspect the breakdown
without recomputing it — `klBalancedLoss` already returns this same kind of breakdown
(`dynLoss`/`repLoss`/`total`) for the same reason.

The decoder's weights are added to `WorldModel`'s existing `trainableVars` list (alongside the
cell's), and the constructor's warmup forward pass now also calls `decoder.decode(...)` once so its
layer is built before `trainableVars` is captured — same reason `RSSMCell.prior()`/`.posterior()`
are warmed up there already (`src/model/worldModel.ts`'s constructor comment).

## Why the unit-variance-Gaussian simplification, not DreamerV3's symlog transform

DreamerV3's own decoder loss is not a bare Gaussian NLL — for continuous-valued targets it applies a
`symlog` transform (`sign(x) * log(1 + |x|)`) before an MSE-style distance, specifically to handle
targets that can span multiple orders of magnitude (their headline use case: raw pixel intensities
and rewards ranging from single digits to thousands). This project's `Observation` vector
(`src/env/types.ts`) is already normalized: every component is either a `{0, 1}` visibility flag or a
position/offset divided by `gridSize`, so every value sits in a fixed, small range (`[-1, 1]` or
`[0, 1]`) by construction — there is no multi-order-of-magnitude spread for `symlog` to compress.
Adding it here would be copying a mechanism from the paper without the problem it exists to solve.
Plain squared error is the defensible simplification for this specific observation space —
**[self_checked, medium confidence]**: the reasoning is direct (bounded-range inputs don't need a
compressive transform) rather than derived from the paper, and it hasn't been checked against a
primary DreamerV3 source this run (same standing autonomous-run caveat as `docs/explainers/0004`'s
KL-loss defaults — `notes/lit-map.md`). If a later increment adds observation components with
unbounded or wide-magnitude range (raw rewards, unnormalized counts), this simplification should be
revisited alongside a fresh primary-source check on the transform DreamerV3 actually specifies for
each target type (it's not one uniform loss across pixels/reward/continuation in the original paper).

## Why the posterior's sample, not the prior's

`WorldModel.step()` already persists `nextStochastic = posteriorDist.sample` as the carried-forward
`RSSMState` (`docs/explainers/0005`, unchanged by this sub-increment) — the posterior is the
observation-conditioned ("encoder") distribution, teacher-forced during training exactly so the
decoder has a genuinely observation-informed `z_t` to reconstruct from, matching every other
DreamerV3-style world model's training-time convention (imagination/prior-only rollouts, which would
use `priorDist.sample` instead, are out of scope — `docs/explainers/0005`'s "what's deliberately not
here yet" already excludes imagination rollouts from this milestone). Decoding from the prior's
sample instead would ask the reconstruction loss to also supervise the dynamics prediction itself,
conflating two different training signals the KL-balanced loss already separates on purpose.

## Interaction with the BPTT-horizon-1 truncation (PR #39 review follow-up 1)

PR #39's review asked `docs/explainers/0005` to explicitly document a design point it left implicit:
`WorldModel.step()`'s single-transition-per-call structure means every step's loss can only
backpropagate through that step's own `h_{t-1} -> h_t` transition — a BPTT horizon of 1, despite
`RSSMCell.step()` itself being fully capable of chained multi-step differentiation. See
`docs/explainers/0005`'s 2026-08-07 amendment for the full explanation and why it doesn't block
priority 4. This sub-increment doesn't change that horizon — `reconstructionLoss` is computed inside
the same single-step `forward()` closure the amendment describes, adding a new loss term to an
existing single-step computation, not a new time axis.

## What's deliberately not here yet

- **Reward and continuation/termination prediction heads.** DreamerV3's full `L_pred` also includes
  `-log p(r_t | h_t, z_t)` and `-log p(c_t | h_t, z_t)` terms; this environment's reward/termination
  signals aren't wired into any world-model loss yet. Out of scope for this sub-increment (observation
  reconstruction only, per `loop/GOAL.md` priority 3's own phrasing: "the world-model losses (KL
  balancing... observation reconstruction)"). The continue/termination head is explicitly listed under
  `loop/GOAL.md` priority 5 (vertical-slice hardening's invariant tests), not here.
- **Sequence/time-batching.** Same scope note as `docs/explainers/0004` and `0005` — still one
  timestep, one transition, per call. See the BPTT-horizon section above for what this implies.
- **Decoder depth/architecture tuning.** The single-dense-layer choice above is a placeholder matched
  to `RSSMCell`'s existing minimalism, not a validated architecture — same status as `WorldModel`'s
  learning-rate default (`docs/explainers/0005`).
- **Per-agent `Rng` for `WorldModel`** (PR #39 review follow-up 2 — sharing `runEpisode`'s single
  `Rng` across all agents' world models means changing one agent's `latentCategoricals`/`latentClasses`
  moves the environment trajectory at a fixed seed, since world-model sampling and env/policy sampling
  interleave into the same stream). The review left this "your call, ride along or wait for priority
  4." Deferred to priority 4's metric plumbing rather than folded in here, to keep this sub-increment
  scoped to the one loss term `loop/GOAL.md` names — recorded under this run's "Assumptions made" in
  `reports/standup/2026-08-07.md`, not silently dropped.

## Test coverage

`test/model/losses.test.ts` gains: `reconstructionLoss` is exactly zero for identical
predicted/target tensors; strictly positive and equal to the hand-computed sum-of-squares for a
known mismatch; sums over the feature axis before batch-meaning (checked the same two-row-vs-
single-row-times-2 way `categoricalKL`'s equivalent test already is, `docs/explainers/0004`'s test
coverage section).

`test/model/decoder.test.ts`, new: constructing an `ObservationDecoder` and calling `decode()`
returns the expected `[batch, observationSize]` shape; `trainableWeights()` is non-empty after one
`decode()` call (mirrors `RSSMCell.trainableWeights()`'s own "only meaningful after build" test
pattern, `test/model/rssm.test.ts`).

`test/model/worldModel.test.ts` gains: `WorldModelStepResult.reconstructionLoss` and `.klLoss` sum to
`.loss`; the repeated-identical-input "loss goes down" test (already existing, `freeBits: 0`) is
joined by an equivalent check that `reconstructionLoss` specifically goes down across the same
repeated-identical-input run (isolating the new term's own learning signal from the KL terms', same
`freeBits: 0` isolation rationale `docs/explainers/0004`'s gradient-separation test already uses); the
existing tensor-leak and train/frozen-weight-identity tests are unchanged in structure but now also
cover the decoder's weights (frozen ⇒ decoder weights bit-identical too, not just the cell's).
