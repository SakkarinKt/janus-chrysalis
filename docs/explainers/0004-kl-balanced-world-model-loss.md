# Explainer: KL-balanced world-model loss (free-bits floor)

`src/model/losses.ts` — proposal `0001`'s Arm-A milestone, `loop/GOAL.md` priority 3 ("RSSM
completion... the world-model losses"), first sub-increment. Written explainer-before-implement per
`loop/GOAL.md`.

## Scope of this sub-increment

Priority 3 groups three things under "RSSM completion": the KL-balanced dynamics/representation
loss (this entry), observation-reconstruction loss, and wiring `RSSMCell` into
`src/experiment/freeze.ts`'s rollout. Splitting these into separate day-increments follows the same
pattern the cell itself was built under — PR #14's review split it into "struct/forward-pass" then
"STE + gradient-check" as two dated sub-increments (`docs/explainers/0003-rssm-world-model-cell.md`).
This entry is the KL-balancing term only. Reconstruction loss and rollout wiring are **not** here —
see "What's deliberately not here yet" below.

## What it is

`klBalancedLoss(prior, posterior, config?)` in `src/model/losses.ts` computes DreamerV3-style
KL-balanced dynamics/representation losses between an `RSSMCell`'s `prior()` and `posterior()`
outputs (`LatentDistribution.probs`, shape `[batch, latentCategoricals, latentClasses]`) for one
timestep:

- `dynLoss = mean_batch( max(freeBits, KL[ sg(posterior.probs) || prior.probs ]) )` — trains the
  prior (dynamics predictor) toward the (stop-gradiented) posterior. Gradients flow into the prior's
  parameters only.
- `repLoss = mean_batch( max(freeBits, KL[ posterior.probs || sg(prior.probs) ]) )` — trains the
  posterior (representation/encoder) toward the (stop-gradiented) prior, weighted down relative to
  `dynLoss` so the representation isn't dragged toward an undertrained prior early in training.
  Gradients flow into the posterior's parameters only.
- `total = betaDyn * dynLoss + betaRep * repLoss`, defaults `betaDyn = 1.0`, `betaRep = 0.1`,
  `freeBits = 1` (nats).

Both KL terms are the categorical KL divergence summed over `latentClasses` (per categorical
variable) and then over `latentCategoricals` (the DreamerV3 latent is `latentCategoricals`
independent categoricals, so their KLs sum), giving one scalar per batch row before the free-bits
clip and the batch mean.

## Why these specific choices

- **Formula and defaults**: DreamerV3 (Hafner et al., arXiv:2301.04104, "Mastering Diverse Domains
  through World Models"), the same paper this project's RSSM cell already follows
  (`docs/explainers/0003-rssm-world-model-cell.md`). **[medium confidence, self_checked]** — primary-source
  fetch (`arxiv.org/abs/2301.04104`, `ar5iv.labs.arxiv.org`, `arxiv.org/pdf/2301.04104`) 403'd from
  this autonomous run's environment, consistent with `notes/lit-map.md`'s standing tooling caveat
  ("`WebFetch` returns 403 for arXiv... treated as a persistent constraint for autonomous runs";
  PR #1 review). Cross-checked via `WebSearch`, which returned a snippet quoting the paper directly
  (query: "DreamerV3 arXiv 2301.04104 free bits KL balancing loss... equation") giving:
  `L_dyn(t) = max(1, KL[sg(q_phi(z_t|h_t,e_t)) || p_phi(z_t|h_t)])`,
  `L_rep(t) = max(1, KL[q_phi(z_t|h_t,e_t) || sg(p_phi(z_t|h_t))])`, `beta_dyn = 1`, `beta_rep = 0.1`
  — matching independent recollection of the paper and a second `WebSearch` summary run. Capped at
  medium (not high/primary-source) per the lit-map convention, since no fetch of the primary PDF/HTML
  succeeded this run. **Flagging for a primary-source pass** in a future interactive (non-autonomous)
  session, same as other ADR-0002 claims that started as search-only and were later corrected by
  direct inspection (`notes/adr-0002-js-ml-stack.md` §4/§5's `tf.stopGradient` correction is exactly
  this failure mode once — worth not repeating here without eventually checking).
- **Free-bits floor applied per-batch-row, before the mean, not after.** The clip is on each row's
  total (summed-over-latent) KL individually (`max(1, KL_row)`), then the batch mean is taken of the
  clipped values — not `max(1, mean(KL))`. This matches the searched formula (`max(1, KL[...])` is a
  per-timestep, i.e., per-batch-row-here, scalar) and matters mechanically: clipping after the mean
  would let some rows sit under the 1-nat floor as long as the batch average clears it, defeating the
  floor's purpose (preventing any individual latent from collapsing to a near-zero-information prior
  match). `categoricalKL` therefore returns a `[batch]` tensor (not a scalar), summed over both
  `latentClasses` and `latentCategoricals` per row, and the `tf.maximum(freeBits, ...)` clip is
  applied to that `[batch]` tensor before `tf.mean`.
- **`stopGradient` reimplemented via `tf.customGrad`, not `tf.stopGradient`.** Same reason as
  `straightThroughEstimator` (`docs/explainers/0003-rssm-world-model-cell.md`): `tf.stopGradient`
  does not exist on this project's pinned `@tensorflow/tfjs-node@4.22.0` (confirmed there by
  actually calling it, not by search). `stopGradient(x)` here is `tf.customGrad` wrapping the
  identity forward pass with an all-zero `gradFunc` — the standard `stopGradient` mechanics reduced
  to `customGrad`'s primitive, mirroring the ADR-0002 §4 note that `customGrad` is the documented
  fallback for exactly this.
- **`categoricalKL` guards `log` with an epsilon floor**, not a bare `tf.log(probs)`. `probs` comes
  from `tf.softmax`, so it's mathematically `> 0`, but float32 can still underflow a low-probability
  class to exactly `0.0`, and `log(0) = -Infinity` would propagate a `NaN` through the KL sum (`0 *
  -Infinity` is `NaN`, not `0`, in IEEE float arithmetic) — the same failure class as `sampleHard`'s
  `GUMBEL_UNIFORM_EPSILON` guard (`src/model/rssm.ts`) for the same reason. Reuses the same `1e-7`
  epsilon value for consistency, not because the two guards share a derivation.
- **No masking/discount for terminated episodes.** This function operates on one timestep's
  already-batched `LatentDistribution` pair — it has no notion of episode boundaries. Whatever calls
  it across a rollout (the not-yet-built wiring into `freeze.ts`) is responsible for excluding
  post-termination steps from whatever aggregate it computes across timesteps, same as any other
  per-timestep loss term here.
- **`tf.tidy`-wrapped, returns a tensor-container object.** `tf.tidy` accepts any `TensorContainer`
  return value (arrays/objects nested with tensors), not just a single tensor — used here to return
  `{ dynLoss, repLoss, total }` in one call while still disposing every intermediate (the KL sums
  pre-clip, the stop-gradiented copies, etc.) automatically. Same pattern `sampleHard` already uses
  in this codebase.

## What's deliberately not here yet

- **Observation reconstruction loss** (`-log p(o_t | h_t, z_t)`) — needs a decoder network this
  sub-increment doesn't build. Next sub-increment under priority 3, or a later run if this one draws
  a human reply first (per `loop/GOAL.md`'s priority-1 rule).
- **The overall per-timestep loss** (`L_pred + betaDyn * L_dyn + betaRep * L_rep`, DreamerV3's full
  form) — `L_pred` (reconstruction + reward + continuation prediction) doesn't exist yet in this
  codebase (no decoder, no reward/continuation heads), so only the KL-balanced pair is assembled
  here. `total` in this entry's `klBalancedLoss` is `betaDyn * dynLoss + betaRep * repLoss` only.
- **Wiring into `src/experiment/freeze.ts`'s rollout** — `runEpisode` doesn't call `RSSMCell` at all
  yet (it operates purely on `Policy.act`/`update`, agnostic to any world model). That's a separate,
  larger sub-increment (constructing prior/posterior each step from the rollout's actions/
  observations, accumulating a loss across the episode, and — per PR #37's review — making `rng`
  required rather than optional at that call site, since `prior(det)` without `rng` silently compiles
  today and only throws at runtime).
- **Sequence/time-batching.** DreamerV3 trains on `[batch, time]`-shaped sequences; this function is
  `[batch]`-only (one timestep). Extending to a time axis is naturally part of the rollout-wiring
  sub-increment, not this one, since there's no sequence-shaped data to feed it until wiring exists.
- **Training loop / optimizer wiring.** This function computes a loss value; nothing here calls
  `tf.variableGrads`/an optimizer on it. That's part of Arm-A metric plumbing (`loop/GOAL.md`
  priority 4) or the rollout-wiring sub-increment, whichever lands first.

## Test coverage

`test/model/losses.test.ts` (new): `categoricalKL` is (near-)zero for two identical distributions
and strictly positive for two different ones (Gibbs' inequality, checked directly rather than
asserted from the math); `stopGradient`'s forward value equals its input exactly, and its gradient
is exactly zero (`tf.grad` check); `klBalancedLoss`'s free-bits floor — near-identical prior/
posterior clips `dynLoss`/`repLoss` to exactly `freeBits`, not below; `klBalancedLoss` respects
`betaDyn`/`betaRep` scaling on `total`, checked against hand-computed values from `dynLoss`/`repLoss`;
**the gradient-separation invariant** — `dynLoss`'s gradient w.r.t. the posterior's underlying
logits variable is exactly zero while its gradient w.r.t. the prior's is non-zero, and vice versa for
`repLoss` w.r.t. the prior's — verified via `tf.variableGrads` over two `tf.variable()`-backed logits
tensors softmax'd into `LatentDistribution.probs`, the same `ownedVariablesAfter`/`variableGrads`
pattern `test/model/rssm.test.ts` already uses. This last one is the actual point of KL balancing
(each term training only its own side); getting the `stopGradient` placement backwards would still
produce a numerically plausible loss value while silently breaking this, so it needs its own direct
check rather than relying on the free-bits/scaling tests to catch it indirectly.
