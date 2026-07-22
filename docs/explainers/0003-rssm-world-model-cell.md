# Explainer: the RSSM world-model cell — deterministic recurrence + stochastic categorical latent

`src/model/rssm.ts` — proposal `0001`'s Arm-A milestone's world-model cell, per PR #14's review
(@SakkarinKt, 2026-07-14: "go ahead and start the RSSM world-model cell... split the cell as you
proposed: struct/forward-pass, then STE + gradient-check") and PR #19's review (2026-07-19:
"resume the RSSM cell increment: prior/posterior heads, straight-through categorical sampling,
gradient-check test"). Landed in two sub-increments:

1. **2026-07-17**: the GRU-based deterministic recurrent state and its update rule (struct/forward-pass
   only).
2. **2026-07-20** (this entry): the stochastic categorical latent — prior `p(z_t | h_t)` and
   posterior `q(z_t | h_t, o_t)` heads, straight-through sampling, and a finite-difference
   gradient-check test.

## What it is

`RSSMCell` wraps `tf.layers.gruCell` inside a `tf.layers.rnn` layer (`returnState: true,
returnSequences: false`) for the deterministic path, plus two `tf.layers.dense` heads (`prior`,
`posterior`) over a categorical stochastic latent. `RSSMConfig` is now `{ deterministicSize,
latentCategoricals, latentClasses }` — the latent is `latentCategoricals` independent categorical
variables, each with `latentClasses` classes (DreamerV3-style), flattened to
`latentCategoricals * latentClasses` wherever it's stored or fed as an input. `RSSMState` now holds
both `deterministic` (`[batch, deterministicSize]`) and `stochastic` (`[batch, latentCategoricals *
latentClasses]`, the most recently sampled latent — zero for the initial state).

The recurrence, matching DreamerV2/V3-style RSSMs (`notes/rssm-vs-ssm-implementation-robustness.md`
§3): `h_t = GRU(h_{t-1}, [onehot(a_{t-1}), z_{t-1}])`, then either `prior(h_t)` (imagination, no
observation) or `posterior(h_t, o_t)` (training, conditions on a real observation) produces `z_t`.
`step(prevState, actions)` computes only `h_t` — it takes `prevState.stochastic` as `z_{t-1}` but
does **not** return a full `RSSMState`, since which of `prior`/`posterior` supplies `z_t` is the
caller's choice, not the cell's. Callers assemble the next `RSSMState` themselves:
`{ deterministic: rssm.step(state, actions), stochastic: rssm.prior(h_t).sample }` (or
`.posterior(h_t, obs).sample` when an observation is available) — see `advance()` in
`test/model/rssm.test.ts` for the pattern.

## Straight-through categorical sampling

`sampleStraightThrough(logits)` draws a hard one-hot sample per categorical (`sampleHard`, via
`tf.multinomial`) and combines it with the soft distribution (`tf.softmax`) via
`straightThroughEstimator(logits, hard)`: forward pass returns exactly `hard`; backward pass,
gradients w.r.t. `logits` equal `softmax(logits)`'s own gradient (computed directly from the
softmax Jacobian-vector product, not a nested `tf.grad` call — see the code comment on why that
alternative doesn't work inside `tf.customGrad`'s `gradFunc`), and gradients w.r.t. `hard` are zero.

**This corrects a standing claim.** `notes/adr-0002-js-ml-stack.md` §4 and
`notes/rssm-vs-ssm-implementation-robustness.md` §4 both described this as needing only
`tf.stopGradient`, "a native, stable TF.js primitive." Writing this code against the actual pinned
`@tensorflow/tfjs-node@4.22.0` found that claim false — `tf.stopGradient` doesn't exist on that
package (or `tfjs-core`/`tfjs`, same version); calling it throws `tf.stopGradient is not a
function`. `tf.customGrad()` (§4 of the ADR notes already names this as the alternative mechanism)
is what's actually used. See `notes/adr-0002-js-ml-stack.md`'s 2026-07-20 entry for the full
correction — this is the first point in the ADR-0002 research thread where a claim was checked by
running the pinned dependency's code rather than searching about it, and search-only verification
turned out not to be enough.

## Why these specific choices

- **`tf.customGrad`, not `tf.stopGradient`.** See above — not a choice so much as a correction once
  the originally-planned primitive turned out not to exist in this pinned version.
- **Finite-difference check targets `softmax(logits)`, not `straightThroughEstimator`'s own forward
  pass.** The estimator's forward value (`hard`) is piecewise-constant in `logits` by
  construction — a discrete sample supplied as a separate argument — so
  `(f(x+e) - f(x-e)) / 2e` on that forward pass is identically zero regardless of whether the custom
  gradient is correct; it isn't measuring anything. The backward pass is *defined* to reproduce
  `softmax(logits)`'s gradient, so that's the finite-difference check's actual target. A companion
  test (`straightThroughEstimator: finite-differencing its own forward pass is identically zero`)
  pins down *why*, rather than leaving it as an unexplained oddity if someone tries the naive check
  later.
- **`hard`/`soft` reduce over the last axis (`-1`), independent of rank.** `straightThroughEstimator`
  and `sampleHard` work on the `[batch, categoricals, classes]` logits the cell actually produces,
  but the estimator itself is written rank-agnostically (tested directly against plain 2D tensors in
  `test/model/rssm.test.ts`) since the straight-through math doesn't care about the categorical
  structure above the sampled axis.
- **`step()` returns `h_t` alone, not a full `RSSMState`.** Bundling `prior()`'s sample into
  `step()`'s return would silently pick imagination-style sampling as the default and make the
  posterior path (observation-conditioned, needed for training) awkward to reach without discarding
  work. Keeping them separate means the caller's choice of prior vs. posterior is explicit at every
  step, matching how DreamerV2/V3 code structures `img_step` vs `obs_step`.
- **`oneHotActions` and `sampleHard` both cast to float32.** `tf.oneHot` defaults to int32;
  `tf.concat`ing an int32 one-hot action encoding against the float32 stochastic latent (or vice
  versa, once a sampled `z_{t-1}` becomes the next step's input) throws a dtype-mismatch error from
  the backend — caught by running the cell's own tests, not by inspection.
- **No batch-size or shape validation added.** Same rationale as the deterministic-only increment:
  mismatched shapes produce a `tf.oneHot`/`reshape`/`concat` error direct from TF.js rather than a
  custom check duplicating what the framework already enforces.

## What's deliberately not here yet

Any observation encoder (the posterior head takes a raw `[batch, obsSize]` observation tensor
directly — no encoder network), loss functions (KL-balancing between prior and posterior, image/
observation reconstruction), training code, and any wiring into `src/experiment/freeze.ts`'s
rollout loop. Tensor lifecycle management (`tf.tidy`/explicit `.dispose()`) is still not addressed —
same reasoning as the prior sub-increment: no training loop exists yet to make tensor-count growth
a real cost, and premature disposal risks disposing a tensor a caller still needs. Worth revisiting
once the cell is wired into an actual multi-episode rollout.

## Update (2026-07-22): the deterministic path no longer goes through `tf.layers.rnn`

**Processing PR #22's review** (the week-3 stack-validation spike's follow-up — see
`notes/adr-0002-js-ml-stack.md` §9): "What it is" above, written 2026-07-20, described `step()` as
wrapping `tf.layers.gruCell` inside a `tf.layers.rnn` layer. That wrapper turned out to crash
`tf.variableGrads` whenever `step()`+`prior()`/`posterior()` is chained across ≥2 timesteps in one
differentiation trace (`tensorflow/tfjs#1529`, `#3550` — a still-open tfjs-layers bug, not this
repo's) — every real multi-step rollout, since `z_{t-1}` feeding `step()` comes from the *previous*
step's hidden state by construction.

`step()` now calls `this.cell.call([recurrentInput, prevDeterministic], {training: false})`
directly — the same lower-level method `tf.layers.rnn`'s own `RNN.call()` calls internally, one
timestep at a time — bypassing the wrapper's sequence-handling machinery (`tfc.unstack`/`tfc.stack`,
which is where the bug traces to) entirely. `GRUCell.call()` is plain differentiable tensor ops
(`tf.split`/matmul/activation), so nothing about the underlying GRU math changed — only which layer
of tfjs-layers gets called. The cell's weights now build lazily on `step()`'s first call via an
explicit `cell.build([null, recurrentInputWidth])`, since `GRUCell.build()` expects a single input
`Shape`, not the `[input, state]` shape pair `Layer.apply()` would infer.

Net effect: the week-3 spike's kill criterion (ADR-0002 decision 5) did not fire. No custom autograd
was needed — `test/model/rssm.test.ts` now has a passing finite-difference check chaining `step()` +
`prior()` across multiple timesteps, plus a standalone regression test pinning the underlying
`tf.layers.rnn` bug directly (independent of `RSSMCell`, in case future code reaches for the wrapper
again).

## Test coverage

`test/model/rssm.test.ts`, 44/44 passing (`npm test`; 35 prior + 9 new): `initialState`'s zero-fill
for both `deterministic` and `stochastic`; `step`'s output shape/determinism/sensitivity/batch-row
independence (same coverage as the deterministic-only increment, now exercised through the grown
`[onehot(action), z_{t-1}]` recurrent input); multi-step chaining via `prior()` (a local `advance()`
helper standing in for the not-yet-built rollout wiring) keeps stable shapes and diverges across
different action sequences; `prior`/`posterior` output shapes, softmax rows summing to 1, and hard
samples being exactly one-hot per categorical; `posterior` producing different logits for different
observations; `straightThroughEstimator`'s forward value equalling the fixed `hard` sample exactly;
its analytic gradient matching a finite-difference check against `softmax(logits)`; the naive
finite-difference-on-its-own-forward-pass being identically zero (documents the check above);
gradient independence from which `hard` sample was fixed; `sampleHard` producing exactly one 1 per
categorical row.
