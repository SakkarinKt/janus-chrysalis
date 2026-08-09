# Explainer: wiring the RSSM world model into the rollout

`src/model/worldModel.ts` and `src/experiment/freeze.ts` — `loop/GOAL.md` priority 3's third
sub-increment ("RSSM completion... wiring `RSSMCell` into `src/experiment/freeze.ts`'s rollout"),
done ahead of the second (observation-reconstruction loss) per PR #38's review (@SakkarinKt,
2026-08-05: "Next: priority 3 sub-increment 3 (rollout wiring) ahead of reconstruction loss — it
also closes PR #37's two open follow-ups"). Proposal `0001`'s Arm-A scope: **independent per-agent
world models, no sharing** — this increment gives each agent its own `RSSMCell` instance that
trains alongside the rollout.

## What it is

`WorldModel` (`src/model/worldModel.ts`) wraps one `RSSMCell`, an Adam optimizer over its
trainable weights, and the single persistent `RSSMState` that recurrence carries across an
episode. `WorldModel.step(action, observation, rng, train)` does, every call:

1. `h_t = cell.step(prevState, [action])` — the deterministic recurrence, matching
   `docs/explainers/0003`'s `advance()` pattern.
2. `prior = cell.prior(h_t, { rng })`, `posterior = cell.posterior(h_t, observationTensor, { rng })`
   — imagination and observation-conditioned distributions over `z_t`.
3. `{ total } = klBalancedLoss(prior, posterior)` — the KL-balanced dynamics/representation loss
   from sub-increment 1 (`src/model/losses.ts`), dyn+rep terms only; no `L_pred` term (no decoder
   yet — sub-increment 2, deliberately skipped ahead of this one).
4. If `train` is true: `tf.variableGrads` over the cell's weights, one `optimizer.applyGradients`
   step. If false: the same forward pass, no gradient step.
5. Either way, internal state advances to `{ deterministic: h_t, stochastic: posterior.sample }`
   and the scalar loss (as a plain number) is returned.

`runEpisode` (`src/experiment/freeze.ts`) takes a new optional `worldModels?: (WorldModel |
undefined)[]` parameter, indexed like `policies`. Each step, after computing `frozen[]` (unchanged
from the existing freeze mechanism), agents with a world model call `worldModel.step(action,
resultObservation, rng, !frozen[i])` — **always**, regardless of frozen status; only the `train`
flag differs. `EpisodeStepRecord` gains `worldModelLoss: (number | undefined)[]`, one entry per
agent (`undefined` where no world model was given), for whatever inspects it next (priority 4's
metric plumbing).

## Why "always advance, sometimes train"

This is the one design point that isn't a straightforward extension of the existing freeze
mechanism, and it comes directly from proposal `0001`'s own metric definition: "Track the frozen
agent's world-model one-step... prediction error on newly collected transitions **over the
following steps**." A frozen agent's world model has to keep being *evaluated* against new
transitions after the freeze point — that's the entire measurement — while its *weights* must stop
moving, or the model would adapt to the partner's drift and erase the very signal being measured
(the proposal's "Minimal experiment" section says this explicitly: "if the measured agent's world
model kept training, it would adapt to the partner's drifting behavior and the signal we're trying
to isolate would be absorbed into that adaptation"). So freezing an agent's world model means:
`train=false` (no `optimizer.applyGradients`), not "stop calling `step()` at all." `WorldModel.step`
takes `train` as an explicit argument for exactly this reason, rather than reading its own frozen
status internally — the freeze *schedule* stays owned by `runEpisode`/`isFrozen`, matching
`docs/explainers/0002`'s existing separation of concerns (freeze logic lives in one place).

This also finally applies `docs/explainers/0002-freeze-mechanism.md`'s PR #12 review follow-up
("Freeze covers the world model" is structural convention, not enforced — tracked, not applied
"until the cell is actually designed"): the cell now exists, and this sub-increment is that
designed application. There is still no single shared gate function forcing "policy and world
model always freeze together" — `runEpisode` computes `frozen[i]` once per agent per step and
passes it to both `policy.update()`'s gate and `worldModel.step()`'s `train` argument, which is the
single-gate approach the follow-up asked for, just expressed as "compute once, use twice" rather
than a separate abstraction.

## Why a discriminated union for prior()/posterior()'s sample source, not two optional params

PR #37's review (@SakkarinKt, 2026-08-04) flagged that `prior(deterministic, fixedHard?, rng?)`
let `rng` be silently omitted at the type level — `rssm.prior(det)` compiled and only threw at
runtime (`sampleStraightThrough`'s own guard). This sub-increment is the first production call
site (`WorldModel.step()`), so it's the point that follow-up named for the fix. Reordering to a
required `rng` positional param (`prior(deterministic, rng, fixedHard?)`) would have worked but
forces every `fixedHard`-only test call (deterministic gradient-check tests that never need
randomness) to also thread an unused `Rng` just to satisfy the type checker. Instead:

```ts
export type LatentSampleSource = { rng: Rng } | { fixedHard: tf.Tensor3D };
prior(deterministic: tf.Tensor2D, source: LatentSampleSource): LatentDistribution
posterior(deterministic: tf.Tensor2D, observation: tf.Tensor2D, source: LatentSampleSource): LatentDistribution
```

`prior(det)` and `prior(det, {})` are both now compile errors — the exact "compiles today, fails
at runtime" gap PR #37 named is closed at the type level, not just moved. Every call site becomes
self-documenting (`{ rng }` vs `{ fixedHard: priorHard }`) instead of the positional
`prior(det, undefined, rng)` pattern scattered through the current test suite. `sampleHard` and
`straightThroughEstimator` are unchanged (still take `rng`/`hard` directly) — only the two public
`RSSMCell` methods and the internal `sampleStraightThrough` helper take the union.

Since Node runs these `.ts` files with type-stripping only (no type checking at test time — see
`package.json`'s zero-dependency test setup), a caller that bypasses TypeScript entirely (e.g. a
cast) can still construct an invalid source at runtime. `sampleStraightThrough` keeps a narrow
runtime fallback (`"fixedHard" in source ? ... : sampleHard(logits, source.rng)`) that throws a
plain `TypeError` from inside `sampleHard` if `source.rng` is `undefined` — not a new explicit
guard, just what naturally happens when `rng.next()` is called on `undefined`. A test exercises this
directly via an explicit type-erasing cast (`{} as unknown as LatentSampleSource`), documenting
that the failure mode is now "only reachable by deliberately defeating the type system," not "the
default outcome of a normal call."

## Why vectorize `sampleHard`'s Gumbel-noise transform, not the uniform draws

PR #37's third follow-up: "the per-element JS Gumbel loop costs 0.140ms/call vs 0.081ms for the
old path at `[64,8,4]`... vectorize it (one `tf.tensor` of uniforms) before batch grows." The
uniform draws themselves (`rng.next()`) can't move off the JS loop — `Rng` (`src/env/rng.ts`) is an
inherently sequential mulberry32 state machine, one scalar per call, and that sequencing is what
makes same-seed runs reproducible (the property finding #1 exists to protect). What *can* move is
the `-Math.log(-Math.log(u))` transform and the epsilon-clamp that used to run per element in the
same JS loop as the draw:

```ts
// before: per-element JS Math.min/Math.max/Math.log/Math.log inside the draw loop
gumbelNoise[i] = -Math.log(-Math.log(Math.min(Math.max(rng.next(), EPS), 1 - EPS)));

// after: the loop only draws; clamp + both logs run once as vectorized tensor ops
uniforms[i] = rng.next();
// ...
const clamped = tf.clipByValue(u, EPS, 1 - EPS);
const gumbelNoise = tf.neg(tf.log(tf.neg(tf.log(clamped))));
```

This is the literal "one `tf.tensor` of uniforms" the follow-up asked for: the JS loop now does
strictly less per element (one `rng.next()` call, no `Math`/branching), and the clamp+log+log+neg
math — the part that scales with batch — moves onto the backend's vectorized tensor ops instead of
running as scalar JS per element. `sampleHard`'s signature, its `rng`-required contract, and its
reproducibility property (quality-pass finding #1) are unchanged.

## What's deliberately not here yet

Observation reconstruction loss / `L_pred` (sub-increment 2, explicitly deferred by PR #38's
review). The drift-attributable-error metric itself — `worldModelLoss` on `EpisodeStepRecord` is
raw per-step KL-balanced loss, not the freeze-vs-control diff proposal `0001` defines; computing
that diff from these records is priority 4. No optimizer/learning-rate tuning — `WorldModel`'s
default learning rate (`1e-3`, Adam's common default) is an engineering placeholder, not a value
derived from DreamerV3 or validated against this environment; nothing in this sub-increment depends
on its exact value being right. No `experiments/0001/...` scaffold, no manifest-producing training
run — this increment only wires the mechanism; a run that actually exercises it end-to-end (with a
`manifest.json`) is priority 4/5's job. Arms B–D (any cross-agent sharing) remain entirely out of
scope — Arm A's whole point is that each agent's `WorldModel` never sees the other's tensors.

## Tensor lifecycle

`WorldModel.step()` is the first production caller of `RSSMCell.step()`/`.prior()`/`.posterior()`
(see the added contract note on `RSSMCell`'s own class doc comment). The `train=true` path
differentiates through them via `tf.variableGrads`, which needs their internal intermediates (the
recurrent-input `tf.concat`, the dense heads' activations) to stay alive until backward pass
completes — so `step()` wraps the *entire* `tf.variableGrads` call (forward and backward together)
in one outer `tf.tidy`, `tf.keep()`-ing only the two tensors that must outlive it (the next
`RSSMState`'s `deterministic`/`stochastic`). The `train=false` path shares the same forward
function and the same outer-`tf.tidy` structure, just without the surrounding `variableGrads`/
`applyGradients` — confirmed tensor-leak-free by `test/model/worldModel.test.ts` (`tf.memory()`
before/after a multi-step run, mixing `train: true`/`false`, net zero beyond the two persisted
state tensors).

## Amendment 2026-08-07: the BPTT horizon is 1, explicitly

PR #39's review (@SakkarinKt, 2026-08-06) named an implicit design point in the tensor-lifecycle
section above and asked for it to be stated explicitly here, before priority 4's metrics are built
on top of it: the `tf.keep()`-ed `nextDeterministic`/`nextStochastic` tensors this sub-increment
persists as the next `RSSMState` become plain leaves on the *following* `step()` call's own fresh
`tf.tidy`/`tf.variableGrads` tape — they carry no gradient history from the step that produced them.
So a given step's loss can only backpropagate through that one step's own `h_{t-1} -> h_t`
transition; gradients never reach further back into the recurrence. This is a backpropagation-
through-time horizon of **1**, not the multi-step chains `RSSMCell.step()` itself is fully capable of
differentiating through (the week-3 stack-validation spike's entire point — ADR-0002 decision 5,
`notes/adr-0002-js-ml-stack.md` §9–11 — validated chained gradients across 2–4 steps of
`RSSMCell.step()` called directly in a single differentiation trace; nothing here uses that
capability, because each `WorldModel.step()` call opens and closes its own `tf.variableGrads` trace).

Whether this holds through priority 4 (Arm-A metric plumbing + the 3-seed instrument-validation
runs): **yes, and it doesn't need to change to get there.** Priority 4's milestone is the
freeze-vs-both-frozen drift-attributable prediction-error measurement, which needs `WorldModel.step()`
to produce a usable per-step loss/error signal under one-transition-at-a-time online training —
exactly what horizon-1 already provides. Nothing in proposal `0001`'s Arm-A milestone requires
multi-step BPTT to be functioning for that measurement to be valid; the measurement is about
prediction error, not about validating a particular training procedure's convergence quality.

Where it will eventually matter (neither is scheduled work, recorded so a future increment doesn't
rediscover this): the human's reserved replay-buffer module (G2) samples individual transitions, so
by itself it doesn't create pressure to lift this truncation — there's no multi-step sequence in a
single sampled transition to chain gradients across regardless. A future increment that instead
trains on sampled *sub-sequences* of consecutive transitions (DreamerV3's actual training regime)
would need to explicitly restructure `WorldModel.step()`'s single-transition-per-call shape into a
sequence-batched one first — horizon-1 would otherwise silently cap how much of that sequence's
gradient signal reaches the recurrent weights, while looking like ordinary sequence training from the
call site. See docs/explainers/0006-observation-reconstruction-loss.md (2026-08-07) for the
sub-increment this amendment shipped alongside — that sub-increment adds a new loss term to the same
single-step `forward()` closure and does not change this horizon.

## Amendment 2026-08-08: `step()`'s throw path leaves `this.state` untouched

PR #40's review (@SakkarinKt, 2026-08-07) found that the tensor-lifecycle section above's `finally`
block — added by PR #39's review to stop `prevState`'s tensors leaking on a throw from `forward()`
(e.g. a shape mismatch) — traded that leak for a worse bug: it disposed `prevState.deterministic`/
`.stochastic` unconditionally, but `this.state` is only ever reassigned to the *next* state on
success (after `forward()` returns without throwing), so on a throw `this.state` was still the same
object as `prevState` — now pointing at disposed tensors. A caught throw left the model unusable
(`wm.currentState.deterministic.arraySync()` itself threw "Tensor is disposed"), which is worse than
the leak it replaced. The review also flagged a second, unmeasured case: a throw happening *after*
the `tf.keep()` calls (e.g. from `decoder.decode()` or `reconstructionLoss()`, both added by
sub-increment 2) would keep `nextDeterministic`/`nextStochastic` alive but unreachable, since nothing
would ever dispose or reference them.

Fixed by restructuring `step()` around a `try`/`catch` rather than `try`/`finally`: `prevState`'s
tensors and `observationTensor` are now only disposed, and `this.state` only reassigned, on the
success path *after* `tf.tidy()` returns without throwing. On a throw, the `catch` block disposes
only what `forward()` actually allocated for that failed attempt — `observationTensor`, plus
`nextDeterministic`/`nextStochastic` if they'd already escaped the inner `tf.tidy` via `tf.keep()`
— and re-throws; `this.state` is never touched, so it stays exactly `prevState`, still valid. Verified
directly: a wrong-length observation (shape mismatch inside `cell.posterior`, the review's own repro)
now leaves `wm.currentState` readable and unchanged after the catch, and a further `step()` call
still works normally. The `train=false` throw path is confirmed tensor-leak-free
(`tf.memory().numTensors` net zero across a caught throw); the `train=true` path leaks nothing of
`step()`'s own beyond a pre-existing, unrelated cost the review already identified and measured as
roughly equal on both the old and new code — `tf.variableGrads`' own internal `tf.tidy` drops ~66
intermediates on its own error path when `forward()` throws mid-differentiation, a library-level
behavior this fix doesn't touch and doesn't attempt to change.

## Amendment 2026-08-09: each agent's `WorldModel` now draws from its own `Rng`

PR #39's review (@SakkarinKt, 2026-08-06) follow-up 2, folded into `loop/GOAL.md` priority 4 per
PR #41's review (@SakkarinKt, 2026-08-08: "Next: priority 4... with PR #39 follow-up (2)'s
per-agent Rng folded in at the start"): `runEpisode` used to take one `Rng` instance and pass it
to every `policies[i].act(obs, rng)` call *and* every `worldModels?.[i]?.step(...)` call, all
sharing one sequential mulberry32 stream. That meant world-model sampling (the Gumbel-max draws
inside `RSSMCell.prior()`/`.posterior()`, `docs/explainers/0003`) was interleaved into the same
stream as the policies' action draws. The intervention/control arms stayed seed-aligned under that
scheme (see the "Why 'always advance, sometimes train'" section above, PR #39 review: "forward()
runs exactly once per step() and draws... uniforms unconditionally, so a frozen world model
consumes exactly the same stream as a training one") — but only because every run in the milestone
used the same `RSSMConfig` dimensions. Change `latentCategoricals`/`latentClasses` (a model-size
ablation) and the number of uniforms a `WorldModel.step()` call draws from the shared stream
changes too, which shifts every *later* policy action draw at a fixed seed — the environment
trajectory would no longer be held constant across a model-size comparison, defeating the point of
fixing a seed at all.

Fixed by giving `runEpisode` a numeric `seed` instead of a pre-built `Rng` (breaking change to its
signature — the two callers, `test/experiment/freeze.test.ts`'s `runEpisode(...)` calls, now pass
plain numbers where they used to pass `new Rng(n)`). Internally it constructs one `Rng` for the
policy-action stream (`new Rng(seed)`) and, only when `worldModels` is given, one further `Rng` per
agent (`new Rng(deriveSeed(seed, i))`, `src/env/rng.ts`) for that agent's `WorldModel.step()`
calls. `deriveSeed` XORs a large odd constant scaled by `salt + 1` into the base seed — a simple
deterministic mix, not a cryptographic hash, which is all this needs given the only caller passes
a handful of small, distinct salts (one per agent index). The RNG-alignment property above still
holds: each world model still consumes a fixed number of draws from *its own* stream every step
regardless of `train`, so the intervention and control arms stay aligned to each other — they're
just no longer aligned to (and can no longer perturb) the policy-action stream, or to each other's
world-model streams.

## Test coverage

`test/model/worldModel.test.ts`: constructing a `WorldModel` builds all layers without throwing
(`trainableWeights()` non-empty); `train: true` changes at least one weight's value across a step,
`train: false` leaves every weight bit-identical; state (`deterministic`/`stochastic`) changes on
every step regardless of `train`; repeated identical-input training steps drive the loss down (a
coarse "does it actually learn something" sanity check, not a convergence guarantee); `reset()`
returns state to the same zero-filled tensors `initialState` produces; a tensor-leak check across a
mixed train/eval multi-step run; a throw from `forward()` (wrong-length observation) leaves
`this.state` exactly as it was and the model still usable afterward; the same throw with
`train=false` leaks no tensors (Amendment 2026-08-08). `test/experiment/freeze.test.ts` gains an
end-to-end case: two
`WorldModel`s wired into `runEpisode` under an intervention `FreezeConfig` — the frozen agent's
world-model weights are bit-identical before vs. after the freeze point while the training agent's
have changed, and *both* agents' `worldModelLoss` entries stay defined (non-`undefined`, finite)
for every post-freeze step, confirming the frozen model keeps being evaluated rather than skipped.
