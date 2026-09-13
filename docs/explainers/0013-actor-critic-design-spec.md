# Explainer/spec: actor-critic design (Dreamer-style imagination training)

Authorized by the human's PR #72 review (2026-09-11, `@SakkarinKt`): "the list gets a new item
... Scope: a **docs-only** actor-critic design spec, `docs/explainers/0013-…`, same treatment as
`0012` — interfaces, invariants, open questions; no implementation, no `src/` changes. It must
consume `computeLambdaReturns` and `ReplayBuffer` exactly as `0012` specifies them (mine to
implement — don't redesign them), and it must state the WM/AC gradient-separation invariant as a
concrete test (what is asserted, on which tensors) so priority 5's last item finally has a
target." Sequenced after PR #73's #69 follow-ups (this run). Nothing here touches `src/` — every
type below is a proposed contract, not a landed file.

**Why now**: `docs/explainers/0012-replay-buffer-lambda-returns-spec.md` specified `ReplayBuffer`
and `computeLambdaReturns` precisely because an actor-critic would need them, but had "no
consumer in the tree yet — by design." This entry is that consumer's contract: what the
actor-critic itself looks like, how it uses those two modules, and — the concrete ask — what
test would actually catch a WM/AC gradient leak, which `loop/GOAL.md` priority 5 has named since
before there was any actor-critic code to leak into (`src/agent/` still holds only `RandomPolicy`
and tabular `QLearningPolicy` as of this run — confirmed by re-reading `src/agent/policy.ts`).

## Blocking dependency this spec does not resolve: no reward head exists

DreamerV3-style imagination training (arXiv:2301.04104, Hafner/Pasukonis/Ba/Lillicrap,
"Mastering Diverse Domains through World Models" — verified via cross-checked search results
this run, `self_checked, high confidence`; not a primary-source `WebFetch` read, since
`arxiv.org` is blocked by this environment's egress proxy, same constraint
`notes/papers/codreamer-2024.md`/`mabl-2024.md` record for an earlier session) trains its critic
against λ-returns computed over an **imagined** rollout's rewards, values, and continues. This
project has a continuation head (`src/model/continueHead.ts`, `docs/explainers/0011`) and a
reconstruction decoder (`src/model/decoder.ts`, `docs/explainers/0006`), but **no reward head** —
`decoder.ts:20`'s doc comment names it explicitly: *"the decoder (and reward/continuation heads,
not built yet)"*, and `docs/explainers/0011`'s own "what's deliberately not here" section confirms
the continuation head closed only half that gap: *"Reward head. DreamerV3's full `L_pred` also
includes a reward-prediction term ... Still out of scope."*

Concretely: `computeLambdaReturns`'s `LambdaReturnsInput.rewards` (`src/agent/lambdaReturns.ts`)
has no source once a rollout leaves real environment steps and enters imagination — `env.step()`
isn't called there, only `RSSMCell.prior()` is. Without a reward head, the actor-critic below can
be fully specified (interfaces, gradient-separation invariant, training-loop shape) but cannot
be exercised end-to-end even once implemented. **This spec deliberately does not design that
head** — it's a world-model addition (extends `WorldModel`, not `src/agent/`), out of a
docs-only actor-critic increment's scope, and belongs as its own future explainer (a natural
next `RSSM completion`-adjacent increment, same single-dense-layer minimalism as `ContinueHead`,
reading `(h_t, z_t)`, predicting the shared scalar reward `StepResult.reward` — symlog-transformed
per DreamerV3, since the gridworld's reward already includes a `collisionPenalty` term that isn't
bounded to a friendly range for a bare linear head). Flagged here as the load-bearing prerequisite
so the human's Gate G2 role-flip implementation doesn't discover it only after building `Actor`/
`Critic` against this spec.

## Architecture

One `Actor` and one `Critic` per agent, both reading off the same `(deterministic, stochastic)`
RSSM state pair every other per-agent head already reads (`ObservationDecoder.decode`,
`ContinueHead.predict`) — matching Arm A's independent-per-agent-world-model topology
(`src/experiment/freeze.ts`'s `(WorldModel | undefined)[]`, one entry per agent, no sharing) and
`docs/explainers/0012`'s design question 4's same per-agent reasoning for `ReplayBuffer`. Not
pinned by any type below — an interface-compatible extension if the human's implementation
decides on a shared actor-critic instead (same escape hatch `0012` leaves open for `ReplayBuffer`).

```ts
// src/agent/actorCritic.ts — proposed, not implemented (Gate G2 role-flip, loop/GOAL.md priority 6)

export interface ActorConfig {
  actionSpaceSize: number;   // Object.keys(Action).length, src/env/types.ts
  /** Same deriveSeed-per-component convention as WorldModelConfig — see below. */
  seed?: number;
  learningRate?: number;
  /**
   * Entropy-bonus coefficient on the action distribution, DreamerV3-style
   * exploration regularization. Optional, undecided default — same
   * "engineering placeholder" status as WorldModelConfig.learningRate.
   */
  entropyCoefficient?: number;
}

export class Actor {
  constructor(config: ActorConfig);
  /** Raw action logits, [batch, actionSpaceSize] — differentiable, state in, logits out. */
  logits(deterministic: tf.Tensor2D, stochastic: tf.Tensor2D): tf.Tensor2D;
  /** Samples one Action per batch row via `rng` — real-env rollout and imagination alike. */
  act(deterministic: tf.Tensor2D, stochastic: tf.Tensor2D, rng: Rng): Action[];
  trainableWeights(): tf.Variable[];
}

export interface CriticConfig {
  seed?: number;
  learningRate?: number;
}

export class Critic {
  constructor(config: CriticConfig);
  /** Scalar value estimate per batch row, [batch, 1] — differentiable. */
  value(deterministic: tf.Tensor2D, stochastic: tf.Tensor2D): tf.Tensor2D;
  trainableWeights(): tf.Variable[];
}
```

- **`logits`/`act` split mirrors `RSSMCell.prior`/`.posterior` vs. `sampleStraightThrough`**: a
  pure differentiable forward (`logits`) separate from the sampling step, so imagination-rollout
  code can choose how to route gradients through the sample (see open question 1 below) without
  `Actor` itself hard-coding a REINFORCE-vs-reparameterized choice.
- **`seed` derivation should follow `WorldModelConfig.seed`'s pattern exactly**
  (`src/model/worldModel.ts:104-111`): one `deriveSeed(config.seed, n)` salt per component
  (`Actor`, `Critic`, and per-agent on top of that, matching `RSSMCell`/`ObservationDecoder`/
  `ContinueHead`'s three-way split), not a shared raw seed — the whole reason PR #73 just
  re-verified `assertPreFreezeParity` still holds is that this project treats uncorrelated
  initialization streams as load-bearing, not incidental.
- **No `update()` matching the `Policy` interface's per-transition hook
  (`src/agent/policy.ts`).** `QLearningPolicy.update` trains synchronously off one real
  transition; this actor-critic trains off **imagined** rollouts in batches (see below), a
  fundamentally different cadence. An `Actor`-wrapping adapter that implements `Policy` for
  `runEpisode`'s `act()` calls, while training happens in a separate imagination phase, is
  proposed but not typed here — it's a thin wrapper, not a new interface.

## Training procedure (imagination rollout)

Sketch only — pseudocode, not a function signature to implement, since several steps depend on
the not-yet-built reward head above:

```
function imaginationTrainStep(worldModel, actor, critic, startStates: RSSMState[], horizon, rng, gamma, lambda):
  states = startStates
  rewards = []; values = []; continues = []
  for t in 0..horizon:
    values.push(critic.value(states.deterministic, states.stochastic))
    actions = actor.act(states.deterministic, states.stochastic, rng)
    nextDeterministic = worldModel.cell.step(states, actions)
    priorDist = worldModel.cell.prior(nextDeterministic, { rng })
    states = { deterministic: nextDeterministic, stochastic: priorDist.sample }
    rewards.push(rewardHead.predict(states.deterministic, states.stochastic))   // does not exist — see above
    continues.push(sigmoid(worldModel.continueHead.predict(states.deterministic, states.stochastic)))
  bootstrapValue = critic.value(states.deterministic, states.stochastic)   // stop-gradient
  returns = computeLambdaReturns({ rewards, values, continues, bootstrapValue, gamma, lambda })  // src/agent/lambdaReturns.ts, unmodified
  criticLoss = meanSquaredError(values, stopGradient(returns))
  actorLoss = -mean(returns)   // sign/estimator per open question 1
  { grads } = tf.variableGrads(() => actorLoss + criticLoss, [...actor.trainableWeights(), ...critic.trainableWeights()])
  optimizer.applyGradients(grads)
```

Note what this rollout differentiates *through* without training: `worldModel.cell.step()`/
`.prior()` (dynamics) and `worldModel.continueHead.predict()` (continuation) both sit inside the
loss computation — imagination has to run the world model forward to produce the states the
actor and critic are scored on — but neither's weights appear in `tf.variableGrads`'s `varList`
above. This is the same pattern `WorldModel.step()` already uses for the opposite direction
(training the world model with an explicit `varList` that excludes anything else live in the
same process, `worldModel.ts:239`, `src/model/rssm.ts`'s class doc comment on why an implicit
varList is unsafe here) — not a new mechanism, the same one pointed the other way.

## The WM/AC gradient-separation invariant, as a concrete test

`loop/GOAL.md` priority 5 has named "WM/AC gradient separation" as an invariant to test since
before any actor-critic existed to violate it. Concretely, once `Actor`/`Critic` exist:

**Test** (`test/agent/actorCritic.test.ts`, proposed):

1. Build a `WorldModel` with a fixed seed; run a handful of real `step()` calls (`train: true`)
   so `cell`/`decoder`/`continueHead` have non-initial weights and `currentState` has real
   history — not the degenerate all-zero initial state.
2. Snapshot every `tf.Variable`'s raw values (`.dataSync()`, copied to a plain array — not just
   tensor identity, which would trivially "pass" if a variable were replaced by reference) for
   `worldModel.cell.trainableWeights()`, `worldModel.decoder.trainableWeights()`, and
   `worldModel.continueHead.trainableWeights()`. Call this `wmBefore`.
3. Run one `imaginationTrainStep` (above) for several imagined steps, starting from
   `worldModel.currentState` — a rollout whose forward pass reads `cell.step`/`.prior` and
   `continueHead.predict` (both of which have real gradients available, since their outputs feed
   `actorLoss`/`criticLoss`), then calls `optimizer.applyGradients(grads)` with `grads` computed
   over `[...actor.trainableWeights(), ...critic.trainableWeights()]` only.
4. Re-snapshot the same three sets — `wmAfter`.
5. **Assert `wmBefore` and `wmAfter` are element-wise identical** for all three components
   (`cell`, `decoder`, `continueHead`) — the world model must not move, despite the imagined
   rollout's forward pass running through it and despite gradients w.r.t. its weights being
   mathematically nonzero if it *had* been included in `varList` (the failure mode this test
   exists to catch: an implicit or over-broad `varList` accidentally pulling world-model
   variables into `tf.variableGrads`, the same class of bug `worldModel.ts:120`'s comment warns
   an implicit varList causes in the *other* direction — "pulls in every other `RSSMCell`'s
   registered variables too, in the same process").
6. **Assert the complementary, non-vacuous half**: at least one entry of
   `actor.trainableWeights()` and at least one entry of `critic.trainableWeights()` strictly
   changed value. Without this, step 5 could pass for the wrong reason — e.g. a training step
   that silently no-ops entirely (a zero learning rate, a broken loss, `applyGradients` never
   called) would also leave the world model untouched, and a gradient-separation test that can't
   fail on a real leak *or* pass vacuously on a no-op is not the "concrete test" `loop/GOAL.md`
   asked for.
7. **`decoder` is included in step 2/4's snapshot even though imagination never calls
   `decoder.decode()`** — asserting on a component the rollout doesn't even touch is a
   deliberately weaker, free check (it can only ever pass), kept because it costs nothing and
   documents that reconstruction is fully out of the imagination path, not because it's expected
   to ever catch anything `cell`/`continueHead`'s checks wouldn't already catch first.

This test cannot be written for real until `Actor`/`Critic` exist (Gate G2) and, per the
blocking-dependency section above, until a reward head exists to make `imaginationTrainStep`
actually runnable end-to-end — but every tensor set and assertion above is concrete now, so the
implementation has a target rather than a restated goal name.

## Open design questions (proposed, not decided — same status as `0012`'s four)

1. **Actor gradient estimator.** Proposed: reuse `src/model/rssm.ts`'s existing
   `straightThroughEstimator`/`sampleHard` machinery for the actor's discrete action sample too
   (categorical straight-through, same as the stochastic latent), rather than a REINFORCE/
   score-function estimator — this project already has working, gradient-checked infrastructure
   for exactly this "differentiable categorical sample" problem (`test/model/rssm.test.ts`'s
   finite-difference check), and reusing it avoids standing up a second, untested gradient
   estimator for what is mathematically the same operation. DreamerV3 itself mixes both
   (reparameterized where the action space allows it, REINFORCE-with-baseline otherwise); this
   environment's action space is small and discrete (`Action`, 5 values), well inside where the
   straight-through path is expected to work, but this is a modeling choice, not forced by any
   type above.
2. **Imagination start states come from the real rollout's own in-memory `RSSMState`s, not from
   `ReplayBuffer.sample()`.** `ReplayBufferEntry` (`0012`) stores a flat `Transition`
   (`observation`/`action`/`reward`/`nextObservation`/`done`) with no accompanying RSSM state —
   correct for its two named consumers (ablation 3's recency-weighted replay, ordinary
   off-policy experience replay), but a single sampled `Transition` cannot supply a valid
   imagination-rollout seed on its own: `RSSMState` is a hidden function of an entire trajectory
   prefix (`RSSMCell.step`'s recurrence, `src/model/rssm.ts:166`), not recoverable from one
   transition's `observation` field in isolation, and `ReplayBuffer.sample()`'s uniform/
   recency-weighted draws are explicitly unordered single entries, not trajectory-contiguous.
   Proposed resolution: seed imagination directly from `WorldModel.currentState`
   (`src/model/worldModel.ts:143`) at points during or after a real rollout, keeping
   `ReplayBuffer` scoped to what `0012` already specified it for — **not** proposing any change
   to `ReplayBufferEntry`'s shape (an embedded-latent-state extension is the alternative DreamerV3
   itself takes via its "online queue," per this run's search summary, but that would be
   redesigning `0012`'s interface, which this spec was explicitly told not to do).
3. **Reward head design** (blocking dependency above) — deliberately unresolved here.
4. **Entropy coefficient default and schedule** — `ActorConfig.entropyCoefficient` is typed above
   as optional with no proposed value; same "engineering placeholder, not tuned" status
   `QLearningConfig.epsilon`/`WorldModelConfig.learningRate` already carry.
5. **Per-agent vs. shared actor-critic** — proposed per-agent (Architecture section), consistent
   with Arm A's independent-world-model topology, but not enforced by any type: nothing here
   prevents a shared `Actor`/`Critic` pair reading a concatenated multi-agent state instead, if
   the human's implementation judges that better once it exists.

## What's deliberately not here

- **No implementation.** `Actor`/`Critic` are proposed interfaces only, same "spec only, Gate G2
  role-flip" status `0012` established for `ReplayBuffer`/`computeLambdaReturns` — nothing under
  `src/` changes in this increment.
- **No reward head**, and no attempt to work around its absence (e.g. no proposal to bootstrap
  λ-returns from ground-truth `StepResult.reward` alone during imagination — that would require
  re-entering the real environment mid-rollout, which is not imagination).
- **No wiring into `runEpisode`/`freeze.ts`.** Even once `Actor`/`Critic`/a reward head all
  exist, threading an actor-critic policy through the freeze mechanism and Arm-A's metric
  plumbing is its own future increment, out of this docs-only spec's scope.
- **The five open design questions above are not decided** — flagged so a future implementation
  and its review start from the same known-open list, matching `0012`'s own closing line.
