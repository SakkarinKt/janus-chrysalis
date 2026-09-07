# Explainer/spec: replay buffer and λ-returns interfaces (Gate G2 handoff)

`loop/GOAL.md` priority 6, first time it's been picked: "Interface/spec/test skeletons for the
human's G2 modules (replay buffer, λ-returns) — spec only, never the implementation." Both modules
are reserved for the human under Gate G2's role-flip (`PLAN.html`: "the human has personally
implemented ≥2 modules (replay buffer, λ-returns) with Claude reviewing") — nothing in this
increment implements either. `src/agent/replayBuffer.ts` and `src/agent/lambdaReturns.ts` define
typed interfaces whose methods/functions throw a `"Gate G2 spec-only skeleton"` error, and
`test/agent/replayBuffer.test.ts` / `test/agent/lambdaReturns.test.ts` pin that throwing behavior
with one real (non-`todo`) test each, then lay out the behavioral properties an implementation
needs to satisfy as `{ todo }` tests — descriptive names now, real assertions once the human fills
in the bodies.

**Why now, not priority 5**: 2026-09-06's stand-up found priority 5's fourth named invariant
("WM/AC gradient separation") has no code to test yet — `src/agent/` holds only `RandomPolicy` and
tabular `QLearningPolicy`, no neural actor-critic sharing a gradient path with the world model —
and proposed moving to priority 6 as a sensible default, unchallenged since (no PR reply on #67).
λ-returns is exactly the actor-critic prerequisite that invariant is waiting on: standard practice
(Dreamer-family) is to compute λ-returns from imagined-rollout rewards/values/continues and train
the critic against them, at which point "does the actor-critic's gradient leak into the world
model's weights" becomes a real, testable question. This increment doesn't build that actor-critic
— it gives its two named dependencies (`loop/GOAL.md`'s own wording) a reviewed contract to be
built against, so Gate G2's role-flip has a concrete starting point instead of a blank file.

## Replay buffer (`src/agent/replayBuffer.ts`)

**Two consumers, one interface.** Proposal 0001's ablation 3 needs "MATWM-style prioritized
recency-weighted replay, layered on top of each arm" — a training-procedure mechanism, independent
of sharing topology, that reweights which past transitions the world model trains on toward more
recent experience (`notes/papers/matwm-2025.md`: "a prioritized replay buffer keeps the world
model's training distribution weighted toward recent transitions, so it doesn't overfit to an
outdated mixture of teammates' past policies"). Once an actor-critic policy exists, it will need
the same kind of buffer — a pool of past `Transition`s to sample training batches from — and there
is no reason for that to be a second, bespoke implementation. `ReplayBuffer` is specified once, with
a `sampling` mode covering both consumers: `"uniform"` for ordinary experience replay, `"recency-
weighted"` for ablation 3.

**Pinned by the interface:**

```ts
export interface ReplayBufferEntry {
  transition: Transition;   // src/agent/policy.ts
  insertedAt: number;       // monotonically increasing sequence number, not a wall-clock timestamp
}

export type ReplaySamplingStrategy = "uniform" | "recency-weighted";

export interface ReplayBufferConfig {
  capacity: number;                 // required — see "no unbounded default" below
  sampling: ReplaySamplingStrategy;
  recencyHalfLife?: number;         // required iff sampling === "recency-weighted"
}

export class ReplayBuffer {
  constructor(config: ReplayBufferConfig);
  add(transition: Transition): void;
  sample(batchSize: number, rng: Rng): ReplayBufferEntry[];
  size(): number;
}
```

- **`insertedAt` is a sequence number, not a timestamp** — matches this codebase's existing
  convention of measuring recency/age in step counts rather than real time (e.g.
  `postFreezeLossSeries`'s steps-since-freeze array-index alignment, `src/experiment/metrics.ts`).
  It also keeps sampling reproducible from a seeded `Rng` alone, with no wall-clock dependency to
  control for in a test.
- **`capacity` has no default.** Proposal 0001's whole budget is laptop-scale (`docs/proposals/0001-
  direct-nonstationarity-measurement.md`, "Estimated cost"); an unbounded buffer is never the
  intended configuration for this project, so the config forces a caller to pick a bound rather than
  silently inheriting one by omission.
- **`sample()` takes an explicit `Rng`**, not an internal random source — same reasoning
  `RandomPolicy.act`/`QLearningPolicy.act` already establish (`src/agent/policy.ts`,
  `docs/explainers/0008`): reproducibility from a seed, and no hidden draw that would shift an
  unrelated stream if this buffer's internals changed size or shape.

**Open design questions, left to the implementation** (interface-compatible either way, so
reviewable without deciding them here):

1. **Eviction policy at capacity.** Proposed default: oldest-first (ring buffer) — it composes with
   recency weighting rather than fighting it (recency weighting already deprioritizes old entries in
   sampling; oldest-first eviction just removes them once they'd contribute negligibly anyway), and
   is the simplest policy to reason about for `size()`/capacity invariants. Not pinned in the
   interface above because it's a behavior, not a type — the human's implementation should record
   whichever policy it uses in its own doc comment.
2. **Exact recency-weighting formula.** MATWM's paper doesn't publish its exact weighting function
   (`notes/papers/matwm-2025.md`'s extraction pass didn't find one); `recencyHalfLife` is proposed as
   the parameter surface (an exponential-decay half-life in insertion-count units) because it is
   the most common recency-weighting parameterization and gives `test/agent/replayBuffer.test.ts`'s
   `{ todo }` tests something concrete to check ("more recent entries sampled more often," "half-
   life scales how quickly that falls off") without committing to a specific decay base.
3. **Sampling with vs. without replacement within one batch.** Proposed: with replacement — simpler,
   and how prioritized replay is conventionally implemented when weights can change between batches
   (no need to remove-and-restore weight mass mid-draw). Flagged, not enforced by the type.
4. **Per-agent buffers vs. one buffer keyed internally by agent index.** Proposed: one `ReplayBuffer`
   instance per agent, mirroring `runEpisode`'s existing per-agent-array convention for `Policy[]`
   and `(WorldModel | undefined)[]` (`src/experiment/freeze.ts`) — Arm A's independent per-agent
   world models each need their own training stream, and a single shared buffer would need an
   `agentIndex` field on every entry plus filtering logic this interface doesn't otherwise need.
   `ReplayBufferEntry` deliberately has no `agentIndex` field under this proposal; adding one is a
   compatible extension if the human's implementation decides against it.

## λ-returns (`src/agent/lambdaReturns.ts`)

**No consumer in the tree yet — by design.** `computeLambdaReturns` is specified as a pure,
self-contained function precisely because nothing calls it today: `src/agent/policy.ts` has no
value-function-based policy, so there is nothing to wire it into yet, and forcing that wiring now
would mean building the actor-critic itself under this "spec only" increment, well past priority
6's scope. It is specified now so that when Gate G2 role-flip work begins, and separately whenever
a future actor-critic needs a return target, both find an already-reviewed contract rather than a
design conversation blocking the first line of policy code.

**Pinned by the interface:**

```ts
export interface LambdaReturnsInput {
  rewards: number[];
  values: number[];       // same length as rewards — see indexing note below
  continues: number[];    // same length as rewards — continuation probability, not a hard done flag
  bootstrapValue: number; // value estimate for the state after the last recorded step
  gamma: number;
  lambda: number;
}

export function computeLambdaReturns(input: LambdaReturnsInput): number[];
```

- **`values`/`continues` are same-length with `rewards`, not `rewards.length + 1`.** The value
  estimate for the state *after* the trajectory segment's last step is `bootstrapValue`, a separate
  field — not an implicit extra array entry. An off-by-one here (a caller passing a same-length
  array and expecting the last element to serve as the bootstrap) is exactly the kind of silent
  misindexing this codebase consistently throws on rather than guesses at (see
  `driftAttributableError`'s and `postFreezeActionDivergenceCount`'s length checks,
  `src/experiment/metrics.ts`) — the implementation should throw on a `values`/`continues`/`rewards`
  length mismatch, not truncate or pad.
- **`continues` is a continuation *probability* in `[0, 1]`, not a boolean `done` flag** — so a
  future actor-critic can feed `ContinueHead.predict()`'s sigmoid output
  (`src/model/continueHead.ts`) directly when bootstrapping through imagined rollouts, matching
  DreamerV3-style λ-return formulations. A caller working from ground-truth `done`s (e.g.
  `EpisodeStepRecord.done`, `src/experiment/freeze.ts`) passes the degenerate hard case, `done ? 0 :
  1` per step — the same target-construction convention `WorldModel.step()` already uses internally
  for its continue-head training target (`src/model/worldModel.ts:184`, `docs/explainers/0011`) —
  not a second convention this function invents.
- **Expected recurrence** (standard backward TD(λ), stated here so the implementation and its
  review have the same reference point):

  ```
  R_t = rewards[t] + gamma * continues[t] * ((1 - lambda) * V_{t+1} + lambda * R_{t+1})
  ```

  where `V_{t+1}` is `values[t + 1]` for `t < rewards.length - 1` and `bootstrapValue` at the last
  step, and `R_{t+1}` is similarly seeded from `bootstrapValue` one step past the end. Output is one
  return per input step, computed backward from the end of the segment.

**Boundary properties `test/agent/lambdaReturns.test.ts`'s `{ todo }` tests check for**, each a
direct consequence of the recurrence above rather than a separate design decision:

- `lambda: 0` collapses to the plain one-step TD target, `rewards[t] + gamma * continues[t] *
  V_{t+1}` — the `R_{t+1}` term's coefficient (`lambda`) is exactly `0`.
- `lambda: 1` collapses to the full discounted return gated by `continues` at every step (no
  `(1 - lambda) * V_{t+1}` term contributes) — the Monte-Carlo end of the interpolation.
- `continues[t] === 0` zeroes the entire bootstrap term at step `t`, severing the recursion there —
  `R_t` reduces to `rewards[t]` alone, matching a true episode end.
- Output length equals `input.rewards.length`.
- Mismatched `rewards`/`values`/`continues` lengths throw rather than silently truncating.

## What's deliberately not here

- **No implementation.** Every method/function in both files throws
  `"Gate G2 spec-only skeleton (loop/GOAL.md priority 6) — implementation is reserved for the
  human"` unconditionally. This is enforced by one real test per file (not `{ todo }`), not just
  documented — the same "a claim needs a repro, not just a comment" bar the quality-pass reports
  hold code changes to (`reports/quality/2026-08-03-quality-pass.md`).
- **No actor-critic.** `computeLambdaReturns` has no caller; wiring it into a real policy is
  out of scope here and, per `loop/GOAL.md`'s "Reserved for the human" boundary, isn't a loop
  increment to build unilaterally even once the module itself is implemented.
- **No ablation-3 wiring.** `ReplayBuffer` is not threaded into `runEpisode` or any experiment
  script — proposal 0001 explicitly scopes ablation 3 (replay-reweighting) outside the current
  Arm-A instrument-validation milestone (`docs/proposals/0001-direct-nonstationarity-measurement.md`,
  "L2 promotion request", "Explicitly out of scope").
- **The four open design questions above are not decided.** They're flagged so the human's Gate G2
  implementation and its review both start from the same list of known-open choices, not so this
  increment can pick an answer on their behalf.
