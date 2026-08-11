# Explainer: tabular Q-learning policy

`src/agent/policy.ts` — `loop/GOAL.md` priority 4's prerequisite sub-increment, named directly by
the human in PR #42's review (@SakkarinKt, 2026-08-09): "a minimal trainable policy is in scope
for priority 4 — build it as the next sub-increment (tabular Q-learning over the discrete action
space is fine) before the 3-seed freeze-vs-both-frozen runs." Processes the "Decisions needed"
item that same run's stand-up (`reports/standup/2026-08-09.md`) raised: `RandomPolicy` (the only
existing `Policy`) has no `update()`, so proposal `0001`'s intervention condition — "prediction
error should show this error *rising* as the still-training partner's policy drifts" — has no
partner that can actually drift. This sub-increment closes that gap; it does not itself run the
3-seed validation (still priority 4's next step after this).

## What it is

**`QLearningPolicy`** (`src/agent/policy.ts`), new, implementing `Policy`: standard tabular
Q-learning with epsilon-greedy action selection. A `Map<string, number[]>` from a discretized
state key to a length-5 array of Q-values (indexed like `ACTION_VALUES`), lazily initialized to
all zeros on first visit.

**`act(observation, rng)`**: draws `rng.next()`; if it's below `epsilon`, returns a uniform-random
action via `rng.nextInt(...)` (one further draw — same mechanism `RandomPolicy.act` already uses).
Otherwise returns the action at the Q-row's highest value, ties broken by lowest index (i.e.
`Action.Stay` wins on an all-zero/tied row — see "Tie-breaking" below).

**`update(transition)`**: the standard one-step tabular TD update —
`Q(s,a) += alpha * (reward + gamma * max_a' Q(s', a') - Q(s,a))`, with the bootstrap term
(`gamma * max_a' Q(s', a')`) dropped to exactly `0` when `transition.done` is `true` — a terminal
transition has no next state to bootstrap from, the standard tabular-Q-learning convention.

**`QLearningConfig`**: `{ alpha?, gamma?, epsilon? }`, defaults `{ alpha: 0.1, gamma: 0.95,
epsilon: 0.1 }` — common small-tabular-MDP placeholder values (e.g. Sutton & Barto's worked
gridworld examples use `alpha` in the same 0.1-ish range), not tuned or validated against this
environment. Same placeholder status as `WorldModelConfig.learningRate`'s documented default
(`src/model/worldModel.ts`) — revisit once the 3-seed validation runs give something to tune
against.

## State discretization: exact key, not a lossy bin

`Observation` (`src/env/types.ts`) is a flat `number[]`, and tabular Q-learning needs a finite
discrete state space. Rather than binning at some resolution the policy would have to guess (the
policy has no access to `GridWorldConfig.gridSize` — it only sees the vector `CooperativeGridWorld`
already produced), `QLearningPolicy` uses the observation vector's own values as the key, via
`observation.join(",")`.

This is exact, not an approximation, **for this environment specifically**: every component of
`CooperativeGridWorld`'s observation (`src/env/gridworld.ts`'s `observe()`) is either a `{0, 1}`
visibility flag or an integer grid coordinate/offset divided by the (fixed, per-episode-constant)
`gridSize` — so the set of reachable observation vectors is already finite (bounded by grid
positions × visibility combinations), and the same grid configuration always produces the
bit-identical float (same division, same operands, same IEEE 754 result) every time it recurs
within a run. Two visits to the same underlying grid state therefore hash to the same key, and
distinct grid states can't collide onto the same key by construction (the vector encodes position
exactly). This is a property of `CooperativeGridWorld`'s specific observation encoding, not a
general guarantee `QLearningPolicy` makes about arbitrary `Observation` vectors — an environment
whose observations carry continuous, non-repeating noise (nothing in this codebase does today)
would fragment the table into a near-unique key per step, and this policy would stop generalizing
across visits. Flagged here so a future environment change doesn't inherit this assumption
silently.

## Tie-breaking: leftmost, not random

On a tied Q-row (every fresh state starts all-zero, so first `epsilon`-failing visit to any new
state is a tie across all 5 actions), `act()` picks the lowest index — `Action.Stay` — rather than
breaking the tie via a further `rng` draw. Chosen for simplicity: `epsilon`-greedy exploration
already guarantees every action gets sampled at every visited state with nonzero probability
independent of how ties resolve, so a leftmost tie-break doesn't block exploration, it only biases
which action wins a state's *first* greedy visit before any learning has happened there. A
random tie-break would consume an extra, variable number of `rng` draws only on tied rows,
complicating the draw-count reasoning `docs/explainers/0005`'s per-agent-`Rng` amendment relies on
for no behavioral necessity. Revisit if the leftmost bias turns out to matter empirically (e.g. if
it measurably slows early learning by under-sampling non-`Stay` actions before `epsilon` first
fires on a given state).

## Interaction with the freeze mechanism

No change needed to `src/experiment/freeze.ts`. `runEpisode` already gates `policy.update()` calls
on `frozen[i]`, generically over any `Policy` with an optional `update` hook
(`docs/explainers/0002-freeze-mechanism.md`) — `QLearningPolicy.act()` still runs (and still
explores) every step regardless of frozen status, matching proposal `0001`'s framing ("freeze one
agent's policy... " means it stops *learning*, not that it stops acting — a frozen agent still has
to produce a trajectory). This is exactly the same contract `RandomPolicy` already satisfied by
having no `update` at all; `QLearningPolicy` is the first `Policy` for which "frozen" is an
observable difference (no `update()` calls => the Q-table stops changing) rather than a vacuous
one.

## What's deliberately not here yet

- **Wiring `QLearningPolicy` into an actual experiment run.** This sub-increment adds the class and
  its tests only. The 3-seed freeze-vs-both-frozen validation runs (`loop/GOAL.md` priority 4's
  next step, per PR #42's review) still need to construct `QLearningPolicy` instances, pick a
  training-vs-frozen pairing, and run `runEpisode` at that scale — none of that happens here.
- **Function-approximation policies** (a small `tf.layers` network trained via policy gradient or
  DQN-style bootstrapping). Explicitly out of scope — the human's review named tabular Q-learning
  specifically as sufficient for this milestone's purpose (giving the training-condition partner
  *something* that drifts), not the final policy architecture.
- **Q-table serialization / persistence.** Nothing in this sub-increment saves a `QLearningPolicy`'s
  learned table between runs; each `runEpisode` call (or a training loop calling it repeatedly)
  starts from a fresh instance unless the caller keeps one around itself.

## Test coverage

`test/agent/policy.test.ts` gains: `act()` always returns a valid `Action`; `act()` is deterministic
given the same seed (mirrors `RandomPolicy`'s equivalent test); `act()` is greedy and picks the
leftmost action on an all-zero/unseen row when `epsilon: 0`; `act()` always explores when
`epsilon: 1` (never gets stuck on a high-value greedy action); `update()` scales the TD update by
`alpha` on a fresh (zero-bootstrap) next state; `update()` bootstraps `gamma * max_a' Q(s', a')`
when `done: false` and suppresses it entirely when `done: true` (a decoy-value pair of tests, each
asserting via which action ends up greedy rather than reading the table directly, since
`QLearningPolicy` exposes no getter); `QLearningPolicy` has an `update` hook (mirrors
`RandomPolicy`'s "has no update hook" test, `reports/quality/2026-08-03-quality-pass.md`'s
`Policy`-typed-reference pattern).
