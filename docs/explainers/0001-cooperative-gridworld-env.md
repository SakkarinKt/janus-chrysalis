# Explainer: the Arm-A cooperative grid-world environment

`src/env/gridworld.ts`, first backbone-agnostic piece of proposal `0001`'s Arm-A milestone.

## What it is

A minimal, deterministic, 2-agent cooperative-navigation grid world (MPE
`simple_spread` in spirit, reimplemented natively — no PettingZoo/JAX
dependency). Two agents move on a grid with 5 discrete actions
(stay/up/down/left/right, clamped at the boundary — no wraparound) and get a
**shared** reward equal to the negative sum of each landmark's distance to
its nearest agent, minus a collision penalty if both agents occupy the same
cell. Episodes are fixed-length (`horizon`, default 75 steps).

## Why these specific choices

- **Manhattan distance, not Euclidean**, for both reward and visibility: it
  matches the 4-directional discrete action space exactly (each action
  changes Manhattan distance by exactly 1), so reward changes are easy to
  reason about and to write exact-value tests against.
- **Fixed-length episodes**, not success-based termination: the milestone's
  measurement (proposal `0001`) needs a clean, predictable point to apply the
  freeze intervention and a known number of post-freeze steps to track
  prediction error over — a variable-length episode would confound "did the
  episode end" with "did the freeze signal appear."
- **Masking, not omitting, out-of-view entities** in the observation vector:
  keeps a fixed-size numeric vector per agent regardless of what's visible,
  which is what a world model needs as an input/target shape that doesn't
  change step to step.
- **Deterministic seeded spawn** (`src/env/rng.ts`, mulberry32): the freeze
  mechanism's control condition (kill criterion #1 — both-frozen prediction
  error must stay flat) needs runs to be exactly reproducible so seed
  variance can be measured cleanly, not conflated with spawn randomness.

## What's deliberately not here yet

Per `loop/GOAL.md`'s own phasing ("environment first, freeze mechanism
next"): no freeze mechanism, no metric plumbing, no world-model backbone, no
`tfjs` dependency. This environment only produces
`(observations, reward, done)` — nothing yet consumes it. Sharing topology
(Arms B-D) is out of scope for Arm A entirely.

## Test coverage

`test/env/gridworld.test.ts`, `test/env/rng.test.ts` — determinism given a
seed, fixed observation shape, boundary clamping, partial-observability
masking, exact reward-formula match, collision penalty, and horizon-exact
episode termination. All pass under `node --test` (see "Tooling" below).

## Tooling: zero dependencies

Node 22.18+ (confirmed: this environment runs v22.22.2) strips TypeScript
syntax natively — no `typescript` package, no build step, no test-runner
dependency (`node:test` is built in). This keeps the scaffold inside
`loop/GOAL.md`'s dependency-approval boundary entirely: nothing here needs a
new `package.json` dependency. One real constraint this imposes: TS `enum`
is not erasable syntax (it compiles to runtime code) and is unsupported in
strip-only mode — `src/env/types.ts` uses a `const` object + derived union
type instead (see its comment). Full static type-checking (`tsc --noEmit`)
still needs the `typescript` package added as a devDependency at some
point — not done this run, flagged as a future increment, likely bundled
with the `tfjs` dependency PR (`loop/GOAL.md`'s pre-approved carve-out) since
that PR already touches `package.json`.
