# Explainer: the freeze mechanism

`src/experiment/freeze.ts` and `src/agent/policy.ts` — the second backbone-agnostic piece of
proposal `0001`'s Arm-A milestone, per `loop/GOAL.md`'s own phasing ("environment first, freeze
mechanism next").

## What it is

Proposal `0001`'s actual novel contribution is not the grid world but the **freeze
intervention**: partway through an episode, stop one agent's policy (and, once one exists, its
world model) from updating while the partner keeps training — then watch whether the frozen
agent's world-model prediction error rises. A **both-frozen control** (neither agent updates
after the freeze point) checks that the metric doesn't drift from noise alone.

This increment builds the scheduling half of that: `isFrozen(agentIndex, step, config)` is a
pure function answering "is this agent frozen for the transition landing on this step?" for
either condition (`"intervention"` — only `frozenAgentIndex` freezes; `"control"` — both do), and
`runEpisode(env, policies, rng, freezeConfig)` wires it into an actual rollout — it calls each
policy's `act()` every step regardless of frozen status (a frozen agent still has to act to
produce a trajectory), but only calls `update()` for agents that aren't frozen this step.

`src/agent/policy.ts` adds the minimal `Policy` interface this needs (`act()` required,
`update()` optional) and `RandomPolicy`, a uniform-random placeholder — no learned policy or
world model exists yet (gated per `notes/adr-0002-js-ml-stack.md` §7), and `RandomPolicy` has
nothing to learn, so freezing it today changes nothing observable about its behavior. What's
being validated here is the wiring, not a learning effect.

## Why these specific choices

- **Gating `update()`, not `act()`.** The proposal is explicit that freezing means both policy
  and world model "stop updating," not that the agent stops acting — an episode needs both
  agents to keep producing actions post-freeze so the *partner's* continued drift has something
  to act against. Making `update()` a separate optional hook on `Policy`, gated by the runner
  rather than by the policy itself, means the freeze logic lives in one place
  (`src/experiment/freeze.ts`) and will apply unchanged once a real learning policy/world-model
  cell implements `update()`.
- **`isFrozen` as a pure function, not state on the runner.** The proposal's two conditions
  (intervention vs. control) and the "before vs. at-or-after `freezeStep`" logic are exactly the
  kind of thing a future metrics module needs to reproduce independently (to know which post-freeze
  transitions belong to which condition when computing the drift-attributable error diff) — keeping
  it a standalone, step-indexed predicate keeps that computation decoupled from the rollout loop
  that happens to have generated the transitions.
- **`frozenAgentIndex` required only for `"intervention"`, validated at call time.** `"control"`
  freezes both agents unconditionally, so a caller-supplied index would be meaningless there;
  throwing eagerly on the invalid intervention case (missing index) surfaces a config mistake
  immediately rather than silently freezing nobody.
- **No metric computation here.** `runEpisode` returns per-step records (`observations`,
  `nextObservations`, `actions`, `reward`, `done`, `frozen`) — enough for a future metrics module
  to filter post-freeze transitions per condition and compute the prediction-error diff proposal
  `0001` specifies, but it does not compute that diff itself, since there is no world model yet to
  measure prediction error against. That is out of scope for this increment (metric plumbing is
  `loop/GOAL.md` priority 2's next sub-step after this one).

## PR #12 review follow-ups (applied / tracked)

- **`observations` was really `nextObservations` (applied this run).** The original field held
  the post-step observation under the pre-step-sounding name `observations`, forcing a future
  metrics module to pair each record with its predecessor to recover the pre-step obs — and the
  very first record's pre-step (reset) observation was never recorded at all. `EpisodeStepRecord`
  now carries both: `observations` (what each policy acted on, the reset observation for step 1)
  and `nextObservations` (the post-step observation, renamed from the old field) — matching
  `Transition`'s `observation`/`nextObservation` naming. Covered by a new chaining test asserting
  `records[i].observations === records[i - 1].nextObservations`.
- **"Freeze covers the world model" is structural convention, not an enforced invariant
  (tracked, not applied this run).** Today `runEpisode` only ever gates `Policy.update()` because
  that is the only learning hook that exists — there is no world-model update path yet for it to
  miss. The reviewer's ask is for the *cell's* design to route through this same gate (a single
  gated update path, or a runner assertion) rather than adding a second, ungated update call.
  Designing that now would mean guessing the cell's shape before the RSSM-vs-SSM/Mamba
  robustness note (`notes/papers/drama-2024.md`, still not started) has landed — deferred to when
  that cell is actually designed, per the reviewer's own "before the cell lands" framing, and
  recorded here so it isn't lost.

  **Applied 2026-08-06** (`loop/GOAL.md` priority 3 sub-increment 3,
  `docs/explainers/0005-world-model-rollout-wiring.md`): `runEpisode` now takes an optional
  `worldModels` param and computes `frozen[i]` once per agent per step, passing it to both
  `policy.update()`'s existing gate and `WorldModel.step()`'s new `train` argument — the "single
  gated update path" the reviewer asked for, expressed as "compute once, use twice" rather than a
  separate shared-gate abstraction. One refinement the design settled that this note didn't
  anticipate: "frozen" for the world model means *stop training*, not *stop evaluating* —
  `WorldModel.step()` still advances its recurrent state and returns a loss every step regardless
  of `frozen[i]`, since proposal `0001`'s whole measurement depends on the frozen agent's world
  model staying evaluable against new transitions after the freeze point. See explainer 0005 for
  why.

## What's deliberately not here yet

No actual policy learning still (`RandomPolicy.update` doesn't exist — only the world model
trains, since 2026-08-06's `worldModels` wiring, see above and explainer 0005). No
`experiments/0001/...` scaffold, no drift-attributable-error metric computation (per-step
`worldModelLoss` is raw loss, not the freeze-vs-control diff proposal `0001` defines — that's
priority 4). Sharing topologies (Arms B-D) remain entirely out of scope.

## Test coverage

`test/experiment/freeze.test.ts` — `isFrozen`'s before/at/after-freezeStep behavior for both
conditions and its validation throw; `runEpisode`'s wiring using a `CountingPolicy` test double
that counts and timestamps its own `update()` calls, confirmed against both conditions and against
no freeze config at all; an end-to-end run with `RandomPolicy` confirming the rollout still
reaches the environment's configured horizon; a chaining test (added in the PR #12 follow-up
above) confirming each record's `observations` equals the prior record's `nextObservations` (and
the first record's `observations` is the reset observation); (2026-08-06) `worldModelLoss`
defaulting to all-`undefined` without `worldModels`, and an end-to-end case with two `WorldModel`s
under an intervention config — see docs/explainers/0005-world-model-rollout-wiring.md's "Test
coverage" for that one's detail. `test/agent/policy.test.ts` — `RandomPolicy` always returns a
valid `Action`, is driven by the passed-in `Rng` (not hidden state), and exposes no `update` hook.
