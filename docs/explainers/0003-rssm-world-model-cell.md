# Explainer: the RSSM world-model cell — deterministic recurrence

`src/model/rssm.ts` — the first piece of proposal `0001`'s Arm-A milestone's world-model cell, per
PR #14's review (@SakkarinKt, 2026-07-14: "go ahead and start the RSSM world-model cell... split
the cell as you proposed: struct/forward-pass, then STE + gradient-check"). This increment covers
*only* the struct/forward-pass half — the GRU-based deterministic recurrent state and its update
rule. The stochastic categorical latent (prior/posterior distributions over the discrete state),
its straight-through gradient estimator (`tf.stopGradient`), and the finite-difference
gradient-check test `notes/rssm-vs-ssm-implementation-robustness.md` §5 recommends shipping with
the cell are the explicitly deferred next sub-increment.

## What it is

`RSSMCell` wraps `tf.layers.gruCell` inside a `tf.layers.rnn` layer (`returnState: true,
returnSequences: false`) and exposes two methods: `initialState(batchSize)`, returning an
all-zero `[batch, deterministicSize]` hidden state, and `step(prevState, actions)`, one
recurrent update `h_t = GRU(h_{t-1}, onehot(action_{t-1}))` for a batch of discrete actions
(one per agent). Calling `.apply()` on the `rnn` layer with concrete `Tensor`s (not
`SymbolicTensor` placeholders) runs eagerly — no `model.compile()`/graph-building step is needed
for this backbone-agnostic recurrence to execute today.

This is deliberately narrower than a full RSSM step. DreamerV2/V3-style RSSMs (per
`notes/rssm-vs-ssm-implementation-robustness.md` §3) feed the *previous stochastic latent sample*
into the deterministic GRU update alongside the action — `h_t = GRU(h_{t-1}, [z_{t-1}, a_{t-1}])`
— but no stochastic latent exists yet in this increment, so `step()`'s recurrent input is
action-only for now. That means `step()`'s input shape will change (grow) once the stochastic
latent lands next — noted here so that's an expected, not surprising, diff in the following PR.

## Why these specific choices

- **`tf.layers.gruCell` wrapped in `tf.layers.rnn`, not hand-rolled gate math.**
  `notes/rssm-vs-ssm-implementation-robustness.md` §3's whole argument for choosing RSSM over
  SSM/Mamba is that TF.js already ships and tests `tf.layers.gru` as a mature primitive — composing
  it, rather than reimplementing sigmoid/tanh gate equations from `tf.matMul`/`tf.sigmoid` calls by
  hand, is the concrete way to cash in that argument. Calling `cell.apply([input, state])` directly
  on the bare `GRUCell` layer does **not** work — its `build()` expects exactly one input shape and
  throws `ValueError: Expected exactly 1 Shape; got 2` on a two-tensor array input (confirmed by
  running it in this session before settling on the final API). `tf.layers.rnn({cell, returnState:
  true})` is the documented way to drive a bare cell with an explicit `initialState`, and was
  confirmed working end-to-end this session: correct output shapes, deterministic given fixed
  weights and input, and output sensitive to changes in the input.
- **One-hot action encoding, no observation input yet.** The deterministic path's only input this
  increment is the previous action, one-hot encoded to `NUM_ACTIONS` (5, from
  `src/env/types.ts`'s `Action`). No observation encoder exists yet — that's part of the posterior
  computation (`q(z_t | h_t, o_t)`), which needs the stochastic latent machinery this increment
  explicitly excludes.
- **`RSSMState` holds only `deterministic`, not a placeholder `stochastic` field.** Adding a
  not-yet-meaningful field now (e.g. a zero-filled or `null` stochastic tensor) would either lie
  about what the struct represents or need immediate follow-up churn once the real field lands next
  increment — one field, added when it's real, keeps `RSSMState`'s shape honest about what this
  increment actually built.
- **Config is `{ deterministicSize }` only, no `stochasticSize`/`numCategoricals` yet.** Same
  reasoning: a config field with no consumer yet is dead weight until the stochastic path exists to
  read it.
- **No batch-size or shape validation added.** `actions.length` must match `prevState`'s batch
  dimension; passing mismatched arrays produces a `tf.oneHot`/`reshape` shape error direct from
  TF.js rather than a custom check duplicating what the framework already enforces.

## What's deliberately not here yet

The stochastic categorical latent (prior `p(z_t | h_t)`, posterior `q(z_t | h_t, o_t)`,
straight-through sampling via `tf.stopGradient`), any observation encoder/decoder, the
finite-difference gradient-check test, loss functions, training code, and any wiring into
`src/experiment/freeze.ts`'s rollout loop. Tensor lifecycle management (`tf.tidy`/explicit
`.dispose()`) is also not addressed this increment — no training loop exists yet to make the
resulting tensor-count growth a real cost, and premature disposal logic risks disposing a state
tensor a caller still needs across `step()` calls. Worth revisiting once the cell is wired into an
actual multi-episode rollout.

## Test coverage

`test/model/rssm.test.ts` — `initialState`'s zero-fill and shape; `step`'s output shape;
determinism (same `prevState`/`actions` twice → identical output, given the cell's weights don't
change between calls); sensitivity (different actions from the same `prevState` → different
output); batch-row independence (one agent's action in a batch of 2 doesn't leak into another
agent's row — checked by comparing a batched call against two single-agent calls with matching
per-row actions); multi-step chaining keeps a stable shape and the state moves away from all-zero
after several steps; two different action sequences from the same initial state diverge. 35/35
tests passing overall (`npm test`; 28 prior + 7 new).
