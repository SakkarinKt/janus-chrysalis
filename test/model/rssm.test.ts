import { test } from "node:test";
import assert from "node:assert/strict";
import tf from "@tensorflow/tfjs-node";
import {
  RSSMCell,
  sampleHard,
  straightThroughEstimator,
  type LatentSampleSource,
  type RSSMState,
} from "../../src/model/rssm.ts";
import { Action } from "../../src/env/types.ts";
import { Rng } from "../../src/env/rng.ts";

const CONFIG = { deterministicSize: 4, latentCategoricals: 3, latentClasses: 5 };

/** Advances one full recurrent step using the prior (no observation) — convenience for tests. */
function advance(rssm: RSSMCell, state: RSSMState, actions: Action[], rng: Rng): RSSMState {
  const deterministic = rssm.step(state, actions);
  const { sample } = rssm.prior(deterministic, { rng });
  return { deterministic, stochastic: sample };
}

test("initialState: deterministic and stochastic are zero, correctly shaped", () => {
  const rssm = new RSSMCell({ deterministicSize: 6, latentCategoricals: 3, latentClasses: 5 });
  const state = rssm.initialState(2);
  assert.deepEqual(state.deterministic.shape, [2, 6]);
  assert.deepEqual(state.stochastic.shape, [2, 15]);
  assert.ok(state.deterministic.arraySync().flat().every((v) => v === 0));
  assert.ok(state.stochastic.arraySync().flat().every((v) => v === 0));
});

test("step: output is shaped [batch, deterministicSize]", () => {
  const rssm = new RSSMCell(CONFIG);
  const state = rssm.initialState(2);
  const next = rssm.step(state, [Action.Up, Action.Left]);
  assert.deepEqual(next.shape, [2, CONFIG.deterministicSize]);
});

test("step: deterministic given the same prevState and actions", () => {
  const rssm = new RSSMCell(CONFIG);
  const state = rssm.initialState(2);
  const actions = [Action.Right, Action.Down];

  const a = rssm.step(state, actions).arraySync();
  const b = rssm.step(state, actions).arraySync();
  assert.deepEqual(a, b);
});

test("step: different actions from the same prevState produce different states", () => {
  const rssm = new RSSMCell(CONFIG);
  const state = rssm.initialState(1);

  const stay = rssm.step(state, [Action.Stay]).arraySync();
  const up = rssm.step(state, [Action.Up]).arraySync();
  assert.notDeepEqual(stay, up);
});

test("step: each batch row is independent — one agent's action doesn't affect another's row", () => {
  const rssm = new RSSMCell(CONFIG);
  const batched = rssm.step(rssm.initialState(2), [Action.Up, Action.Down]).arraySync();

  const singleUp = rssm.step(rssm.initialState(1), [Action.Up]).arraySync();
  const singleDown = rssm.step(rssm.initialState(1), [Action.Down]).arraySync();

  assert.deepEqual(batched[0], singleUp[0]);
  assert.deepEqual(batched[1], singleDown[0]);
});

test("step: chaining across multiple steps (via prior) keeps a stable shape and accumulates a non-zero state", () => {
  const rssm = new RSSMCell(CONFIG);
  let state = rssm.initialState(1);
  const actionSequence = [Action.Up, Action.Up, Action.Right, Action.Stay];
  const rng = new Rng(1);

  for (const action of actionSequence) {
    state = advance(rssm, state, [action], rng);
    assert.deepEqual(state.deterministic.shape, [1, CONFIG.deterministicSize]);
    assert.deepEqual(state.stochastic.shape, [1, CONFIG.latentCategoricals * CONFIG.latentClasses]);
  }
  const final = state.deterministic.arraySync()[0];
  assert.ok(final.some((v) => v !== 0), "expected the recurrent state to move away from zero after several steps");
});

test("step: two different action sequences from the same initial state diverge", () => {
  const rssm = new RSSMCell(CONFIG);
  const initial = rssm.initialState(1);
  const rng = new Rng(1);

  let stateA = initial;
  for (const action of [Action.Up, Action.Up, Action.Up]) {
    stateA = advance(rssm, stateA, [action], rng);
  }

  let stateB = initial;
  for (const action of [Action.Down, Action.Down, Action.Down]) {
    stateB = advance(rssm, stateB, [action], rng);
  }

  assert.notDeepEqual(stateA.deterministic.arraySync(), stateB.deterministic.arraySync());
});

test("prior: probs and sample are shaped [batch, C, K] / [batch, C*K], probs sum to 1 per categorical", () => {
  const rssm = new RSSMCell(CONFIG);
  const state = rssm.initialState(2);
  const { probs, sample } = rssm.prior(state.deterministic, { rng: new Rng(1) });

  assert.deepEqual(probs.shape, [2, CONFIG.latentCategoricals, CONFIG.latentClasses]);
  assert.deepEqual(sample.shape, [2, CONFIG.latentCategoricals * CONFIG.latentClasses]);

  const sums = tf.sum(probs, -1).arraySync() as number[][];
  for (const row of sums) {
    for (const s of row) assert.ok(Math.abs(s - 1) < 1e-5, `expected softmax row to sum to 1, got ${s}`);
  }
});

test("prior: the sample is a hard one-hot per categorical (exactly one 1, rest 0)", () => {
  const rssm = new RSSMCell(CONFIG);
  const state = rssm.initialState(4);
  const { sample } = rssm.prior(state.deterministic, { rng: new Rng(1) });
  const reshaped = sample.reshape([4, CONFIG.latentCategoricals, CONFIG.latentClasses]).arraySync() as number[][][];

  for (const batchRow of reshaped) {
    for (const categorical of batchRow) {
      const ones = categorical.filter((v) => v === 1).length;
      const zeros = categorical.filter((v) => v === 0).length;
      assert.equal(ones, 1, `expected exactly one 1 per categorical, got ${categorical}`);
      assert.equal(zeros, CONFIG.latentClasses - 1);
    }
  }
});

test("prior: an Rng-seeded Gumbel-max draw reproduces the sample under the same seed, diverges under a different one (quality-pass finding #1, reports/quality/2026-08-03-quality-pass.md; renamed from its earlier 'tf.multinomial seed' name per PR #37's review — sampleHard no longer uses tf.multinomial)", () => {
  const rssm = new RSSMCell(CONFIG);
  const state = rssm.initialState(1);

  const a = rssm.prior(state.deterministic, { rng: new Rng(42) }).sample.arraySync();
  const b = rssm.prior(state.deterministic, { rng: new Rng(42) }).sample.arraySync();
  assert.deepEqual(a, b, "same seed on a fresh Rng should reproduce the same sample");

  const c = rssm.prior(state.deterministic, { rng: new Rng(43) }).sample.arraySync();
  assert.notDeepEqual(a, c, "a different seed should (with overwhelming probability) draw a different sample");
});

test("prior: a source with neither rng nor fixedHard throws rather than silently drawing an unseeded sample (normally a compile error — LatentSampleSource requires one; this exercises the runtime fallback for callers that bypass the type system, e.g. plain-JS callers, since Node runs these .ts files with type-stripping only)", () => {
  const rssm = new RSSMCell(CONFIG);
  const state = rssm.initialState(1);
  assert.throws(() => rssm.prior(state.deterministic, {} as unknown as LatentSampleSource));
});

test("posterior: conditions on the observation — different observations give different logits", () => {
  const rssm = new RSSMCell(CONFIG);
  const state = rssm.initialState(1);
  const obsA = tf.tensor2d([[1, 0, 0, 0]]);
  const obsB = tf.tensor2d([[0, 0, 0, 1]]);

  const a = rssm.posterior(state.deterministic, obsA, { rng: new Rng(1) }).probs.arraySync();
  const b = rssm.posterior(state.deterministic, obsB, { rng: new Rng(1) }).probs.arraySync();
  assert.notDeepEqual(a, b);
});

test("posterior: output shapes match the prior's", () => {
  const rssm = new RSSMCell(CONFIG);
  const state = rssm.initialState(3);
  const obs = tf.zeros([3, 7]) as tf.Tensor2D;
  const { probs, sample } = rssm.posterior(state.deterministic, obs, { rng: new Rng(1) });

  assert.deepEqual(probs.shape, [3, CONFIG.latentCategoricals, CONFIG.latentClasses]);
  assert.deepEqual(sample.shape, [3, CONFIG.latentCategoricals * CONFIG.latentClasses]);
});

test("straightThroughEstimator: forward pass equals the fixed hard sample exactly, for any hard/logits pair", () => {
  const logits = tf.tensor2d([
    [2, -1, 0.3],
    [0, 0, 0],
  ]);
  // Deliberately not the argmax of `logits` — the forward value must equal
  // `hard` regardless of whether it matches what `logits` would predict.
  const hard = tf.tensor2d([
    [0, 1, 0],
    [1, 0, 0],
  ]);
  const out = straightThroughEstimator(logits, hard);
  assert.deepEqual(out.arraySync(), hard.arraySync());
});

test("straightThroughEstimator: analytic gradient matches a finite-difference check against softmax(logits) (fixed hard, per notes/rssm-vs-ssm-implementation-robustness.md §5)", () => {
  // Deliberately checked against softmax(logits), not straightThroughEstimator's own
  // forward pass: that forward value is the fixed `hard` tensor, piecewise-constant in
  // `logits`, so a naive finite-difference on it is identically zero regardless of whether
  // the custom gradient is correct. The straight-through backward is *defined* to reproduce
  // softmax's gradient, so that's what a finite-difference check has to target.
  const hard = tf.tensor2d([[0, 1, 0]]);
  const weights = tf.tensor2d([[0.7, -0.3, 1.1]]);
  const baseLogits = [2, -1, 0.3];
  const epsilon = 1e-4;

  const softLoss = (logitsArr: number[]) => {
    const l = tf.tensor2d([logitsArr]);
    return tf.sum(tf.mul(tf.softmax(l, -1), weights)).arraySync() as number;
  };

  const gradFn = tf.grad((l: tf.Tensor) => tf.sum(tf.mul(straightThroughEstimator(l, hard), weights)) as tf.Scalar);
  const analytic = (gradFn(tf.tensor2d([baseLogits])).arraySync() as number[][])[0];

  for (let i = 0; i < baseLogits.length; i++) {
    const plus = [...baseLogits];
    plus[i] += epsilon;
    const minus = [...baseLogits];
    minus[i] -= epsilon;
    const numeric = (softLoss(plus) - softLoss(minus)) / (2 * epsilon);
    assert.ok(
      Math.abs(numeric - analytic[i]) < 1e-3,
      `dim ${i}: finite-difference gradient ${numeric} vs analytic gradient ${analytic[i]}`,
    );
  }
});

test("straightThroughEstimator: finite-differencing its own forward pass is identically zero (documents why the check above targets softmax instead)", () => {
  const hard = tf.tensor2d([[0, 1, 0]]);
  const weights = tf.tensor2d([[0.7, -0.3, 1.1]]);
  const baseLogits = [2, -1, 0.3];
  const epsilon = 1e-4;

  const steLoss = (logitsArr: number[]) => {
    const l = tf.tensor2d([logitsArr]);
    return tf.sum(tf.mul(straightThroughEstimator(l, hard), weights)).arraySync() as number;
  };

  for (let i = 0; i < baseLogits.length; i++) {
    const plus = [...baseLogits];
    plus[i] += epsilon;
    const minus = [...baseLogits];
    minus[i] -= epsilon;
    const numeric = (steLoss(plus) - steLoss(minus)) / (2 * epsilon);
    assert.equal(numeric, 0, `expected finite-differencing the discrete forward pass to be exactly 0, got ${numeric}`);
  }
});

test("straightThroughEstimator: gradient is independent of which hard sample was fixed (flows through softmax(logits) only)", () => {
  const logits = tf.tensor2d([[2, -1, 0.3]]);
  const weights = tf.tensor2d([[0.7, -0.3, 1.1]]);
  const gradWithHard = (hard: number[][]) =>
    (tf
      .grad((l: tf.Tensor) => tf.sum(tf.mul(straightThroughEstimator(l, tf.tensor2d(hard)), weights)) as tf.Scalar)(logits)
      .arraySync() as number[][])[0];

  const gradHard0 = gradWithHard([[1, 0, 0]]);
  const gradHard1 = gradWithHard([[0, 1, 0]]);
  assert.deepEqual(gradHard0, gradHard1);
});

/**
 * A fixed one-hot per [batch, categorical] slot (always class 0) — a valid
 * "hard" sample for straightThroughEstimator regardless of the actual
 * logits, per the "gradient is independent of which hard sample was fixed"
 * test above. Used to make a multi-step forward pass deterministic across
 * the repeated calls a finite-difference check requires.
 */
function fixedHard(batch: number, categoricals: number, classes: number): tf.Tensor3D {
  return tf
    .oneHot(tf.fill([batch * categoricals], 0, "int32"), classes)
    .toFloat()
    .reshape([batch, categoricals, classes]) as tf.Tensor3D;
}

/**
 * Runs `forward()` once to force lazy layer building, then diffs tf's global
 * variable registry to find exactly the tf.Variables the layers created —
 * passed explicitly to `tf.variableGrads` below as `varList`, since every
 * earlier test in this file that constructs an `RSSMCell` leaves its own
 * variables registered globally for the rest of the process (tfjs never
 * garbage-collects them), and letting `variableGrads` default to "all
 * trainable variables" pulls all of those unrelated ones in too — which
 * throws inside tfjs's own gradient bookkeeping (confirmed by running it).
 */
function ownedVariablesAfter(forward: () => tf.Scalar): tf.Variable[] {
  const before = new Set(Object.keys(tf.engine().registeredVariables));
  forward();
  const names = Object.keys(tf.engine().registeredVariables).filter((name) => !before.has(name));
  return names.map((name) => tf.engine().registeredVariables[name]);
}

/**
 * Finite-difference-checks `grad` against every component of `variable`.
 * Previously capped at 3 components regardless of `variable`'s size (PR #26
 * review: partial coverage was part of why the 2026-07-22 check passed by
 * chance) — checking all of them is cheap at this model's scale (tens to
 * low hundreds of elements per variable) and removes that gap.
 *
 * Each `forward()` call and the `variable.assign()` source tensors are
 * wrapped in `tf.tidy` — full-coverage checking otherwise leaks one tensor
 * per assign plus every intermediate `forward()` creates (PR #27 review:
 * 7730 live tensors for the chained test's 135-element sweep). `assign`
 * copies into the variable rather than adopting the source tensor (verified
 * empirically: disposing the source after assign leaves the variable's
 * value intact), so it's safe to dispose everything the closure creates.
 * `tf.tidy` itself never disposes `tf.Variable`s, even ones created inside
 * it (PR #28 review), so this wrap stays safe even if a future `forward()`
 * builds a variable lazily.
 */
function checkFiniteDifference(variable: tf.Variable, grad: tf.Tensor, forward: () => tf.Scalar): void {
  const original = Array.from(variable.dataSync());
  const analytic = Array.from(grad.dataSync());
  const epsilon = 1e-4;

  for (let i = 0; i < original.length; i++) {
    const lossPlus = tf.tidy(() => {
      const plus = original.slice();
      plus[i] += epsilon;
      variable.assign(tf.tensor(plus, variable.shape));
      return forward().arraySync() as number;
    });

    const lossMinus = tf.tidy(() => {
      const minus = original.slice();
      minus[i] -= epsilon;
      variable.assign(tf.tensor(minus, variable.shape));
      return forward().arraySync() as number;
    });

    tf.tidy(() => variable.assign(tf.tensor(original, variable.shape)));

    const numeric = (lossPlus - lossMinus) / (2 * epsilon);
    assert.ok(
      Math.abs(numeric - analytic[i]) < 5e-3,
      `${variable.name}[${i}]: finite-difference gradient ${numeric} vs analytic gradient ${analytic[i]}`,
    );
  }
}

test(
  "RSSMCell: single training step (GRU step + prior head) gradient matches finite differences " +
    "on every weight tf.variableGrads finds (week-3 stack-validation spike, extending the standalone STE check)",
  () => {
    const config = { deterministicSize: 3, latentCategoricals: 2, latentClasses: 3 };
    const rssm = new RSSMCell(config);
    const priorHard = fixedHard(1, config.latentCategoricals, config.latentClasses);
    const stochasticSize = config.latentCategoricals * config.latentClasses;
    const lossWeights = tf.tensor2d([Array.from({ length: stochasticSize }, (_, i) => 0.1 * (i + 1) - 0.3)]);

    // Loss is built from `probs` (plain softmax), not `sample` (the STE
    // output): `sample`'s forward value is the fixed hard tensor, piecewise-
    // constant in the weights and therefore identically finite-difference-
    // zero regardless of correctness — exactly the trap the standalone
    // straightThroughEstimator tests above document and route around. The
    // STE's own gradient correctness is already covered by those tests; this
    // one's new ground is the GRU step + dense head it's now chained behind.
    function forward(): tf.Scalar {
      const state = rssm.initialState(1);
      const deterministic = rssm.step(state, [Action.Up]);
      const { probs } = rssm.prior(deterministic, { fixedHard: priorHard });
      const flatProbs = probs.reshape([1, stochasticSize]) as tf.Tensor2D;
      return tf.sum(tf.mul(flatProbs, lossWeights)) as tf.Scalar;
    }

    const ownedVariables = ownedVariablesAfter(forward);
    assert.ok(ownedVariables.length > 0, "expected the forward pass to have built at least one trainable variable");

    // `value` (the forward pass's loss scalar) isn't needed after the grads
    // are pulled out — dispose it explicitly rather than letting the
    // destructure silently drop it undisposed (PR #25's leak class; see
    // reports/quality/2026-08-03-quality-pass.md).
    const { value, grads } = tf.variableGrads(forward, ownedVariables);
    value.dispose();
    for (const variable of ownedVariables) {
      checkFiniteDifference(variable, grads[variable.name], forward);
    }
  },
);

test(
  "RSSMCell: chaining step()+prior() across >=2 timesteps — the GRU cell's own gradient (kernel/" +
    "recurrent_kernel/bias) matches finite differences once the recurrent z_{t-1} feedback is " +
    "detached to a constant, sidestepping the STE forward/backward trap (2026-07-25, correcting a " +
    "same-day misdiagnosis — see notes/adr-0002-js-ml-stack.md §11 and " +
    "experiments/2026-07-25-ste-chained-finite-difference-trap/repro.ts). With `priorHard` fixed, " +
    "straightThroughEstimator's forward VALUE is exactly `priorHard` regardless of any upstream " +
    "weight (per its own doc comment: 'forward pass returns exactly hard') — so feeding its output " +
    "forward into the next timestep makes that path numerically invariant to the weights being " +
    "differentiated, while the STE's backward pass still injects a nonzero (intentional, by-design) " +
    "softmax-Jacobian gradient through it. Naive finite-differencing the STE-forward chain therefore " +
    "can't validate anything beyond what the single-step test already covers (`straightThroughEstimator`'s " +
    "own gradient, checked directly against softmax) — the discrepancy IS the STE's designed proxy " +
    "gradient, not a computation bug, and grows with chain length only because each additional " +
    "timestep adds one more STE hop whose injected gradient the forward pass can't reflect. This test " +
    "instead checks what CAN be validated: the GRU cell's own gradient through the chain, using a " +
    "recurrent feedback value that is identical numerically to what `prior()` actually produces (so " +
    "the forward pass — and hence the loss value — is unchanged) but is a literal constant on the " +
    "backward pass, so finite-differencing it is meaningful.",
  () => {
    const config = { deterministicSize: 3, latentCategoricals: 2, latentClasses: 3 };
    const rssm = new RSSMCell(config);
    const priorHard = fixedHard(1, config.latentCategoricals, config.latentClasses);
    const priorHardFlat = priorHard.reshape([1, config.latentCategoricals * config.latentClasses]) as tf.Tensor2D;

    // Force lazy build, then overwrite with fixed (non-random) weights —
    // determinism isn't load-bearing for correctness here (the detached
    // construction below is finite-difference-testable regardless of
    // weights), but it keeps this test's failure mode reproducible if it
    // ever does fail.
    const det0 = rssm.step(rssm.initialState(1), [Action.Up]);
    rssm.prior(det0, { rng: new Rng(1) });
    const cell = (rssm as unknown as { cell: { trainableWeights: Array<{ name: string; val: tf.Variable }> } }).cell;
    const priorDense = (
      rssm as unknown as { priorDense: { trainableWeights: Array<{ name: string; val: tf.Variable }> } }
    ).priorDense;
    let offset = 0;
    for (const w of [...cell.trainableWeights, ...priorDense.trainableWeights]) {
      const size = w.val.shape.reduce((a, b) => a * b, 1);
      const data = Array.from({ length: size }, (_, i) => 0.05 * (((i + offset) % 13) - 6));
      w.val.assign(tf.tensor(data, w.val.shape));
      offset += 7;
    }

    // `rssm.prior()` is still called each step (so priorDense's forward pass
    // is exercised, matching what a real rollout computes), but its `sample`
    // output is deliberately not what feeds the next state — `priorHardFlat`
    // is, a tensor with no dependency on any weight. Confirmed by construction
    // (not just by value) to equal `sample` numerically at every step, since
    // `sample`'s forward value is defined to be exactly `priorHard`.
    function forward(): tf.Scalar {
      let state = rssm.initialState(1);
      for (const action of [Action.Up, Action.Right, Action.Down] as Action[]) {
        const deterministic = rssm.step(state, [action]);
        rssm.prior(deterministic, { fixedHard: priorHard });
        state = { deterministic, stochastic: priorHardFlat };
      }
      return tf.sum(state.deterministic) as tf.Scalar;
    }

    const cellVariables = cell.trainableWeights.map((w) => w.val);
    // See the single-step test above: dispose `value` explicitly instead of
    // dropping it undisposed via destructuring.
    const { value, grads } = tf.variableGrads(forward, cellVariables);
    value.dispose();
    for (const variable of cellVariables) {
      checkFiniteDifference(variable, grads[variable.name], forward);
    }
  },
);

test(
  "regression pin: tf.layers.rnn (the wrapper RSSMCell.step() no longer uses) still crashes " +
    "tf.variableGrads when chained across >=2 timesteps — tensorflow/tfjs#1529, #3550, reproduced " +
    "directly against raw tf.layers primitives so this stays caught if any future code reaches for " +
    "the wrapper again, independent of RSSMCell's own implementation",
  () => {
    // Root-caused by direct experiment (see the 2026-07-21/07-22 stand-ups and
    // notes/adr-0002-js-ml-stack.md §8-9): whenever a `tf.layers.rnn`-wrapped
    // cell's own hidden output feeds back into the *feature* input (not just
    // `initialState`) of its own next `apply()` call within one
    // differentiation trace, tfjs-layers' RNN backward pass throws this exact
    // error — traced to `rnn()`'s internal `tfc.unstack`/`tfc.stack`
    // bookkeeping, which runs on every call regardless of sequence length.
    // tensorflow/tfjs#1529 (opened 2019, tfjs 1.0.1) and #3550 (tfjs 2.0.0)
    // report the same error for single-timestep RNN training generally;
    // still reproduces on this repo's pinned 4.22.0. Uses raw
    // `tf.layers.gruCell`/`tf.layers.rnn` rather than `RSSMCell`, since
    // `RSSMCell.step()` (2026-07-22) no longer goes through the wrapper.
    const units = 3;
    const cell = tf.layers.gruCell({ units });
    const rnn = tf.layers.rnn({ cell, returnState: true, returnSequences: false });

    function forward(): tf.Scalar {
      let h = tf.zeros([1, units]) as tf.Tensor2D;
      let feature = tf.zeros([1, units]) as tf.Tensor2D;
      for (let i = 0; i < 2; i++) {
        const input = feature.reshape([1, 1, units]);
        const output = rnn.apply(input, { initialState: [h] }) as tf.Tensor2D[];
        h = output[0];
        feature = h; // next feature depends on the previous hidden output, like RSSMCell's z_{t-1} via prior(h)
      }
      return tf.sum(h) as tf.Scalar;
    }

    const ownedVariables = ownedVariablesAfter(forward);
    assert.throws(
      () => tf.variableGrads(forward, ownedVariables),
      /Argument tensors passed to stack must be a `Tensor\[\]` or `TensorLike\[\]`/,
      "expected the known tfjs-layers RNN bug to still reproduce — if this now passes, tfjs-layers may " +
        "have fixed tensorflow/tfjs#1529/#3550 upstream, which is worth its own note but doesn't change " +
        "RSSMCell (it no longer depends on this wrapper either way)",
    );
  },
);

test("sampleHard: exactly one 1 per [batch, categorical] row, rest 0", () => {
  const logits = tf.tensor3d([
    [
      [1, 2, 3],
      [0, 0, 0],
    ],
    [
      [-1, -2, -3],
      [5, 5, 5],
    ],
  ]);
  const hard = sampleHard(logits, new Rng(1)).arraySync() as number[][][];
  for (const batchRow of hard) {
    for (const categorical of batchRow) {
      assert.equal(categorical.filter((v) => v === 1).length, 1);
      assert.equal(categorical.filter((v) => v === 0).length, 2);
    }
  }
});
