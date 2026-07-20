import { test } from "node:test";
import assert from "node:assert/strict";
import tf from "@tensorflow/tfjs-node";
import {
  RSSMCell,
  sampleHard,
  straightThroughEstimator,
  type RSSMState,
} from "../../src/model/rssm.ts";
import { Action } from "../../src/env/types.ts";

const CONFIG = { deterministicSize: 4, latentCategoricals: 3, latentClasses: 5 };

/** Advances one full recurrent step using the prior (no observation) — convenience for tests. */
function advance(rssm: RSSMCell, state: RSSMState, actions: Action[]): RSSMState {
  const deterministic = rssm.step(state, actions);
  const { sample } = rssm.prior(deterministic);
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

  for (const action of actionSequence) {
    state = advance(rssm, state, [action]);
    assert.deepEqual(state.deterministic.shape, [1, CONFIG.deterministicSize]);
    assert.deepEqual(state.stochastic.shape, [1, CONFIG.latentCategoricals * CONFIG.latentClasses]);
  }
  const final = state.deterministic.arraySync()[0];
  assert.ok(final.some((v) => v !== 0), "expected the recurrent state to move away from zero after several steps");
});

test("step: two different action sequences from the same initial state diverge", () => {
  const rssm = new RSSMCell(CONFIG);
  const initial = rssm.initialState(1);

  let stateA = initial;
  for (const action of [Action.Up, Action.Up, Action.Up]) {
    stateA = advance(rssm, stateA, [action]);
  }

  let stateB = initial;
  for (const action of [Action.Down, Action.Down, Action.Down]) {
    stateB = advance(rssm, stateB, [action]);
  }

  assert.notDeepEqual(stateA.deterministic.arraySync(), stateB.deterministic.arraySync());
});

test("prior: probs and sample are shaped [batch, C, K] / [batch, C*K], probs sum to 1 per categorical", () => {
  const rssm = new RSSMCell(CONFIG);
  const state = rssm.initialState(2);
  const { probs, sample } = rssm.prior(state.deterministic);

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
  const { sample } = rssm.prior(state.deterministic);
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

test("posterior: conditions on the observation — different observations give different logits", () => {
  const rssm = new RSSMCell(CONFIG);
  const state = rssm.initialState(1);
  const obsA = tf.tensor2d([[1, 0, 0, 0]]);
  const obsB = tf.tensor2d([[0, 0, 0, 1]]);

  const a = rssm.posterior(state.deterministic, obsA).probs.arraySync();
  const b = rssm.posterior(state.deterministic, obsB).probs.arraySync();
  assert.notDeepEqual(a, b);
});

test("posterior: output shapes match the prior's", () => {
  const rssm = new RSSMCell(CONFIG);
  const state = rssm.initialState(3);
  const obs = tf.zeros([3, 7]) as tf.Tensor2D;
  const { probs, sample } = rssm.posterior(state.deterministic, obs);

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
  const hard = sampleHard(logits).arraySync() as number[][][];
  for (const batchRow of hard) {
    for (const categorical of batchRow) {
      assert.equal(categorical.filter((v) => v === 1).length, 1);
      assert.equal(categorical.filter((v) => v === 0).length, 2);
    }
  }
});
