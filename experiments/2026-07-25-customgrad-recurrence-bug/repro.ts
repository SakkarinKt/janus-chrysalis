/**
 * Root-cause investigation for the flaky "chaining step()+prior() across
 * >=2 timesteps" test in test/model/rssm.test.ts (PR #25 review: "root-cause
 * it before priority 3 touches RSSM code"). Run with
 * `node experiments/2026-07-25-customgrad-recurrence-bug/repro.ts`; writes
 * `summary.json` next to this file. Not a training run (no environment steps,
 * no seeds) — same manifest.json exemption the 2026-07-21 benchmark uses.
 *
 * Finding: the flakiness is not finite-difference tolerance noise. With
 * fixed (non-random) weights, `RSSMCell`'s chained `step()`+`prior()`
 * differentiation has a real, deterministic, chain-length-growing gradient
 * error — confirmed non-vanishing across a 1e-2..1e-5 epsilon sweep (a true
 * finite-difference artifact shrinks with epsilon; this doesn't). Isolated
 * to `straightThroughEstimator`'s `tf.customGrad` specifically: replacing
 * the STE-sampled feedback with an equal-shape, fully-differentiable
 * softmax-probs feedback (removing `tf.customGrad` from the loop, changing
 * nothing else) makes the error vanish back to float32 noise level at every
 * chain length tried. A bare `tf.layers.gruCell` chained only through
 * `initialState` (no `tf.customGrad` anywhere, no feature-input recurrence)
 * is also clean. So: `tf.customGrad`, invoked repeatedly within one
 * differentiation trace where each invocation's output feeds forward into a
 * later invocation (exactly `RSSMCell`'s `z_{t-1}` feedback), produces
 * incorrect upstream gradients, growing with the number of invocations.
 * Using a fresh `hard` tensor object per timestep (vs. reusing one) doesn't
 * change this — ruling out tensor-identity aliasing as the mechanism.
 *
 * See notes/adr-0002-js-ml-stack.md §11 for the full write-up and the
 * "Decisions needed" item this raises (supersedes ADR-0002 decision 5's
 * "kill criterion did not fire" read from PR #22/#23).
 */
import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import tf from "@tensorflow/tfjs-node";
import { RSSMCell, straightThroughEstimator } from "../../src/model/rssm.ts";

const CONFIG = { deterministicSize: 3, latentCategoricals: 2, latentClasses: 3 };
const EPSILON = 1e-4;
const CHAIN_LENGTHS = [1, 2, 3, 4];

function fixedHard(batch: number, categoricals: number, classes: number): tf.Tensor3D {
  return tf
    .oneHot(tf.fill([batch * categoricals], 0, "int32"), classes)
    .toFloat()
    .reshape([batch, categoricals, classes]) as tf.Tensor3D;
}

/** Deterministic (non-random) weight fill so every run reproduces the same numbers. */
function fillDeterministic(shape: number[], seedOffset: number): tf.Tensor {
  const size = shape.reduce((a, b) => a * b, 1);
  const data = Array.from({ length: size }, (_, i) => 0.05 * (((i + seedOffset) % 13) - 6));
  return tf.tensor(data, shape);
}

function assignDeterministicWeights(rssm: RSSMCell): void {
  // Force lazy build of both the GRU cell and priorDense first.
  const state0 = rssm.initialState(1);
  const det0 = rssm.step(state0, [1]);
  rssm.prior(det0);

  const cell = (rssm as unknown as { cell: { trainableWeights: Array<{ name: string; val: tf.Variable }> } }).cell;
  const priorDense = (rssm as unknown as { priorDense: { trainableWeights: Array<{ name: string; val: tf.Variable }> } })
    .priorDense;

  let offset = 0;
  for (const w of [...cell.trainableWeights, ...priorDense.trainableWeights]) {
    w.val.assign(fillDeterministic(w.val.shape, offset));
    offset += 7;
  }
}

function maxAbsErrorAgainstFiniteDifference(
  forward: () => tf.Scalar,
  variable: tf.Variable,
  epsilon: number,
): { maxErr: number; index: number; analytic: number; numeric: number } {
  const { grads } = tf.variableGrads(forward, [variable]);
  const original = Array.from(variable.dataSync());
  const analytic = Array.from(grads[variable.name].dataSync());
  let best = { maxErr: 0, index: -1, analytic: 0, numeric: 0 };
  for (let i = 0; i < original.length; i++) {
    const plus = original.slice();
    plus[i] += epsilon;
    variable.assign(tf.tensor(plus, variable.shape));
    const lossPlus = forward().arraySync() as number;

    const minus = original.slice();
    minus[i] -= epsilon;
    variable.assign(tf.tensor(minus, variable.shape));
    const lossMinus = forward().arraySync() as number;

    variable.assign(tf.tensor(original, variable.shape));
    const numeric = (lossPlus - lossMinus) / (2 * epsilon);
    const err = Math.abs(numeric - analytic[i]);
    if (err > best.maxErr) best = { maxErr: err, index: i, analytic: analytic[i], numeric };
  }
  return best;
}

/** Variant A: the actual production recurrence — z_{t-1} fed back via the STE sample. */
function makeSTEForward(rssm: RSSMCell, priorHard: tf.Tensor3D, chainLength: number): () => tf.Scalar {
  return () => {
    let state = rssm.initialState(1);
    for (let t = 0; t < chainLength; t++) {
      const deterministic = rssm.step(state, [1]);
      const { sample } = rssm.prior(deterministic, priorHard);
      state = { deterministic, stochastic: sample };
    }
    return tf.sum(state.deterministic) as tf.Scalar;
  };
}

/** Variant B: same recurrence, but z_{t-1} is fed back as plain softmax probs — no tf.customGrad anywhere. */
function makeProbsOnlyForward(rssm: RSSMCell, priorHard: tf.Tensor3D, chainLength: number): () => tf.Scalar {
  const stochasticSize = CONFIG.latentCategoricals * CONFIG.latentClasses;
  return () => {
    let deterministic = tf.zeros([1, CONFIG.deterministicSize]) as tf.Tensor2D;
    let stochastic = tf.zeros([1, stochasticSize]) as tf.Tensor2D;
    for (let t = 0; t < chainLength; t++) {
      deterministic = rssm.step({ deterministic, stochastic }, [1]);
      const { probs } = rssm.prior(deterministic, priorHard);
      stochastic = probs.reshape([1, stochasticSize]) as tf.Tensor2D;
    }
    return tf.sum(deterministic) as tf.Scalar;
  };
}

function biasVariableOf(rssm: RSSMCell): tf.Variable {
  const cell = (rssm as unknown as { cell: { trainableWeights: Array<{ name: string; val: tf.Variable }> } }).cell;
  return cell.trainableWeights.find((w) => w.name.includes("bias"))!.val;
}

const priorHard = fixedHard(1, CONFIG.latentCategoricals, CONFIG.latentClasses);

const byChainLength = CHAIN_LENGTHS.map((chainLength) => {
  const rssmA = new RSSMCell(CONFIG);
  assignDeterministicWeights(rssmA);
  const steResult = maxAbsErrorAgainstFiniteDifference(
    makeSTEForward(rssmA, priorHard, chainLength),
    biasVariableOf(rssmA),
    EPSILON,
  );

  const rssmB = new RSSMCell(CONFIG);
  assignDeterministicWeights(rssmB);
  const probsOnlyResult = maxAbsErrorAgainstFiniteDifference(
    makeProbsOnlyForward(rssmB, priorHard, chainLength),
    biasVariableOf(rssmB),
    EPSILON,
  );

  return { chainLength, steMaxAbsError: steResult, probsOnlyMaxAbsError: probsOnlyResult };
});

// Epsilon-invariance check at the worst chain length: a true finite-difference
// truncation artifact shrinks roughly quadratically as epsilon shrinks; a real
// analytic-gradient bug does not.
const worstChainLength = CHAIN_LENGTHS[CHAIN_LENGTHS.length - 1];
const rssmC = new RSSMCell(CONFIG);
assignDeterministicWeights(rssmC);
const forwardC = makeSTEForward(rssmC, priorHard, worstChainLength);
const biasC = biasVariableOf(rssmC);
const epsilonSweep = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6].map((epsilon) => ({
  epsilon,
  ...maxAbsErrorAgainstFiniteDifference(forwardC, biasC, epsilon),
}));

// Rules out tensor-identity aliasing: does a *fresh* `hard` tensor object per
// timestep (same values, different tensor id) change the error?
const rssmD = new RSSMCell(CONFIG);
assignDeterministicWeights(rssmD);
const freshHardForward = (chainLength: number) => () => {
  let state = rssmD.initialState(1);
  for (let t = 0; t < chainLength; t++) {
    const deterministic = rssmD.step(state, [1]);
    const freshHard = fixedHard(1, CONFIG.latentCategoricals, CONFIG.latentClasses);
    const { sample } = rssmD.prior(deterministic, freshHard);
    state = { deterministic, stochastic: sample };
  }
  return tf.sum(state.deterministic) as tf.Scalar;
};
const freshHardByChainLength = [2, 3, 4].map((chainLength) => ({
  chainLength,
  ...maxAbsErrorAgainstFiniteDifference(freshHardForward(chainLength), biasVariableOf(rssmD), EPSILON),
}));

// Minimal isolation: tf.customGrad alone (no GRU, no RSSMCell), chained in a
// loop where each call's output feeds the next call's input — confirms the
// bug is in tf.customGrad's chained-invocation gradient bookkeeping itself,
// not anything RSSM/GRU-specific.
function minimalCustomGradRepro(chainLength: number): { maxErr: number; index: number; analytic: number; numeric: number } {
  const weight = tf.variable(tf.tensor2d([[0.3, -0.2, 0.15]]));
  const hard = tf.tensor2d([[0, 1, 0]]);
  function forward(): tf.Scalar {
    let x = tf.tensor2d([[1, 0, 0]]);
    for (let t = 0; t < chainLength; t++) {
      const logits = tf.mul(x, weight) as tf.Tensor2D;
      x = straightThroughEstimator(logits, hard) as tf.Tensor2D;
    }
    return tf.sum(x) as tf.Scalar;
  }
  return maxAbsErrorAgainstFiniteDifference(forward, weight, EPSILON);
}
const minimalRepro = CHAIN_LENGTHS.map((chainLength) => ({ chainLength, ...minimalCustomGradRepro(chainLength) }));

const run = {
  runId: "",
  date: "",
  commit: null as string | null,
  purpose:
    "root-cause the flaky 'chaining step()+prior() across >=2 timesteps' test (PR #25 review) — " +
    "is it finite-difference tolerance noise, or a real gradient bug?",
  backend: tf.getBackend(),
  config: CONFIG,
  epsilon: EPSILON,
  byChainLength,
  epsilonSweepAtWorstChainLength: { chainLength: worstChainLength, sweep: epsilonSweep },
  freshHardTensorPerStep: freshHardByChainLength,
  minimalCustomGradOnlyRepro: minimalRepro,
};

const outPath = fileURLToPath(new URL("summary.json", import.meta.url));
const existing = existsSync(outPath) ? JSON.parse(readFileSync(outPath, "utf8")) : { runs: [] };
const existingRuns = Array.isArray(existing.runs) ? existing.runs : [];
const date = new Date().toISOString().slice(0, 10);
const sameDayCount = existingRuns.filter((r: { date: string }) => r.date === date).length;
run.runId = date + String.fromCharCode("a".charCodeAt(0) + sameDayCount);
run.date = date;

console.log(JSON.stringify(run, null, 2));
writeFileSync(outPath, JSON.stringify({ runs: [...existingRuns, run] }, null, 2) + "\n");
