/**
 * Root-cause investigation for the flaky "chaining step()+prior() across
 * >=2 timesteps" test in test/model/rssm.test.ts (PR #25 review: "root-cause
 * it before priority 3 touches RSSM code"). Run with
 * `node experiments/2026-07-25-ste-chained-finite-difference-trap/repro.ts`;
 * writes `summary.json` next to this file. Not a training run (no
 * environment steps, no seeds) — same manifest.json exemption the
 * 2026-07-21 benchmark uses.
 *
 * **Corrected same day, in review** (@SakkarinKt on PR #26): this directory
 * was originally named `2026-07-25-customgrad-recurrence-bug` and concluded
 * the flakiness was a real tfjs `tf.customGrad` gradient-correctness bug.
 * That conclusion was wrong. The actual mechanism, confirmed below:
 *
 * `straightThroughEstimator`'s forward pass is defined to return exactly
 * `hard` (see its doc comment in src/model/rssm.ts) — literally,
 * unconditionally, regardless of `logits`. When `RSSMCell.prior()` is
 * called with a *fixed* `hard` (as every gradient-check test in this file
 * does, to make repeated forward calls deterministic), the resulting
 * `sample` fed into the next timestep's recurrent input is therefore
 * numerically **invariant to every upstream weight** — confirmed below by
 * bumping the GRU cell's bias by 0.5 and observing `sample` doesn't change
 * by a single bit. Finite-differencing that path correctly reports zero
 * sensitivity. But the STE's whole *point* is to inject a nonzero,
 * intentionally-biased backward gradient (the softmax Jacobian-vector
 * product) through what would otherwise be a non-differentiable hard
 * sample — that's what "straight-through" means. So `tf.variableGrads`
 * correctly reports a nonzero analytic gradient for any weight upstream of
 * a chained STE sample, and naive central-difference checking correctly
 * reports zero sensitivity through that same path. The "mismatch" is not
 * noise and does not indicate a bug — it IS the STE's designed proxy
 * gradient, and no amount of epsilon-tuning can make finite-differencing
 * see it, because the two are deliberately measuring different things.
 * `src/model/rssm.ts` already documents this trap for the single-step case
 * ("naive finite-difference checking *cannot* validate this function's
 * gradient against its own forward pass"); this generalizes it to the
 * chain>=2 case, where it's the GRU *cell's* weights (not just
 * `priorDense`'s) that route through a chained STE hop once z_{t-1} feeds
 * back in.
 *
 * The growth with chain length that looked like a growing bug is exactly
 * what's expected: each additional timestep adds one more STE hop, each
 * injecting its own proxy-gradient contribution that the forward pass
 * still can't reflect.
 */
import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import tf from "@tensorflow/tfjs-node";
import { RSSMCell, straightThroughEstimator } from "../../src/model/rssm.ts";
import { Rng } from "../../src/env/rng.ts";

const CONFIG = { deterministicSize: 3, latentCategoricals: 2, latentClasses: 3 };
const EPSILON = 1e-4;
const CHAIN_LENGTHS = [1, 2, 3, 4];

function fixedHard(batch: number, categoricals: number, classes: number): tf.Tensor3D {
  return tf
    .oneHot(tf.fill([batch * categoricals], 0, "int32"), classes)
    .toFloat()
    .reshape([batch, categoricals, classes]) as tf.Tensor3D;
}

function fillDeterministic(shape: number[], seedOffset: number): tf.Tensor {
  const size = shape.reduce((a, b) => a * b, 1);
  const data = Array.from({ length: size }, (_, i) => 0.05 * (((i + seedOffset) % 13) - 6));
  return tf.tensor(data, shape);
}

function assignDeterministicWeights(rssm: RSSMCell): void {
  const state0 = rssm.initialState(1);
  const det0 = rssm.step(state0, [1]);
  // Only forces priorDense's lazy build (weights are overwritten below); the
  // sample this draws is discarded, so the rng's seed is arbitrary — but
  // prior() requires one when fixedHard is omitted (quality-pass finding #1).
  rssm.prior(det0, undefined, new Rng(1));
  const cell = (rssm as unknown as { cell: { trainableWeights: Array<{ val: tf.Variable }> } }).cell;
  const priorDense = (rssm as unknown as { priorDense: { trainableWeights: Array<{ val: tf.Variable }> } }).priorDense;
  let offset = 0;
  for (const w of [...cell.trainableWeights, ...priorDense.trainableWeights]) {
    w.val.assign(fillDeterministic(w.val.shape, offset));
    offset += 7;
  }
}

function biasVariableOf(rssm: RSSMCell): tf.Variable {
  const cell = (rssm as unknown as { cell: { trainableWeights: Array<{ name: string; val: tf.Variable }> } }).cell;
  return cell.trainableWeights.find((w) => w.name.includes("bias"))!.val;
}

function maxAbsErrorAgainstFiniteDifference(
  forward: () => tf.Scalar,
  variable: tf.Variable,
  epsilon: number,
): { maxErr: number; analytic: number[] } {
  const { grads } = tf.variableGrads(forward, [variable]);
  const original = Array.from(variable.dataSync());
  const analytic = Array.from(grads[variable.name].dataSync());
  let maxErr = 0;
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
    maxErr = Math.max(maxErr, Math.abs(numeric - analytic[i]));
  }
  return { maxErr, analytic };
}

const priorHard = fixedHard(1, CONFIG.latentCategoricals, CONFIG.latentClasses);
const priorHardFlat = priorHard.reshape([1, CONFIG.latentCategoricals * CONFIG.latentClasses]) as tf.Tensor2D;

/** The actual production recurrence: z_{t-1} fed back via the STE sample. */
function makeSTEForward(rssm: RSSMCell, chainLength: number): () => tf.Scalar {
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

/**
 * Same recurrence, same loss VALUE at every step (confirmed below) — but
 * z_{t-1} is a literal constant with no weight dependency, instead of the
 * STE sample. `prior()` is still called each step so priorDense's forward
 * pass is exercised identically; only what feeds the *next* state differs.
 */
function makeDetachedForward(rssm: RSSMCell, chainLength: number): () => tf.Scalar {
  return () => {
    let state = rssm.initialState(1);
    for (let t = 0; t < chainLength; t++) {
      const deterministic = rssm.step(state, [1]);
      rssm.prior(deterministic, priorHard);
      state = { deterministic, stochastic: priorHardFlat };
    }
    return tf.sum(state.deterministic) as tf.Scalar;
  };
}

// 1. Forward-pass invariance: bumping the bias doesn't change `sample` at all.
const invarianceCheck = (() => {
  const rssm = new RSSMCell(CONFIG);
  assignDeterministicWeights(rssm);
  const bias = biasVariableOf(rssm);
  const original = Array.from(bias.dataSync());

  const det = rssm.step(rssm.initialState(1), [1]);
  const before = rssm.prior(det, priorHard).sample.arraySync();

  const bumped = original.slice();
  bumped[0] += 0.5;
  bias.assign(tf.tensor(bumped, bias.shape));
  const detAfter = rssm.step(rssm.initialState(1), [1]);
  const after = rssm.prior(detAfter, priorHard).sample.arraySync();
  bias.assign(tf.tensor(original, bias.shape));

  return { before, after, identical: JSON.stringify(before) === JSON.stringify(after) };
})();

// 2. STE-forward vs detached-forward: identical loss values, but only the
//    detached forward's gradient matches finite differences. The STE
//    forward's "error" against finite differences is (near-)exactly the
//    difference between the two analytic gradients — i.e. the STE's
//    injected proxy gradient, not a computation bug.
const byChainLength = CHAIN_LENGTHS.map((chainLength) => {
  const rssmSTE = new RSSMCell(CONFIG);
  assignDeterministicWeights(rssmSTE);
  const steForward = makeSTEForward(rssmSTE, chainLength);
  const steBias = biasVariableOf(rssmSTE);
  const steResult = maxAbsErrorAgainstFiniteDifference(steForward, steBias, EPSILON);

  const rssmDetached = new RSSMCell(CONFIG);
  assignDeterministicWeights(rssmDetached);
  const detachedForward = makeDetachedForward(rssmDetached, chainLength);
  const detachedBias = biasVariableOf(rssmDetached);
  const detachedResult = maxAbsErrorAgainstFiniteDifference(detachedForward, detachedBias, EPSILON);

  const lossSTE = steForward().arraySync() as number;
  const lossDetached = detachedForward().arraySync() as number;

  let maxAnalyticDiff = 0;
  for (let i = 0; i < steResult.analytic.length; i++) {
    maxAnalyticDiff = Math.max(maxAnalyticDiff, Math.abs(steResult.analytic[i] - detachedResult.analytic[i]));
  }

  return {
    chainLength,
    lossValuesIdentical: lossSTE === lossDetached,
    steMaxAbsErrorVsFiniteDifference: steResult.maxErr,
    detachedMaxAbsErrorVsFiniteDifference: detachedResult.maxErr,
    maxAbsDiffBetweenSTEAndDetachedAnalyticGradient: maxAnalyticDiff,
  };
});

// 3. Minimal isolation, corrected: a SINGLE (non-chained) tf.customGrad call
//    already shows this "mismatch" whenever the downstream loss gradient
//    isn't degenerate — it has nothing to do with chaining or the GRU cell.
//    `loss = sum(x)` makes `dy` uniform across the class dimension, which
//    makes the STE backward formula's `(dy - <dy, softmax>)` term vanish
//    identically — a special case, not evidence chaining was required.
function minimalCustomGradCase(lossWeights: number[]): { maxErr: number } {
  const weight = tf.variable(tf.tensor2d([[0.3, -0.2, 0.15]]));
  const hard = tf.tensor2d([[0, 1, 0]]);
  const w = tf.tensor2d([lossWeights]);
  function forward(): tf.Scalar {
    const logits = tf.mul(tf.tensor2d([[1, 0, 0]]), weight) as tf.Tensor2D;
    const x = straightThroughEstimator(logits, hard) as tf.Tensor2D;
    return tf.sum(tf.mul(x, w)) as tf.Scalar;
  }
  return { maxErr: maxAbsErrorAgainstFiniteDifference(forward, weight, EPSILON).maxErr };
}
const minimalCustomGradIsolation = {
  degenerateUniformDy: minimalCustomGradCase([1, 1, 1]), // equivalent to loss = sum(x)
  nonDegenerateReweighted: minimalCustomGradCase([1, 2, 3]),
};

const run = {
  runId: "",
  date: "",
  commit: null as string | null,
  purpose:
    "corrected root-cause of the flaky 'chaining step()+prior() across >=2 timesteps' test (PR #25 " +
    "review) — confirms it's the straight-through estimator's known forward/backward mismatch, not a " +
    "tfjs/tf.customGrad bug (retracts this directory's original 2026-07-25 conclusion, per @SakkarinKt's " +
    "PR #26 review)",
  backend: tf.getBackend(),
  config: CONFIG,
  epsilon: EPSILON,
  forwardPassInvarianceToWeights: invarianceCheck,
  byChainLength,
  minimalCustomGradIsolation,
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
