import { test } from "node:test";
import assert from "node:assert/strict";
import tf from "@tensorflow/tfjs-node";
import { categoricalKL, stopGradient, klBalancedLoss, reconstructionLoss } from "../../src/model/losses.ts";
import type { LatentDistribution } from "../../src/model/rssm.ts";

test("categoricalKL: zero for two identical distributions", () => {
  const p = tf.tensor3d([[[0.2, 0.5, 0.3], [0.1, 0.1, 0.8]]]);
  const kl = categoricalKL(p, p).arraySync() as number[];
  for (const v of kl) assert.ok(Math.abs(v) < 1e-5, `expected ~0, got ${v}`);
});

test("categoricalKL: strictly positive for two different distributions (Gibbs' inequality)", () => {
  const p = tf.tensor3d([[[0.9, 0.05, 0.05], [0.3, 0.3, 0.4]]]);
  const q = tf.tensor3d([[[0.1, 0.1, 0.8], [0.5, 0.25, 0.25]]]);
  const kl = categoricalKL(p, q).arraySync() as number[];
  for (const v of kl) assert.ok(v > 0, `expected > 0, got ${v}`);
});

test("categoricalKL: sums over both classes and categoricals into one scalar per batch row", () => {
  // Two categoricals, both p != q identically — the per-row KL must be strictly
  // greater than either categorical's own KL alone (sums, doesn't average or pick one).
  const p = tf.tensor3d([[[0.9, 0.1], [0.9, 0.1]]]);
  const q = tf.tensor3d([[[0.1, 0.9], [0.1, 0.9]]]);
  // batch size is 1, so .mean() over the batch axis is the row's own value —
  // collapses to a Scalar so .arraySync() returns a plain number, sidestepping
  // noUncheckedIndexedAccess on a would-be array index.
  const rowKL = tf.mean(categoricalKL(p, q)).arraySync() as number;

  const pSingle = tf.tensor3d([[[0.9, 0.1]]]);
  const qSingle = tf.tensor3d([[[0.1, 0.9]]]);
  const singleKL = tf.mean(categoricalKL(pSingle, qSingle)).arraySync() as number;

  assert.ok(Math.abs(rowKL - 2 * singleKL) < 1e-4, `expected sum over categoricals: ${rowKL} vs 2*${singleKL}`);
});

test("reconstructionLoss: exactly zero for identical predicted/target tensors", () => {
  const x = tf.tensor2d([[0.1, -0.2, 0.3], [0.5, 0.5, 0.5]]);
  assert.equal(reconstructionLoss(x, x).arraySync(), 0);
});

test("reconstructionLoss: sums squared error over features, then means over the batch", () => {
  // Row 0: predicted - target = [1, 0, 0] -> sum of squares 1. Row 1: [0, 2, 0] -> sum of squares 4.
  // Mean over the 2-row batch: (1 + 4) / 2 = 2.5.
  const predicted = tf.tensor2d([[1, 0, 0], [0, 2, 0]]);
  const target = tf.tensor2d([[0, 0, 0], [0, 0, 0]]);
  assert.equal(reconstructionLoss(predicted, target).arraySync(), 2.5);
});

test("reconstructionLoss: batch-means across rows (two identical rows give the same loss as one)", () => {
  const predictedPair = tf.tensor2d([[1, 2], [1, 2]]);
  const targetPair = tf.tensor2d([[0, 0], [0, 0]]);
  const pairLoss = reconstructionLoss(predictedPair, targetPair).arraySync() as number;

  const predictedSingle = tf.tensor2d([[1, 2]]);
  const targetSingle = tf.tensor2d([[0, 0]]);
  const singleLoss = reconstructionLoss(predictedSingle, targetSingle).arraySync() as number;

  assert.ok(Math.abs(pairLoss - singleLoss) < 1e-5, `expected equal per-row loss to survive batching: ${pairLoss} vs ${singleLoss}`);
});

test("stopGradient: forward value equals the input exactly", () => {
  const x = tf.tensor2d([[1, -2, 3.5]]);
  assert.deepEqual(stopGradient(x).arraySync(), x.arraySync());
});

test("stopGradient: gradient is exactly zero", () => {
  const weights = tf.tensor2d([[0.7, -0.3, 1.1]]);
  const gradFn = tf.grad((x: tf.Tensor) => tf.sum(tf.mul(stopGradient(x), weights)) as tf.Scalar);
  const grad = gradFn(tf.tensor2d([[1, -2, 3.5]])).arraySync() as number[][];
  for (const row of grad) for (const v of row) assert.equal(v, 0);
});

/** Builds a LatentDistribution from a fixed probs tensor — `sample` is unused by klBalancedLoss. */
function distFromProbs(probs: tf.Tensor3D): LatentDistribution {
  return { probs, sample: tf.zeros([probs.shape[0], probs.shape[1] * probs.shape[2]]) as tf.Tensor2D };
}

/**
 * `tf.variableGrads`'s `grads` is a `NamedTensorMap` (string-indexed), so
 * indexing it is `Tensor | undefined` under this project's
 * `noUncheckedIndexedAccess` — same shape as the pattern `checkFiniteDifference`'s
 * call sites hit in test/model/rssm.test.ts, but resolved here with an
 * explicit check instead of matching that file's tolerated `tsc` diagnostics,
 * since this is new code, not an existing one being extended.
 */
function gradFor(grads: tf.NamedTensorMap, variable: tf.Variable): tf.Tensor {
  const g = grads[variable.name];
  if (!g) throw new Error(`no gradient recorded for ${variable.name}`);
  return g;
}

test("klBalancedLoss: free-bits floor clips dynLoss/repLoss to exactly freeBits when prior and posterior nearly coincide", () => {
  const nearlyEqualPrior = tf.tensor3d([[[0.34, 0.33, 0.33]]]);
  const nearlyEqualPosterior = tf.tensor3d([[[0.33, 0.34, 0.33]]]);
  const { dynLoss, repLoss } = klBalancedLoss(distFromProbs(nearlyEqualPrior), distFromProbs(nearlyEqualPosterior), {
    freeBits: 1,
  });
  assert.equal(dynLoss.arraySync(), 1, "expected the floor, not the (much smaller) true KL");
  assert.equal(repLoss.arraySync(), 1, "expected the floor, not the (much smaller) true KL");
});

test("klBalancedLoss: total = betaDyn * dynLoss + betaRep * repLoss", () => {
  const prior = tf.tensor3d([[[0.7, 0.2, 0.1]]]);
  const posterior = tf.tensor3d([[[0.1, 0.2, 0.7]]]);
  const { dynLoss, repLoss, total } = klBalancedLoss(distFromProbs(prior), distFromProbs(posterior), {
    freeBits: 0,
    betaDyn: 2,
    betaRep: 5,
  });
  const expected = 2 * (dynLoss.arraySync() as number) + 5 * (repLoss.arraySync() as number);
  assert.ok(Math.abs((total.arraySync() as number) - expected) < 1e-5);
});

test(
  "klBalancedLoss: dynLoss's gradient flows only into the prior's logits, repLoss's only into the " +
    "posterior's — the actual point of KL balancing (getting stopGradient's placement backwards " +
    "would still produce a plausible loss value while silently breaking this). freeBits: 0 to " +
    "isolate the stop-gradient placement from the (separately tested) free-bits clip, which would " +
    "otherwise zero out both gradients whenever KL happens to sit below the floor.",
  () => {
    const priorLogitsVar = tf.variable(tf.tensor3d([[[0.2, -0.5, 1.1], [0.3, 0.1, -0.2]]]));
    const posteriorLogitsVar = tf.variable(tf.tensor3d([[[1.0, 0.0, -1.0], [-0.4, 0.2, 0.6]]]));

    function distFromLogits(logitsVar: tf.Variable): LatentDistribution {
      return distFromProbs(tf.softmax(logitsVar, -1) as tf.Tensor3D);
    }

    function dynLossOnly(): tf.Scalar {
      return klBalancedLoss(distFromLogits(priorLogitsVar), distFromLogits(posteriorLogitsVar), { freeBits: 0 })
        .dynLoss;
    }
    function repLossOnly(): tf.Scalar {
      return klBalancedLoss(distFromLogits(priorLogitsVar), distFromLogits(posteriorLogitsVar), { freeBits: 0 })
        .repLoss;
    }

    const dyn = tf.variableGrads(dynLossOnly, [priorLogitsVar, posteriorLogitsVar]);
    dyn.value.dispose();
    assert.ok(
      Array.from(gradFor(dyn.grads, priorLogitsVar).dataSync()).some((v) => v !== 0),
      "dynLoss should have a nonzero gradient w.r.t. the prior's logits",
    );
    assert.ok(
      Array.from(gradFor(dyn.grads, posteriorLogitsVar).dataSync()).every((v) => v === 0),
      "dynLoss should have an exactly zero gradient w.r.t. the posterior's logits",
    );

    const rep = tf.variableGrads(repLossOnly, [priorLogitsVar, posteriorLogitsVar]);
    rep.value.dispose();
    assert.ok(
      Array.from(gradFor(rep.grads, posteriorLogitsVar).dataSync()).some((v) => v !== 0),
      "repLoss should have a nonzero gradient w.r.t. the posterior's logits",
    );
    assert.ok(
      Array.from(gradFor(rep.grads, priorLogitsVar).dataSync()).every((v) => v === 0),
      "repLoss should have an exactly zero gradient w.r.t. the prior's logits",
    );
  },
);
