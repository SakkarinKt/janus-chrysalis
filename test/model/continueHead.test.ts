import { test } from "node:test";
import assert from "node:assert/strict";
import tf from "@tensorflow/tfjs-node";
import { ContinueHead } from "../../src/model/continueHead.ts";

test("ContinueHead: predict() returns [batch, 1]", () => {
  const head = new ContinueHead();
  const deterministic = tf.zeros([2, 4]) as tf.Tensor2D;
  const stochastic = tf.zeros([2, 6]) as tf.Tensor2D;

  const output = head.predict(deterministic, stochastic);
  assert.deepEqual(output.shape, [2, 1]);
});

test("ContinueHead: trainableWeights() is non-empty after one predict() call", () => {
  const head = new ContinueHead();
  const deterministic = tf.zeros([1, 4]) as tf.Tensor2D;
  const stochastic = tf.zeros([1, 6]) as tf.Tensor2D;

  head.predict(deterministic, stochastic);
  assert.ok(head.trainableWeights().length > 0);
});

test("ContinueHead: same seed reproduces identical initial weights; different seeds diverge (same convention as ObservationDecoder, processing PR #45's review)", () => {
  const build = (seed?: number) => {
    const head = new ContinueHead({ seed });
    head.predict(tf.zeros([1, 4]) as tf.Tensor2D, tf.zeros([1, 6]) as tf.Tensor2D);
    return head.trainableWeights().map((w) => Array.from(w.dataSync()));
  };
  assert.deepEqual(build(42), build(42));
  assert.notDeepEqual(build(1), build(2));
});
