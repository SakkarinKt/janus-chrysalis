import { test } from "node:test";
import assert from "node:assert/strict";
import tf from "@tensorflow/tfjs-node";
import { RewardHead } from "../../src/model/rewardHead.ts";

test("RewardHead: predict() returns [batch, 1]", () => {
  const head = new RewardHead();
  const deterministic = tf.zeros([2, 4]) as tf.Tensor2D;
  const stochastic = tf.zeros([2, 6]) as tf.Tensor2D;

  const output = head.predict(deterministic, stochastic);
  assert.deepEqual(output.shape, [2, 1]);
});

test("RewardHead: trainableWeights() is non-empty after one predict() call", () => {
  const head = new RewardHead();
  const deterministic = tf.zeros([1, 4]) as tf.Tensor2D;
  const stochastic = tf.zeros([1, 6]) as tf.Tensor2D;

  head.predict(deterministic, stochastic);
  assert.ok(head.trainableWeights().length > 0);
});

test("RewardHead: same seed reproduces identical initial weights; different seeds diverge (same convention as ContinueHead/ObservationDecoder)", () => {
  const build = (seed?: number) => {
    const head = new RewardHead({ seed });
    head.predict(tf.zeros([1, 4]) as tf.Tensor2D, tf.zeros([1, 6]) as tf.Tensor2D);
    return head.trainableWeights().map((w) => Array.from(w.dataSync()));
  };
  assert.deepEqual(build(42), build(42));
  assert.notDeepEqual(build(1), build(2));
});
