import { test } from "node:test";
import assert from "node:assert/strict";
import tf from "@tensorflow/tfjs-node";
import { ObservationDecoder } from "../../src/model/decoder.ts";

test("ObservationDecoder: decode() returns [batch, observationSize]", () => {
  const decoder = new ObservationDecoder({ observationSize: 5 });
  const deterministic = tf.zeros([2, 4]) as tf.Tensor2D;
  const stochastic = tf.zeros([2, 6]) as tf.Tensor2D;

  const output = decoder.decode(deterministic, stochastic);
  assert.deepEqual(output.shape, [2, 5]);
});

test("ObservationDecoder: trainableWeights() is non-empty after one decode() call", () => {
  const decoder = new ObservationDecoder({ observationSize: 5 });
  const deterministic = tf.zeros([1, 4]) as tf.Tensor2D;
  const stochastic = tf.zeros([1, 6]) as tf.Tensor2D;

  decoder.decode(deterministic, stochastic);
  assert.ok(decoder.trainableWeights().length > 0);
});
