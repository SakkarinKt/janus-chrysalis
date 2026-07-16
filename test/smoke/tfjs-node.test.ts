import { test } from "node:test";
import assert from "node:assert/strict";
import tf from "@tensorflow/tfjs-node";

// Guards against the Apple-Silicon / native-build install failures flagged in
// notes/adr-0002-js-ml-stack.md §3 — surfaces a broken native binding here
// rather than inside a future world-model-cell test.
test("tfjs-node loads its native binding and runs one op", () => {
  const a = tf.tensor([1, 2, 3]);
  const b = tf.tensor([4, 5, 6]);
  const result = Array.from(a.add(b).dataSync());
  assert.deepEqual(result, [5, 7, 9]);
  assert.equal(tf.getBackend(), "tensorflow");
});
