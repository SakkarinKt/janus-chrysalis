import { test } from "node:test";
import assert from "node:assert/strict";
import { Rng, deriveSeed } from "../../src/env/rng.ts";

test("same seed produces the same sequence", () => {
  const a = new Rng(42);
  const b = new Rng(42);
  const seqA = Array.from({ length: 20 }, () => a.next());
  const seqB = Array.from({ length: 20 }, () => b.next());
  assert.deepEqual(seqA, seqB);
});

test("different seeds diverge", () => {
  const a = new Rng(1);
  const b = new Rng(2);
  const seqA = Array.from({ length: 10 }, () => a.next());
  const seqB = Array.from({ length: 10 }, () => b.next());
  assert.notDeepEqual(seqA, seqB);
});

test("next() stays within [0, 1)", () => {
  const rng = new Rng(7);
  for (let i = 0; i < 1000; i++) {
    const v = rng.next();
    assert.ok(v >= 0 && v < 1, `value ${v} out of range`);
  }
});

test("nextInt(n) stays within [0, n)", () => {
  const rng = new Rng(123);
  for (let i = 0; i < 1000; i++) {
    const v = rng.nextInt(8);
    assert.ok(Number.isInteger(v) && v >= 0 && v < 8, `value ${v} out of range`);
  }
});

test("deriveSeed: deterministic — same seed and salt always derive the same seed", () => {
  assert.equal(deriveSeed(1, 0), deriveSeed(1, 0));
  assert.equal(deriveSeed(42, 3), deriveSeed(42, 3));
});

test("deriveSeed: different salts derive different seeds from the same base seed, producing diverging Rng streams", () => {
  const seedA = deriveSeed(7, 0);
  const seedB = deriveSeed(7, 1);
  assert.notEqual(seedA, seedB);

  const seqA = Array.from({ length: 10 }, () => new Rng(seedA).next());
  const seqB = Array.from({ length: 10 }, () => new Rng(seedB).next());
  assert.notDeepEqual(seqA, seqB);
});

test("deriveSeed: also differs from the base seed itself (agent streams decouple from the policy stream)", () => {
  const base = 99;
  assert.notEqual(deriveSeed(base, 0), base);
});
