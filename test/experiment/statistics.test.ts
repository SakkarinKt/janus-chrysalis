import { test } from "node:test";
import assert from "node:assert/strict";
import { binomialTwoSided, binomialUpperTail } from "../../src/experiment/statistics.ts";

test("binomialUpperTail: P(X>=0) is 1 regardless of n", () => {
  assert.equal(binomialUpperTail(0, 5), 1);
  assert.equal(binomialUpperTail(0, 10), 1);
});

test("binomialUpperTail: P(X>=n) is exactly 1/2^n", () => {
  assert.equal(binomialUpperTail(5, 5), 1 / 32);
  assert.equal(binomialUpperTail(10, 10), 1 / 1024);
});

test("binomialUpperTail: matches a hand-computed value (k=5, n=10)", () => {
  // C(10,5..10) = 252+210+120+45+10+1 = 638; 638/1024 = 0.623046875
  assert.equal(binomialUpperTail(5, 10), 638 / 1024);
});

test("binomialUpperTail: matches a hand-computed value (k=8, n=10)", () => {
  // C(10,8..10) = 45+10+1 = 56; 56/1024 = 0.0546875
  assert.equal(binomialUpperTail(8, 10), 56 / 1024);
});

test("binomialUpperTail: symmetric under k -> n-k+1 complement (P(X>=k) + P(X<=k-1) = 1)", () => {
  const n = 10;
  for (let k = 1; k <= n; k++) {
    assert.ok(
      Math.abs(binomialUpperTail(k, n) + binomialUpperTail(n - k + 1, n) - 1) < 1e-9,
      `k=${k}`,
    );
  }
});

test("binomialTwoSided: k at the exact center (n even) is 1", () => {
  assert.equal(binomialTwoSided(5, 10), 1);
});

test("binomialTwoSided: matches a hand-computed value (k=8, n=10)", () => {
  // upper = P(X>=8) = 56/1024; lower = P(X>=2) = 1013/1024; 2*min = 112/1024 = 0.109375
  assert.equal(binomialTwoSided(8, 10), 0.109375);
});

test("binomialTwoSided: k=0 is well-defined (does not divide by zero or return NaN)", () => {
  // upper = P(X>=0) = 1; lower = P(X>=5) = 1/32; 2*min(1, 1/32) = 1/16 = 0.0625
  assert.equal(binomialTwoSided(0, 5), 0.0625);
});

test("binomialTwoSided: is symmetric in k <-> n-k", () => {
  assert.equal(binomialTwoSided(3, 9), binomialTwoSided(6, 9));
});

test("binomialTwoSided: never exceeds 1", () => {
  for (let n = 1; n <= 12; n++) {
    for (let k = 0; k <= n; k++) {
      assert.ok(binomialTwoSided(k, n) <= 1, `k=${k}, n=${n}`);
    }
  }
});
