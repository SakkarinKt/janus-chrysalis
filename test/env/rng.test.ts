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

  // One Rng per stream, drawn 10 times — the pre-2026-09-23 version built a
  // fresh Rng per draw, comparing ten copies of each stream's *first* value.
  const rngA = new Rng(seedA);
  const rngB = new Rng(seedB);
  const seqA = Array.from({ length: 10 }, () => rngA.next());
  const seqB = Array.from({ length: 10 }, () => rngB.next());
  assert.notDeepEqual(seqA, seqB);
});

test("deriveSeed: also differs from the base seed itself (agent streams decouple from the policy stream)", () => {
  const base = 99;
  assert.notEqual(deriveSeed(base, 0), base);
});

test("deriveSeed: nested derivation is order-sensitive and never cancels back to the base seed (session audit N2, 2026-09-23)", () => {
  // Callers nest: experiment deriveSeed(seed, agent) → WorldModel salts 0–3 →
  // RSSMCell salts 0–3. A commutative/self-inverse mix (plain XOR) made
  // agent 0's GRU kernel seed equal agent 1's recurrent-kernel seed.
  for (const base of [1001, 1002, 1003, 0, 0xffffffff]) {
    for (const a of [0, 1, 2, 3]) {
      assert.notEqual(deriveSeed(deriveSeed(base, a), a), base, `f(f(${base},${a}),${a}) collapsed to base`);
      for (const b of [0, 1, 2, 3]) {
        if (a === b) continue;
        assert.notEqual(
          deriveSeed(deriveSeed(base, a), b),
          deriveSeed(deriveSeed(base, b), a),
          `f(f(${base},${a}),${b}) === f(f(${base},${b}),${a})`,
        );
      }
    }
  }
});

test("deriveSeed: the full experiment → WorldModel → RSSMCell seed tree for two agents has no duplicate seeds", () => {
  const seen = new Map<number, string>();
  for (const seed of [1001, 1002, 1003]) {
    for (const agent of [0, 1]) {
      const wm = deriveSeed(seed, agent);
      for (const [label, value] of [
        ["wm", wm],
        ...[0, 1, 2, 3].map((s) => [`wm.salt${s}`, deriveSeed(wm, s)] as const),
        ...[0, 1, 2, 3].map((s) => [`rssm.salt${s}`, deriveSeed(deriveSeed(wm, 0), s)] as const),
      ] as const) {
        const key = `seed${seed}/agent${agent}/${label}`;
        assert.ok(!seen.has(value), `${key} duplicates ${seen.get(value)} (${value})`);
        seen.set(value, key);
      }
    }
    assert.ok(!seen.has(seed), `raw seed ${seed} reappears in the derived tree as ${seen.get(seed)}`);
  }
});
