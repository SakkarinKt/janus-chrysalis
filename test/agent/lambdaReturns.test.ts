import { test } from "node:test";
import assert from "node:assert/strict";
import { computeLambdaReturns } from "../../src/agent/lambdaReturns.ts";
import type { LambdaReturnsInput } from "../../src/agent/lambdaReturns.ts";

// Gate G2 role-flip (docs/explainers/0012 + its 2026-09-23 errata). The ten property tests below
// were written by Claude as reviewer BEFORE any implementation existed, from the contract alone,
// per CONTRIBUTING.md's "nobody verifies their own work". They were first committed as f9754f5 on
// a branch that never got a PR; this file rescues them (2026-11-09) so they are not lost.
//
// computeLambdaReturns is still a throwing stub, so each property test carries `todo: PENDING`:
// node:test runs the body, reports it as TODO, and does not fail `npm test` while it is red.
// The first test is real and pins the stub's current behaviour.
//
// When you implement computeLambdaReturns: delete the stub test, delete `{ todo: PENDING }` from
// every other test, and all ten must pass unchanged.
//
// Every expected value is hand-computed from the backward recursion
//
//   R_t = r_t + γ · c_t · ((1 − λ) · V_{t+1} + λ · R_{t+1}),   V_T = R_T = bootstrapValue,
//   V_{t+1} = values[t + 1] for t < T − 1,
//
// with γ = λ = 0.5 so every intermediate value is exact in binary floating point. The tolerance is
// only there so either loop direction or association order passes.

const PENDING = "Gate G2 — red until computeLambdaReturns is implemented (docs/explainers/0012)";

const EPS = 1e-12;

function assertReturnsClose(actual: number[], expected: number[], message: string): void {
  assert.equal(actual.length, expected.length, `${message}: length ${actual.length} vs ${expected.length}`);
  actual.forEach((a, i) => {
    const e = expected[i] ?? NaN;
    assert.ok(Math.abs(a - e) < EPS, `${message}: R_${i} = ${a}, expected ${e}`);
  });
}

/** The worked example every other test perturbs. values[0] = 10 is deliberately never read. */
function base(overrides: Partial<LambdaReturnsInput> = {}): LambdaReturnsInput {
  return {
    rewards: [1, 2, 3],
    values: [10, 20, 30],
    continues: [1, 1, 1],
    bootstrapValue: 40,
    gamma: 0.5,
    lambda: 0.5,
    ...overrides,
  };
}

test("computeLambdaReturns is an intentional Gate G2 stub — calling it throws", () => {
  assert.throws(() => computeLambdaReturns(base()), /Gate G2 spec-only skeleton/);
});

test("worked example: γ = λ = 0.5 gives [9.8125, 15.25, 23] (hand-computed backward recursion)", { todo: PENDING }, () => {
  // t=2: 3 + 0.5·(0.5·40 + 0.5·40)    = 23
  // t=1: 2 + 0.5·(0.5·30 + 0.5·23)    = 15.25
  // t=0: 1 + 0.5·(0.5·20 + 0.5·15.25) = 9.8125
  assertReturnsClose(computeLambdaReturns(base()), [9.8125, 15.25, 23], "worked example");
});

test("lambda: 0 reduces to the one-step TD target r_t + γ·c_t·V_{t+1}", { todo: PENDING }, () => {
  // [1 + 0.5·20, 2 + 0.5·30, 3 + 0.5·40]
  assertReturnsClose(computeLambdaReturns(base({ lambda: 0 })), [11, 17, 23], "λ = 0");
});

test(
  "lambda: 1 is the discounted return plus a bootstrapValue tail — pure Monte-Carlo only when bootstrapValue = 0 (0012 errata #2)",
  { todo: PENDING },
  () => {
    // With the tail: R_2 = 3 + 0.5·40 = 23, R_1 = 2 + 0.5·23 = 13.5, R_0 = 1 + 0.5·13.5 = 7.75.
    assertReturnsClose(computeLambdaReturns(base({ lambda: 1 })), [7.75, 13.5, 23], "λ = 1, bootstrap 40");
    // Without it: the plain discounted sum of rewards. R_2 = 3, R_1 = 2 + 1.5 = 3.5, R_0 = 1 + 1.75 = 2.75.
    assertReturnsClose(
      computeLambdaReturns(base({ lambda: 1, bootstrapValue: 0 })),
      [2.75, 3.5, 3],
      "λ = 1, bootstrap 0 (Monte-Carlo)",
    );
  },
);

test(
  "continues[t] === 0 severs bootstrapping at step t: R_t = r_t, and earlier returns stop depending on anything after t",
  { todo: PENDING },
  () => {
    // continues [1, 0, 1]: R_2 = 23; R_1 = 2 (severed); R_0 = 1 + 0.5·(0.5·20 + 0.5·2) = 6.5.
    const severed = computeLambdaReturns(base({ continues: [1, 0, 1] }));
    assertReturnsClose(severed, [6.5, 2, 23], "continues[1] = 0");

    // Changing everything past the cut (r_2, V_2, bootstrap) must leave R_0 and R_1 untouched.
    const perturbed = computeLambdaReturns(
      base({ continues: [1, 0, 1], rewards: [1, 2, -100], values: [10, 20, -100], bootstrapValue: -100 }),
    );
    assertReturnsClose(perturbed.slice(0, 2), [6.5, 2], "past-the-cut perturbation leaks backward");
  },
);

test("fractional continues scale the bootstrap term — a continuation probability, not a boolean", { todo: PENDING }, () => {
  // continues [1, 0.5, 1]: R_2 = 23; R_1 = 2 + 0.5·0.5·(0.5·30 + 0.5·23) = 8.625;
  // R_0 = 1 + 0.5·(0.5·20 + 0.5·8.625) = 8.15625.
  assertReturnsClose(computeLambdaReturns(base({ continues: [1, 0.5, 1] })), [8.15625, 8.625, 23], "continues[1] = 0.5");
});

test(
  "truncation keeps bootstrapping: with continues all 1 (this env's done is always a time limit), the last return is r_{T-1} + γ·bootstrapValue, not r_{T-1} (0012 errata #1)",
  { todo: PENDING },
  () => {
    const returns = computeLambdaReturns(base());
    assert.ok(Math.abs((returns.at(-1) ?? NaN) - (3 + 0.5 * 40)) < EPS, `R_{T-1} = ${returns.at(-1)}`);
    assert.notEqual(returns.at(-1), 3, "last return equals the bare reward: bootstrapping was severed at a truncation");
  },
);

test(
  "values[0] is never read by the recursion — changing it changes no return (0012 errata #3; catches an off-by-one in V_{t+1})",
  { todo: PENDING },
  () => {
    assertReturnsClose(computeLambdaReturns(base({ values: [-999, 20, 30] })), [9.8125, 15.25, 23], "values[0] perturbed");
  },
);

test("output length equals rewards.length, including a single-step segment", { todo: PENDING }, () => {
  assert.equal(computeLambdaReturns(base()).length, 3);
  // One step: R_0 = 5 + 0.5·(0.5·2 + 0.5·2) = 6 — values[0] unused, bootstrap does both jobs.
  assertReturnsClose(
    computeLambdaReturns({ rewards: [5], values: [7], continues: [1], bootstrapValue: 2, gamma: 0.5, lambda: 0.5 }),
    [6],
    "single step",
  );
});

test(
  "mismatched rewards/values/continues lengths throw rather than silently truncating or padding",
  { todo: PENDING },
  () => {
    // Passes vacuously while the stub exists (it throws for every input), which is why this test
    // stays flagged with the rest. The contract fixes no error message, so none is asserted.
    assert.throws(() => computeLambdaReturns(base({ values: [10, 20] })));
    assert.throws(
      () => computeLambdaReturns(base({ values: [10, 20, 30, 40] })),
      "a length-(T+1) values array must not be read as including the bootstrap",
    );
    assert.throws(() => computeLambdaReturns(base({ continues: [1, 1] })));
    assert.throws(() => computeLambdaReturns(base({ continues: [1, 1, 1, 1] })));
  },
);

test("does not mutate its input arrays", { todo: PENDING }, () => {
  const input = base();
  const snapshot = structuredClone(input);
  computeLambdaReturns(input);
  assert.deepEqual(input, snapshot);
});
