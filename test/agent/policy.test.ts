import { test } from "node:test";
import assert from "node:assert/strict";
import { QLearningPolicy, RandomPolicy } from "../../src/agent/policy.ts";
import type { Policy } from "../../src/agent/policy.ts";
import { Action } from "../../src/env/types.ts";
import { Rng } from "../../src/env/rng.ts";

test("RandomPolicy.act always returns a valid action", () => {
  const policy = new RandomPolicy();
  const rng = new Rng(42);
  const validActions = new Set(Object.values(Action));
  for (let i = 0; i < 100; i++) {
    const action = policy.act([], rng);
    assert.ok(validActions.has(action), `${action} is not a valid Action`);
  }
});

test("RandomPolicy.act is driven by the passed-in rng, not internal state", () => {
  const runActions = (seed: number) => {
    const policy = new RandomPolicy();
    const rng = new Rng(seed);
    return Array.from({ length: 20 }, () => policy.act([], rng));
  };
  assert.deepEqual(runActions(7), runActions(7));
});

test("RandomPolicy has no update hook", () => {
  // Typed as `Policy`, not `RandomPolicy`: the class itself declares no
  // `update` property (only `Policy`'s optional signature does), so
  // `policy.update` doesn't type-check against the concrete class — see
  // reports/quality/2026-08-03-quality-pass.md.
  const policy: Policy = new RandomPolicy();
  assert.equal(policy.update, undefined);
});

test("QLearningPolicy.act always returns a valid action", () => {
  const policy = new QLearningPolicy();
  const rng = new Rng(42);
  const validActions = new Set(Object.values(Action));
  for (let i = 0; i < 100; i++) {
    const action = policy.act([0, 0], rng);
    assert.ok(validActions.has(action), `${action} is not a valid Action`);
  }
});

test("QLearningPolicy.act is deterministic given the same rng seed", () => {
  const runActions = (seed: number) => {
    const policy = new QLearningPolicy();
    const rng = new Rng(seed);
    return Array.from({ length: 20 }, () => policy.act([0, 0], rng));
  };
  assert.deepEqual(runActions(7), runActions(7));
});

test("QLearningPolicy has an update hook", () => {
  const policy: Policy = new QLearningPolicy();
  assert.equal(typeof policy.update, "function");
});

test("QLearningPolicy.act defaults to the leftmost action (Stay) on an unseen observation, epsilon 0", () => {
  const policy = new QLearningPolicy({ epsilon: 0 });
  const rng = new Rng(1);
  assert.equal(policy.act([9, 9], rng), Action.Stay);
});

test("QLearningPolicy.act is greedy toward the highest-updated action when epsilon is 0", () => {
  const policy = new QLearningPolicy({ epsilon: 0 });
  const rng = new Rng(1);
  policy.update({
    observation: [1, 1],
    action: Action.Down,
    reward: 10,
    nextObservation: [0, 0],
    done: true,
  });
  assert.equal(policy.act([1, 1], rng), Action.Down);
});

test("QLearningPolicy.act always explores when epsilon is 1", () => {
  const policy = new QLearningPolicy({ epsilon: 1 });
  // Give Stay a very high Q-value; if exploration weren't happening every
  // call, the greedy action (Stay) would dominate the sample.
  policy.update({
    observation: [0, 0],
    action: Action.Stay,
    reward: 1000,
    nextObservation: [0, 0],
    done: true,
  });
  const rng = new Rng(3);
  const actions = Array.from({ length: 50 }, () => policy.act([0, 0], rng));
  assert.ok(
    actions.some((a) => a !== Action.Stay),
    "epsilon=1 should sometimes pick a non-greedy action",
  );
});

test("QLearningPolicy.update scales the TD update by alpha", () => {
  const policy = new QLearningPolicy({ alpha: 0.5, gamma: 0.9, epsilon: 0 });
  const rng = new Rng(1);
  const obsA = [1, 0];
  const obsFresh = [0, 1]; // never visited elsewhere -> zero row, nextMax = 0

  // Q(obsA, Up) = 0 + 0.5 * (4 + 0.9*0 - 0) = 2
  policy.update({ observation: obsA, action: Action.Up, reward: 4, nextObservation: obsFresh, done: false });
  // Decoy: Q(obsA, Down) = 0 + 0.5 * (1 + 0.9*0 - 0) = 0.5, below Up's 2.
  policy.update({ observation: obsA, action: Action.Down, reward: 1, nextObservation: obsFresh, done: false });

  assert.equal(policy.act(obsA, rng), Action.Up);
});

test("QLearningPolicy.update bootstraps gamma*max(next) when done is false", () => {
  const policy = new QLearningPolicy({ alpha: 1, gamma: 0.9, epsilon: 0 });
  const rng = new Rng(1);
  const obsA = [2, 2];
  const obsB = [3, 3];
  const obsZ = [9, 9];

  // Q(obsB, Stay) = 1000, isolated (obsZ's row is all-zero -> no bootstrap contribution here).
  policy.update({ observation: obsB, action: Action.Stay, reward: 1000, nextObservation: obsZ, done: false });
  // Decoy: Q(obsA, Down) = 800, isolated the same way.
  policy.update({ observation: obsA, action: Action.Down, reward: 800, nextObservation: obsZ, done: false });
  // Q(obsA, Up) = 5 + 0.9 * max(Q(obsB, *)) = 5 + 900 = 905, above the 800 decoy.
  policy.update({ observation: obsA, action: Action.Up, reward: 5, nextObservation: obsB, done: false });

  assert.equal(policy.act(obsA, rng), Action.Up);
});

test("QLearningPolicy.update does not bootstrap from the next observation when done is true", () => {
  const policy = new QLearningPolicy({ alpha: 1, gamma: 0.9, epsilon: 0 });
  const rng = new Rng(1);
  const obsA = [2, 2];
  const obsB = [3, 3]; // holds a large decoy Q-value that must NOT leak in via bootstrap
  const obsZ = [9, 9];

  policy.update({ observation: obsB, action: Action.Stay, reward: 1000, nextObservation: obsZ, done: false });
  // Decoy: Q(obsA, Down) = 800, isolated the same way.
  policy.update({ observation: obsA, action: Action.Down, reward: 800, nextObservation: obsZ, done: false });
  // If done: true correctly suppresses the bootstrap, Q(obsA, Up) = 5 + 0.9*0 = 5, well below
  // the 800 decoy. A bootstrap bug would instead land it at 5 + 0.9*1000 = 905, above the decoy,
  // flipping the greedy action.
  policy.update({ observation: obsA, action: Action.Up, reward: 5, nextObservation: obsB, done: true });

  assert.equal(policy.act(obsA, rng), Action.Down);
});
