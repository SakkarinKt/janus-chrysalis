import { test } from "node:test";
import assert from "node:assert/strict";
import { RandomPolicy } from "../../src/agent/policy.ts";
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
  const policy = new RandomPolicy();
  assert.equal(policy.update, undefined);
});
