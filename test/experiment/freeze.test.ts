import { test } from "node:test";
import assert from "node:assert/strict";
import { isFrozen, runEpisode } from "../../src/experiment/freeze.ts";
import type { FreezeConfig } from "../../src/experiment/freeze.ts";
import { CooperativeGridWorld } from "../../src/env/gridworld.ts";
import { RandomPolicy } from "../../src/agent/policy.ts";
import type { Policy, Transition } from "../../src/agent/policy.ts";
import { Action } from "../../src/env/types.ts";
import { Rng } from "../../src/env/rng.ts";

/** Test double: acts deterministically (always Stay) and counts update() calls. */
class CountingPolicy implements Policy {
  updateCalls = 0;
  updateSteps: number[] = [];
  private step = 0;

  act(): Action {
    return Action.Stay;
  }

  update(_transition: Transition): void {
    this.updateCalls += 1;
    this.step += 1;
    this.updateSteps.push(this.step);
  }
}

test("isFrozen: before freezeStep, neither condition freezes anyone", () => {
  const control: FreezeConfig = { freezeStep: 10, condition: "control" };
  const intervention: FreezeConfig = { freezeStep: 10, condition: "intervention", frozenAgentIndex: 0 };
  for (const step of [0, 1, 9]) {
    assert.equal(isFrozen(0, step, control), false);
    assert.equal(isFrozen(1, step, control), false);
    assert.equal(isFrozen(0, step, intervention), false);
    assert.equal(isFrozen(1, step, intervention), false);
  }
});

test("isFrozen: control condition freezes both agents at and after freezeStep", () => {
  const config: FreezeConfig = { freezeStep: 5, condition: "control" };
  assert.equal(isFrozen(0, 5, config), true);
  assert.equal(isFrozen(1, 5, config), true);
  assert.equal(isFrozen(0, 12, config), true);
  assert.equal(isFrozen(1, 12, config), true);
});

test("isFrozen: intervention condition freezes only the named agent", () => {
  const config: FreezeConfig = { freezeStep: 5, condition: "intervention", frozenAgentIndex: 1 };
  assert.equal(isFrozen(1, 5, config), true);
  assert.equal(isFrozen(0, 5, config), false);
  assert.equal(isFrozen(1, 20, config), true);
  assert.equal(isFrozen(0, 20, config), false);
});

test("isFrozen: intervention without frozenAgentIndex throws", () => {
  const config = { freezeStep: 5, condition: "intervention" } as FreezeConfig;
  assert.throws(() => isFrozen(0, 5, config));
});

test("runEpisode: without a freezeConfig, no agent is ever frozen and both policies update every step", () => {
  const env = new CooperativeGridWorld({ seed: 1, horizon: 6 });
  const a = new CountingPolicy();
  const b = new CountingPolicy();
  const records = runEpisode(env, [a, b], new Rng(1));

  assert.equal(records.length, 6);
  for (const record of records) {
    assert.deepEqual(record.frozen, [false, false]);
  }
  assert.equal(a.updateCalls, 6);
  assert.equal(b.updateCalls, 6);
});

test("runEpisode: intervention condition stops the frozen agent's updates from freezeStep onward, not the partner's", () => {
  const env = new CooperativeGridWorld({ seed: 2, horizon: 8 });
  const frozenAgent = new CountingPolicy();
  const trainingAgent = new CountingPolicy();
  const freezeConfig: FreezeConfig = { freezeStep: 4, condition: "intervention", frozenAgentIndex: 0 };

  const records = runEpisode(env, [frozenAgent, trainingAgent], new Rng(2), freezeConfig);

  assert.equal(records.length, 8);
  for (const record of records) {
    assert.deepEqual(record.frozen, [isFrozen(0, record.step, freezeConfig), isFrozen(1, record.step, freezeConfig)]);
  }
  // Steps 1-3 update both; steps 4-8 update only the training agent.
  assert.equal(frozenAgent.updateCalls, 3);
  assert.deepEqual(frozenAgent.updateSteps, [1, 2, 3]);
  assert.equal(trainingAgent.updateCalls, 8);
});

test("runEpisode: control condition stops both agents' updates from freezeStep onward", () => {
  const env = new CooperativeGridWorld({ seed: 3, horizon: 8 });
  const a = new CountingPolicy();
  const b = new CountingPolicy();
  const freezeConfig: FreezeConfig = { freezeStep: 4, condition: "control" };

  runEpisode(env, [a, b], new Rng(3), freezeConfig);

  assert.equal(a.updateCalls, 3);
  assert.equal(b.updateCalls, 3);
  assert.deepEqual(a.updateSteps, [1, 2, 3]);
  assert.deepEqual(b.updateSteps, [1, 2, 3]);
});

test("runEpisode: works end-to-end with RandomPolicy (no update hook) and reaches the configured horizon", () => {
  const env = new CooperativeGridWorld({ seed: 4, horizon: 10 });
  const records = runEpisode(env, [new RandomPolicy(), new RandomPolicy()], new Rng(4), {
    freezeStep: 5,
    condition: "intervention",
    frozenAgentIndex: 1,
  });
  assert.equal(records.length, 10);
  assert.equal(records[records.length - 1].done, true);
});

test("runEpisode: wrong number of policies throws", () => {
  const env = new CooperativeGridWorld({ seed: 5, horizon: 3 });
  assert.throws(() => runEpisode(env, [new RandomPolicy()], new Rng(5)));
});
