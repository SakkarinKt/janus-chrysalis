import { test } from "node:test";
import assert from "node:assert/strict";
import { CooperativeGridWorld } from "../../src/env/gridworld.ts";
import { Action } from "../../src/env/types.ts";

test("reset returns two fixed-length observations at step 0", () => {
  const env = new CooperativeGridWorld({ seed: 1 });
  const { observations, step } = env.reset();
  assert.equal(step, 0);
  assert.equal(observations.length, 2);
  for (const obs of observations) {
    assert.equal(obs.length, env.observationLength);
  }
});

test("same seed + same actions produce identical trajectories", () => {
  const actions: Action[][] = [
    [Action.Up, Action.Right],
    [Action.Right, Action.Up],
    [Action.Stay, Action.Down],
  ];
  const run = (seed: number) => {
    const env = new CooperativeGridWorld({ seed });
    env.reset();
    return actions.map((a) => env.step(a));
  };
  const a = run(99);
  const b = run(99);
  assert.deepEqual(a, b);
});

test("different seeds diverge in spawn position", () => {
  const envA = new CooperativeGridWorld({ seed: 1 });
  const envB = new CooperativeGridWorld({ seed: 2 });
  const a = envA.reset();
  const b = envB.reset();
  assert.notDeepEqual(a.observations, b.observations);
});

test("movement clamps at grid boundaries", () => {
  const env = new CooperativeGridWorld({ seed: 5, gridSize: 6, horizon: 50 });
  env.reset();
  // Drive both agents into the top-left corner, well past the grid edge.
  for (let i = 0; i < 20; i++) {
    env.step([Action.Up, Action.Left]);
  }
  const [agent0] = env.getAgentPositions();
  assert.equal(agent0.y, 0);
  const [, agent1] = env.getAgentPositions();
  assert.equal(agent1.x, 0);
});

test("partial observability masks entities outside the view radius", () => {
  const env = new CooperativeGridWorld({ seed: 3, gridSize: 8, viewRadius: 1, numLandmarks: 2 });
  const { observations } = env.reset();
  const [selfPos, otherPos] = env.getAgentPositions();
  const landmarks = env.getLandmarkPositions();

  const manhattan = (a: { x: number; y: number }, b: { x: number; y: number }) =>
    Math.abs(a.x - b.x) + Math.abs(a.y - b.y);

  const obs0 = observations[0];
  // Layout: [selfX, selfY, lm0_vis, lm0_dx, lm0_dy, lm1_vis, lm1_dx, lm1_dy, other_vis, other_dx, other_dy]
  landmarks.forEach((lm, i) => {
    const visible = obs0[2 + i * 3];
    const expectedVisible = manhattan(selfPos, lm) <= 1 ? 1 : 0;
    assert.equal(visible, expectedVisible, `landmark ${i} visibility mismatch`);
    if (!expectedVisible) {
      assert.equal(obs0[2 + i * 3 + 1], 0);
      assert.equal(obs0[2 + i * 3 + 2], 0);
    }
  });
  const otherOffset = 2 + landmarks.length * 3;
  const expectedOtherVisible = manhattan(selfPos, otherPos) <= 1 ? 1 : 0;
  assert.equal(obs0[otherOffset], expectedOtherVisible);
});

test("shared reward matches the coverage formula when agents hold still", () => {
  const env = new CooperativeGridWorld({ seed: 4, gridSize: 8, numLandmarks: 2 });
  env.reset();
  const agents = env.getAgentPositions();
  const landmarks = env.getLandmarkPositions();
  const manhattan = (a: { x: number; y: number }, b: { x: number; y: number }) =>
    Math.abs(a.x - b.x) + Math.abs(a.y - b.y);
  const expectedCoverage = landmarks.reduce(
    (sum, lm) => sum + Math.min(...agents.map((a) => manhattan(a, lm))),
    0,
  );
  const { reward } = env.step([Action.Stay, Action.Stay]);
  assert.equal(reward, -expectedCoverage / 8);
});

test("collision penalty applies when agents share a cell", () => {
  const env = new CooperativeGridWorld({
    seed: 6,
    gridSize: 4,
    horizon: 50,
    collisionPenalty: 0.5,
  });
  env.reset();
  // Force both agents toward the same corner until they occupy one cell.
  let last;
  for (let i = 0; i < 10; i++) {
    last = env.step([Action.Up, Action.Up]);
    last = env.step([Action.Left, Action.Left]);
  }
  const [a0, a1] = env.getAgentPositions();
  assert.deepEqual(a0, a1);
  assert.ok(last !== undefined);
});

test("episode reaches done=true exactly at the configured horizon", () => {
  const horizon = 5;
  const env = new CooperativeGridWorld({ seed: 8, horizon });
  env.reset();
  for (let i = 1; i < horizon; i++) {
    const result = env.step([Action.Stay, Action.Stay]);
    assert.equal(result.done, false, `expected not done at step ${i}`);
    assert.equal(result.step, i);
  }
  const final = env.step([Action.Stay, Action.Stay]);
  assert.equal(final.done, true);
  assert.equal(final.step, horizon);
});

test("stepping past the horizon without reset throws", () => {
  const env = new CooperativeGridWorld({ seed: 9, horizon: 2 });
  env.reset();
  env.step([Action.Stay, Action.Stay]);
  env.step([Action.Stay, Action.Stay]);
  assert.throws(() => env.step([Action.Stay, Action.Stay]));
});

test("wrong action-array length throws", () => {
  const env = new CooperativeGridWorld({ seed: 10 });
  env.reset();
  assert.throws(() => env.step([Action.Stay]));
});
