import { test } from "node:test";
import assert from "node:assert/strict";
import { postFreezeLossSeries, driftAttributableError, postFreezeActionDivergenceCount } from "../../src/experiment/metrics.ts";
import type { EpisodeStepRecord } from "../../src/experiment/freeze.ts";
import { Action } from "../../src/env/types.ts";

/** Minimal fabricated records — only `step`/`frozen`/`worldModelLoss` matter to metrics.ts. */
function record(step: number, worldModelLoss: (number | undefined)[], actions: Action[] = [Action.Stay, Action.Stay]): EpisodeStepRecord {
  return {
    step,
    observations: [[], []],
    nextObservations: [[], []],
    actions,
    reward: 0,
    done: false,
    frozen: [false, false],
    worldModelLoss,
  };
}

test("postFreezeLossSeries: extracts one agent's loss for every step at or after freezeStep, in order", () => {
  const records = [
    record(1, [1.0, 2.0]),
    record(2, [1.1, 2.1]),
    record(3, [1.2, 2.2]),
    record(4, [1.3, 2.3]),
  ];
  assert.deepEqual(postFreezeLossSeries(records, 3, 0), [1.2, 1.3]);
  assert.deepEqual(postFreezeLossSeries(records, 3, 1), [2.2, 2.3]);
  assert.deepEqual(postFreezeLossSeries(records, 1, 0), [1.0, 1.1, 1.2, 1.3]);
});

test("postFreezeLossSeries: throws when freezeStep never occurs in records", () => {
  const records = [record(1, [1.0, 2.0]), record(2, [1.1, 2.1])];
  assert.throws(() => postFreezeLossSeries(records, 5, 0), /freezeStep 5 never occurs/);
});

test("postFreezeLossSeries: throws when the agent's worldModelLoss is undefined at a post-freeze step", () => {
  const records = [record(1, [1.0, undefined]), record(2, [1.1, undefined])];
  assert.throws(() => postFreezeLossSeries(records, 1, 1), /worldModelLoss\[1\] is undefined at step 1/);
});

test("driftAttributableError: elementwise intervention-minus-control diff, aligned by steps-since-freeze", () => {
  const intervention = [1.0, 1.5, 2.0];
  const control = [1.0, 1.0, 1.0];
  assert.deepEqual(driftAttributableError(intervention, control), [0, 0.5, 1.0]);
});

test("driftAttributableError: control flat at the same level as intervention gives all-zero drift", () => {
  const series = [0.5, 0.5, 0.5];
  assert.deepEqual(driftAttributableError(series, series), [0, 0, 0]);
});

test("driftAttributableError: throws on length mismatch rather than truncating", () => {
  assert.throws(() => driftAttributableError([1, 2, 3], [1, 2]), /must be the same length/);
});

test("postFreezeActionDivergenceCount: counts post-freeze steps where actions differ, aligned by steps-since-freeze", () => {
  const control = [
    record(1, [1.0, 1.0], [Action.Stay, Action.Stay]),
    record(2, [1.0, 1.0], [Action.Up, Action.Stay]),
    record(3, [1.0, 1.0], [Action.Up, Action.Down]),
    record(4, [1.0, 1.0], [Action.Left, Action.Right]),
  ];
  const intervention = [
    record(1, [1.0, 1.0], [Action.Stay, Action.Stay]),
    record(2, [1.0, 1.0], [Action.Up, Action.Stay]),
    record(3, [1.0, 1.0], [Action.Down, Action.Down]),
    record(4, [1.0, 1.0], [Action.Left, Action.Left]),
  ];
  assert.deepEqual(postFreezeActionDivergenceCount(control, intervention, 2), {
    postFreezeSteps: 3,
    divergentSteps: 2,
  });
});

test("postFreezeActionDivergenceCount: zero divergent steps when every post-freeze action matches", () => {
  const control = [record(5, [1.0, 1.0], [Action.Up, Action.Down]), record(6, [1.0, 1.0], [Action.Left, Action.Right])];
  const intervention = [record(5, [1.0, 1.0], [Action.Up, Action.Down]), record(6, [1.0, 1.0], [Action.Left, Action.Right])];
  assert.deepEqual(postFreezeActionDivergenceCount(control, intervention, 5), {
    postFreezeSteps: 2,
    divergentSteps: 0,
  });
});

test("postFreezeActionDivergenceCount: throws when freezeStep never occurs in control's records", () => {
  const records = [record(1, [1.0, 1.0]), record(2, [1.0, 1.0])];
  assert.throws(() => postFreezeActionDivergenceCount(records, records, 5), /freezeStep 5 never occurs/);
});

test("postFreezeActionDivergenceCount: throws on post-freeze step-count mismatch rather than truncating", () => {
  const control = [record(1, [1.0, 1.0]), record(2, [1.0, 1.0]), record(3, [1.0, 1.0])];
  const intervention = [record(1, [1.0, 1.0]), record(2, [1.0, 1.0])];
  assert.throws(() => postFreezeActionDivergenceCount(control, intervention, 1), /control has 3 post-freeze steps, intervention has 2/);
});
