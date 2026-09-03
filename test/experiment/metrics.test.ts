import { test } from "node:test";
import assert from "node:assert/strict";
import {
  postFreezeLossSeries,
  driftAttributableError,
  postFreezeActionDivergenceCount,
  postFreezeObservationDivergenceCount,
  postFreezePartnerVisibleCount,
  postFreezeLandmarkVisibleCount,
} from "../../src/experiment/metrics.ts";
import type { EpisodeStepRecord } from "../../src/experiment/freeze.ts";
import { Action } from "../../src/env/types.ts";

/**
 * Minimal fabricated records — only `step`/`frozen`/`worldModelLossBreakdown` matter to
 * metrics.ts. `reconKl` is each agent's `reconstructionLoss + klLoss` (what `postFreezeLossSeries`
 * reads); a fixed non-zero `continueLoss` is baked into `worldModelLoss` (the total) so a
 * regression to reading the total instead of the breakdown fails loudly rather than silently
 * passing — see PR #63's review / docs/explainers/0007's addendum.
 */
const CONTINUE_LOSS_OFFSET = 1000;

function record(
  step: number,
  reconKl: (number | undefined)[],
  actions: Action[] = [Action.Stay, Action.Stay],
  nextObservations: number[][] = [[], []],
): EpisodeStepRecord {
  return {
    step,
    observations: [[], []],
    nextObservations,
    actions,
    reward: 0,
    done: false,
    frozen: [false, false],
    worldModelLoss: reconKl.map((v) => (v === undefined ? undefined : v + CONTINUE_LOSS_OFFSET)),
    worldModelLossBreakdown: reconKl.map((v) =>
      v === undefined ? undefined : { reconstructionLoss: v, klLoss: 0, continueLoss: CONTINUE_LOSS_OFFSET },
    ),
  };
}

test("postFreezeLossSeries: extracts one agent's reconstructionLoss+klLoss for every step at or after freezeStep, in order — excluding continueLoss", () => {
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

test("postFreezeLossSeries: throws when the agent's worldModelLossBreakdown is undefined at a post-freeze step", () => {
  const records = [record(1, [1.0, undefined]), record(2, [1.1, undefined])];
  assert.throws(
    () => postFreezeLossSeries(records, 1, 1),
    /worldModelLossBreakdown\[1\] is undefined at step 1/,
  );
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

test("postFreezeObservationDivergenceCount: counts post-freeze steps where agentIndex's nextObservation differs", () => {
  const control = [
    record(1, [1.0, 1.0], undefined, [[0, 0], [0, 0]]),
    record(2, [1.0, 1.0], undefined, [[0.1, 0], [0, 0]]),
    record(3, [1.0, 1.0], undefined, [[0.1, 0.2], [0, 0]]),
    record(4, [1.0, 1.0], undefined, [[0.1, 0.2], [0, 0]]),
  ];
  const intervention = [
    record(1, [1.0, 1.0], undefined, [[0, 0], [0, 0]]),
    record(2, [1.0, 1.0], undefined, [[0.1, 0], [0, 0]]),
    record(3, [1.0, 1.0], undefined, [[0.3, 0.2], [0, 0]]),
    record(4, [1.0, 1.0], undefined, [[0.1, 0.5], [0, 0]]),
  ];
  assert.deepEqual(postFreezeObservationDivergenceCount(control, intervention, 2, 0), {
    postFreezeSteps: 3,
    divergentSteps: 2,
  });
});

test("postFreezeObservationDivergenceCount: only agentIndex's own observation is compared, not the other agent's", () => {
  const control = [
    record(1, [1.0, 1.0], undefined, [[0, 0], [1, 1]]),
    record(2, [1.0, 1.0], undefined, [[0, 0], [2, 2]]),
  ];
  const intervention = [
    record(1, [1.0, 1.0], undefined, [[0, 0], [9, 9]]),
    record(2, [1.0, 1.0], undefined, [[0, 0], [8, 8]]),
  ];
  assert.deepEqual(postFreezeObservationDivergenceCount(control, intervention, 1, 0), {
    postFreezeSteps: 2,
    divergentSteps: 0,
  });
});

test("postFreezeObservationDivergenceCount: zero divergent steps when every post-freeze observation matches", () => {
  const control = [record(5, [1.0, 1.0], undefined, [[0.4, 0.4], [0, 0]])];
  const intervention = [record(5, [1.0, 1.0], undefined, [[0.4, 0.4], [0, 0]])];
  assert.deepEqual(postFreezeObservationDivergenceCount(control, intervention, 5, 0), {
    postFreezeSteps: 1,
    divergentSteps: 0,
  });
});

test("postFreezeObservationDivergenceCount: throws when freezeStep never occurs in control's records", () => {
  const records = [record(1, [1.0, 1.0]), record(2, [1.0, 1.0])];
  assert.throws(() => postFreezeObservationDivergenceCount(records, records, 5, 0), /freezeStep 5 never occurs/);
});

test("postFreezeObservationDivergenceCount: throws on post-freeze step-count mismatch rather than truncating", () => {
  const control = [record(1, [1.0, 1.0]), record(2, [1.0, 1.0]), record(3, [1.0, 1.0])];
  const intervention = [record(1, [1.0, 1.0]), record(2, [1.0, 1.0])];
  assert.throws(
    () => postFreezeObservationDivergenceCount(control, intervention, 1, 0),
    /control has 3 post-freeze steps, intervention has 2/,
  );
});

/** Otherwise-arbitrary vector whose last 3 elements are [otherAgent_visible, otherAgent_dx, otherAgent_dy]. */
const obsWithVisibility = (visible: 0 | 1): number[] => [0, 0, visible, visible ? 0.1 : 0, visible ? 0.2 : 0];

test("postFreezePartnerVisibleCount: counts a step visible when the partner is visible in either control or intervention", () => {
  const control = [
    record(1, [1.0, 1.0], undefined, [obsWithVisibility(0), []]), // neither visible
    record(2, [1.0, 1.0], undefined, [obsWithVisibility(1), []]), // control only
    record(3, [1.0, 1.0], undefined, [obsWithVisibility(0), []]), // intervention only
    record(4, [1.0, 1.0], undefined, [obsWithVisibility(1), []]), // both visible
  ];
  const intervention = [
    record(1, [1.0, 1.0], undefined, [obsWithVisibility(0), []]),
    record(2, [1.0, 1.0], undefined, [obsWithVisibility(0), []]),
    record(3, [1.0, 1.0], undefined, [obsWithVisibility(1), []]),
    record(4, [1.0, 1.0], undefined, [obsWithVisibility(1), []]),
  ];
  assert.deepEqual(postFreezePartnerVisibleCount(control, intervention, 1, 0), {
    postFreezeSteps: 4,
    visibleSteps: 3,
  });
});

test("postFreezePartnerVisibleCount: zero visible steps when the partner is masked in both conditions throughout", () => {
  const control = [record(5, [1.0, 1.0], undefined, [obsWithVisibility(0), []])];
  const intervention = [record(5, [1.0, 1.0], undefined, [obsWithVisibility(0), []])];
  assert.deepEqual(postFreezePartnerVisibleCount(control, intervention, 5, 0), {
    postFreezeSteps: 1,
    visibleSteps: 0,
  });
});

test("postFreezePartnerVisibleCount: throws when freezeStep never occurs in control's records", () => {
  const records = [record(1, [1.0, 1.0], undefined, [obsWithVisibility(0), []]), record(2, [1.0, 1.0], undefined, [obsWithVisibility(0), []])];
  assert.throws(() => postFreezePartnerVisibleCount(records, records, 5, 0), /freezeStep 5 never occurs/);
});

test("postFreezePartnerVisibleCount: throws on post-freeze step-count mismatch rather than truncating", () => {
  const control = [
    record(1, [1.0, 1.0], undefined, [obsWithVisibility(0), []]),
    record(2, [1.0, 1.0], undefined, [obsWithVisibility(0), []]),
    record(3, [1.0, 1.0], undefined, [obsWithVisibility(0), []]),
  ];
  const intervention = [record(1, [1.0, 1.0], undefined, [obsWithVisibility(0), []]), record(2, [1.0, 1.0], undefined, [obsWithVisibility(0), []])];
  assert.throws(
    () => postFreezePartnerVisibleCount(control, intervention, 1, 0),
    /control has 3 post-freeze steps, intervention has 2/,
  );
});

test("postFreezePartnerVisibleCount: throws when an observation is too short to contain a visibility flag", () => {
  const control = [record(1, [1.0, 1.0], undefined, [[], []])];
  const intervention = [record(1, [1.0, 1.0], undefined, [obsWithVisibility(0), []])];
  assert.throws(() => postFreezePartnerVisibleCount(control, intervention, 1, 0), /observation too short/);
});

/** Vector shaped [selfX, selfY, lm0_vis, lm0_dx, lm0_dy, lm1_vis, lm1_dx, lm1_dy], numLandmarks=2. */
const obsWithLandmarkVisibility = (lm0: 0 | 1, lm1: 0 | 1): number[] => [
  0,
  0,
  lm0,
  lm0 ? 0.1 : 0,
  lm0 ? 0.2 : 0,
  lm1,
  lm1 ? 0.3 : 0,
  lm1 ? 0.4 : 0,
];

test("postFreezeLandmarkVisibleCount: counts a step visible when any landmark is visible in either control or intervention", () => {
  const control = [
    record(1, [1.0, 1.0], undefined, [obsWithLandmarkVisibility(0, 0), []]), // neither visible, either run
    record(2, [1.0, 1.0], undefined, [obsWithLandmarkVisibility(1, 0), []]), // control's landmark 0 only
    record(3, [1.0, 1.0], undefined, [obsWithLandmarkVisibility(0, 0), []]), // intervention's landmark 1 only
    record(4, [1.0, 1.0], undefined, [obsWithLandmarkVisibility(1, 1), []]), // both visible
  ];
  const intervention = [
    record(1, [1.0, 1.0], undefined, [obsWithLandmarkVisibility(0, 0), []]),
    record(2, [1.0, 1.0], undefined, [obsWithLandmarkVisibility(0, 0), []]),
    record(3, [1.0, 1.0], undefined, [obsWithLandmarkVisibility(0, 1), []]),
    record(4, [1.0, 1.0], undefined, [obsWithLandmarkVisibility(1, 1), []]),
  ];
  assert.deepEqual(postFreezeLandmarkVisibleCount(control, intervention, 1, 0, 2), {
    postFreezeSteps: 4,
    visibleSteps: 3,
  });
});

test("postFreezeLandmarkVisibleCount: zero visible steps when every landmark is masked in both conditions throughout", () => {
  const control = [record(5, [1.0, 1.0], undefined, [obsWithLandmarkVisibility(0, 0), []])];
  const intervention = [record(5, [1.0, 1.0], undefined, [obsWithLandmarkVisibility(0, 0), []])];
  assert.deepEqual(postFreezeLandmarkVisibleCount(control, intervention, 5, 0, 2), {
    postFreezeSteps: 1,
    visibleSteps: 0,
  });
});

test("postFreezeLandmarkVisibleCount: throws when freezeStep never occurs in control's records", () => {
  const records = [record(1, [1.0, 1.0], undefined, [obsWithLandmarkVisibility(0, 0), []])];
  assert.throws(() => postFreezeLandmarkVisibleCount(records, records, 5, 0, 2), /freezeStep 5 never occurs/);
});

test("postFreezeLandmarkVisibleCount: throws on post-freeze step-count mismatch rather than truncating", () => {
  const control = [
    record(1, [1.0, 1.0], undefined, [obsWithLandmarkVisibility(0, 0), []]),
    record(2, [1.0, 1.0], undefined, [obsWithLandmarkVisibility(0, 0), []]),
  ];
  const intervention = [record(1, [1.0, 1.0], undefined, [obsWithLandmarkVisibility(0, 0), []])];
  assert.throws(
    () => postFreezeLandmarkVisibleCount(control, intervention, 1, 0, 2),
    /control has 2 post-freeze steps, intervention has 1/,
  );
});

test("postFreezeLandmarkVisibleCount: throws when an observation is too short to contain a landmark's visibility flag", () => {
  const control = [record(1, [1.0, 1.0], undefined, [[], []])];
  const intervention = [record(1, [1.0, 1.0], undefined, [obsWithLandmarkVisibility(0, 0), []])];
  assert.throws(() => postFreezeLandmarkVisibleCount(control, intervention, 1, 0, 2), /observation too short/);
});
