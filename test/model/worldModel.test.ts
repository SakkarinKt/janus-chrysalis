import { test } from "node:test";
import assert from "node:assert/strict";
import tf from "@tensorflow/tfjs-node";
import { WorldModel, WorldModelNaNError } from "../../src/model/worldModel.ts";
import { continueLoss } from "../../src/model/losses.ts";
import { Action } from "../../src/env/types.ts";
import { Rng } from "../../src/env/rng.ts";

const CONFIG = { deterministicSize: 4, latentCategoricals: 3, latentClasses: 5 };
const OBSERVATION_SIZE = 6;
const OBSERVATION = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6];

test("WorldModel: construction builds every layer without throwing, trainableWeights() non-empty", () => {
  const wm = new WorldModel({ rssm: CONFIG, observationSize: OBSERVATION_SIZE });
  assert.ok(wm.cell.trainableWeights().length > 0);
  wm.dispose();
});

test("WorldModel: seed reproduces identical initial weights (cell + decoder + continueHead) across independent instances; different seeds diverge — the fix for PR #45's review (gate (b) isn't readable until init is seeded or paired)", () => {
  const weightsOf = (wm: WorldModel) => [
    ...wm.cell.trainableWeights().map((w) => Array.from(w.dataSync())),
    ...wm.decoder.trainableWeights().map((w) => Array.from(w.dataSync())),
    ...wm.continueHead.trainableWeights().map((w) => Array.from(w.dataSync())),
  ];

  const a = new WorldModel({ rssm: CONFIG, observationSize: OBSERVATION_SIZE, seed: 1001 });
  const b = new WorldModel({ rssm: CONFIG, observationSize: OBSERVATION_SIZE, seed: 1001 });
  assert.deepEqual(weightsOf(a), weightsOf(b), "same seed must draw identical initial weights");
  a.dispose();
  b.dispose();

  const c = new WorldModel({ rssm: CONFIG, observationSize: OBSERVATION_SIZE, seed: 1001 });
  const d = new WorldModel({ rssm: CONFIG, observationSize: OBSERVATION_SIZE, seed: 1002 });
  assert.notDeepEqual(weightsOf(c), weightsOf(d), "different seeds must draw different initial weights");
  c.dispose();
  d.dispose();
});

test("WorldModel: initial state is zero, matching RSSMCell.initialState(1)", () => {
  const wm = new WorldModel({ rssm: CONFIG, observationSize: OBSERVATION_SIZE });
  const { deterministic, stochastic } = wm.currentState;
  assert.deepEqual(deterministic.shape, [1, CONFIG.deterministicSize]);
  assert.deepEqual(stochastic.shape, [1, CONFIG.latentCategoricals * CONFIG.latentClasses]);
  assert.ok(deterministic.arraySync().flat().every((v) => v === 0));
  assert.ok(stochastic.arraySync().flat().every((v) => v === 0));
  wm.dispose();
});

test("WorldModel: state (deterministic + stochastic) changes every step, regardless of train", () => {
  const wm = new WorldModel({ rssm: CONFIG, observationSize: OBSERVATION_SIZE });
  const rng = new Rng(1);

  const before = wm.currentState.deterministic.arraySync();
  wm.step(Action.Up, OBSERVATION, rng, true, false);
  const afterTrain = wm.currentState.deterministic.arraySync();
  assert.notDeepEqual(before, afterTrain, "state should change after a training step");

  wm.step(Action.Down, OBSERVATION, rng, false, false);
  const afterFrozen = wm.currentState.deterministic.arraySync();
  assert.notDeepEqual(afterTrain, afterFrozen, "state should still change after a frozen (eval-only) step");

  wm.dispose();
});

test("WorldModel: train=true changes at least one trainable weight; train=false leaves every weight bit-identical", () => {
  const wm = new WorldModel({ rssm: CONFIG, observationSize: OBSERVATION_SIZE, lossConfig: { freeBits: 0 } });
  const rng = new Rng(1);
  const weightsOf = () => wm.cell.trainableWeights().map((w) => Array.from(w.dataSync()));

  const beforeFrozen = weightsOf();
  wm.step(Action.Up, OBSERVATION, rng, false, false);
  const afterFrozen = weightsOf();
  assert.deepEqual(beforeFrozen, afterFrozen, "train=false must not move any weight");

  const beforeTrain = weightsOf();
  wm.step(Action.Up, OBSERVATION, rng, true, false);
  const afterTrain = weightsOf();
  assert.notDeepEqual(beforeTrain, afterTrain, "train=true must move at least one weight");

  wm.dispose();
});

test("WorldModel: reconstructionLoss + klLoss + continueLoss equals loss, every step", () => {
  const wm = new WorldModel({ rssm: CONFIG, observationSize: OBSERVATION_SIZE });
  const rng = new Rng(1);

  for (const train of [true, false, true]) {
    const result = wm.step(Action.Up, OBSERVATION, rng, train, false);
    assert.ok(
      Math.abs(result.reconstructionLoss + result.klLoss + result.continueLoss - result.loss) < 1e-4,
      `expected reconstructionLoss + klLoss + continueLoss ~= loss, got ${result.reconstructionLoss} + ` +
        `${result.klLoss} + ${result.continueLoss} != ${result.loss}`,
    );
  }

  wm.dispose();
});

test("WorldModel: train=true changes at least one decoder weight; train=false leaves every decoder weight bit-identical", () => {
  const wm = new WorldModel({ rssm: CONFIG, observationSize: OBSERVATION_SIZE, lossConfig: { freeBits: 0 } });
  const rng = new Rng(1);
  const decoderWeightsOf = () => wm.decoder.trainableWeights().map((w) => Array.from(w.dataSync()));

  const beforeFrozen = decoderWeightsOf();
  wm.step(Action.Up, OBSERVATION, rng, false, false);
  const afterFrozen = decoderWeightsOf();
  assert.deepEqual(beforeFrozen, afterFrozen, "train=false must not move any decoder weight");

  const beforeTrain = decoderWeightsOf();
  wm.step(Action.Up, OBSERVATION, rng, true, false);
  const afterTrain = decoderWeightsOf();
  assert.notDeepEqual(beforeTrain, afterTrain, "train=true must move at least one decoder weight");

  wm.dispose();
});

test("WorldModel: train=true changes at least one continueHead weight; train=false leaves every continueHead weight bit-identical (docs/explainers/0011)", () => {
  const wm = new WorldModel({ rssm: CONFIG, observationSize: OBSERVATION_SIZE, lossConfig: { freeBits: 0 } });
  const rng = new Rng(1);
  const continueHeadWeightsOf = () => wm.continueHead.trainableWeights().map((w) => Array.from(w.dataSync()));

  const beforeFrozen = continueHeadWeightsOf();
  wm.step(Action.Up, OBSERVATION, rng, false, false);
  const afterFrozen = continueHeadWeightsOf();
  assert.deepEqual(beforeFrozen, afterFrozen, "train=false must not move any continueHead weight");

  const beforeTrain = continueHeadWeightsOf();
  wm.step(Action.Up, OBSERVATION, rng, true, false);
  const afterTrain = continueHeadWeightsOf();
  assert.notDeepEqual(beforeTrain, afterTrain, "train=true must move at least one continueHead weight");

  wm.dispose();
});

test("WorldModel: repeated identical-input training steps drive reconstructionLoss down specifically (freeBits: 0 isolates it from the KL floor, same convention as the combined-loss test below)", () => {
  const wm = new WorldModel({ rssm: CONFIG, observationSize: OBSERVATION_SIZE, lossConfig: { freeBits: 0 } });
  const rng = new Rng(11);

  // 150 steps / first-20-vs-last-20 rather than the 40-step / first-5-vs-last-5 window the
  // KL-only version of this coarse check used (before this sub-increment added a second,
  // independently-initialized loss term sharing the same optimizer step): empirically, at
  // 40 steps with a 5-sample window, some seeds' last-5 average lands above their first-5
  // average purely from the per-step Gumbel-sample variance a short window doesn't average
  // out, even while the underlying trend is a clear decrease (checked directly against
  // several seeds, not asserted from theory alone). Wider windows over a longer run made the
  // decrease reliable across every seed tried.
  const reconLosses: number[] = [];
  for (let i = 0; i < 150; i++) {
    reconLosses.push(wm.step(Action.Up, OBSERVATION, rng, true, false).reconstructionLoss);
  }

  const firstTwenty = reconLosses.slice(0, 20).reduce((a, b) => a + b, 0) / 20;
  const lastTwenty = reconLosses.slice(-20).reduce((a, b) => a + b, 0) / 20;
  assert.ok(
    lastTwenty < firstTwenty,
    `expected mean reconstructionLoss to drop across training (first 20 avg ${firstTwenty}, last 20 avg ${lastTwenty})`,
  );

  wm.dispose();
});

test("WorldModel: repeated identical-input, identical-done training steps drive continueLoss down specifically, in both done directions (docs/explainers/0011 — the target is a constant 0 or 1 per direction, so this is a shape/gradient-wiring check, not a claim about learning a real termination signal)", () => {
  for (const done of [false, true]) {
    const wm = new WorldModel({ rssm: CONFIG, observationSize: OBSERVATION_SIZE, lossConfig: { freeBits: 0 } });
    const rng = new Rng(11);

    const continueLosses: number[] = [];
    for (let i = 0; i < 150; i++) {
      continueLosses.push(wm.step(Action.Up, OBSERVATION, rng, true, done).continueLoss);
    }

    const firstTwenty = continueLosses.slice(0, 20).reduce((a, b) => a + b, 0) / 20;
    const lastTwenty = continueLosses.slice(-20).reduce((a, b) => a + b, 0) / 20;
    assert.ok(
      lastTwenty < firstTwenty,
      `expected mean continueLoss to drop across training with done=${done} (first 20 avg ${firstTwenty}, ` +
        `last 20 avg ${lastTwenty})`,
    );

    wm.dispose();
  }
});

test(
  "WorldModel: continueTargetTensor is exactly done ? 0 : 1 (worldModel.ts:184's polarity), pinned " +
    "directly via a stubbed continueHead logit rather than the training-convergence proxy the test " +
    "above uses — that proxy can't distinguish the documented polarity from its reverse, since it " +
    "only checks that loss decreases toward whatever constant target is set, in both directions " +
    "symmetrically. Carried forward from PR #63's review through the 2026-09-03/04 stand-ups " +
    "(loop/GOAL.md priority 5).",
  () => {
    for (const [done, expectedTarget] of [[false, 1], [true, 0]] as const) {
      const wm = new WorldModel({ rssm: CONFIG, observationSize: OBSERVATION_SIZE });
      const rng = new Rng(1);
      const stubLogit = tf.tensor2d([[10]]);
      wm.continueHead.predict = () => stubLogit;

      const result = wm.step(Action.Up, OBSERVATION, rng, false, done);
      const expectedLoss = tf.tidy(
        () => continueLoss(stubLogit, tf.tensor2d([[expectedTarget]])).arraySync() as number,
      );
      assert.ok(
        Math.abs(result.continueLoss - expectedLoss) < 1e-4,
        `done=${done}: expected continueLoss ${expectedLoss} (target ${expectedTarget} against the ` +
          `stubbed logit), got ${result.continueLoss} — target polarity does not match ` +
          "docs/explainers/0011's done ? 0 : 1 convention",
      );

      stubLogit.dispose();
      wm.dispose();
    }
  },
);

test("WorldModel: reset() returns to a fresh zero-filled state and disposes the previous one", () => {
  const wm = new WorldModel({ rssm: CONFIG, observationSize: OBSERVATION_SIZE });
  const rng = new Rng(1);
  wm.step(Action.Up, OBSERVATION, rng, true, false);
  wm.step(Action.Down, OBSERVATION, rng, true, false);

  wm.reset();
  const { deterministic, stochastic } = wm.currentState;
  assert.ok(deterministic.arraySync().flat().every((v) => v === 0));
  assert.ok(stochastic.arraySync().flat().every((v) => v === 0));

  wm.dispose();
});

test("WorldModel: repeated identical-input training steps drive the combined (reconstruction + KL-balanced + continue) loss down (freeBits: 0, isolating signal from the KL floor per test/model/losses.test.ts's convention — coarse 'does it learn' check, not a convergence guarantee)", () => {
  const wm = new WorldModel({ rssm: CONFIG, observationSize: OBSERVATION_SIZE, lossConfig: { freeBits: 0 } });
  const rng = new Rng(7);

  // 150 steps / first-20-vs-last-20: see the reconstructionLoss-specific version of this test
  // above for why this sub-increment widened the window from the original 40-step / first-5-
  // vs-last-5 check (this test predates the reconstruction term; adding it changed the
  // per-step variance enough that the narrower window stopped being reliable).
  const losses: number[] = [];
  for (let i = 0; i < 150; i++) {
    losses.push(wm.step(Action.Up, OBSERVATION, rng, true, false).loss);
  }

  const firstTwenty = losses.slice(0, 20).reduce((a, b) => a + b, 0) / 20;
  const lastTwenty = losses.slice(-20).reduce((a, b) => a + b, 0) / 20;
  assert.ok(
    lastTwenty < firstTwenty,
    `expected mean loss to drop across training (first 20 avg ${firstTwenty}, last 20 avg ${lastTwenty})`,
  );

  wm.dispose();
});

test("WorldModel: a warm (post-first-step) run of mixed train/eval steps leaves 0 net tensors beyond the two persisted state tensors (tensor-leak check)", () => {
  const wm = new WorldModel({ rssm: CONFIG, observationSize: OBSERVATION_SIZE });
  const rng = new Rng(3);

  // Warm up first: Adam lazily creates one moment + one velocity variable
  // per trainable weight on its *first* applyGradients call — a one-time,
  // intentionally-persistent allocation, not a leak. Measuring only after
  // that warm-up isolates genuine per-step growth.
  for (let i = 0; i < 3; i++) wm.step(i % 2 === 0 ? Action.Up : Action.Down, OBSERVATION, rng, true, false);

  const before = tf.memory().numTensors;
  for (let i = 0; i < 20; i++) {
    wm.step(i % 2 === 0 ? Action.Up : Action.Down, OBSERVATION, rng, i % 3 !== 0, false);
  }
  const after = tf.memory().numTensors;

  assert.equal(after - before, 0, `expected 0 net tensor growth across 20 mixed train/eval steps, got ${after - before}`);

  wm.dispose();
});

test("WorldModel: a throw from forward() (wrong-length observation, shape mismatch inside cell.posterior) leaves this.state exactly as it was — still usable, not disposed (PR #40 review follow-up 1)", () => {
  const wm = new WorldModel({ rssm: CONFIG, observationSize: OBSERVATION_SIZE });
  const rng = new Rng(1);
  wm.step(Action.Up, OBSERVATION, rng, true, false);

  const before = wm.currentState.deterministic.arraySync();
  const wrongLengthObservation = OBSERVATION.slice(0, 3);
  assert.throws(() => wm.step(Action.Up, wrongLengthObservation, rng, true, false));

  // Previously: the finally block disposed prevState's tensors
  // unconditionally, and since this.state is only reassigned on success,
  // this.state still pointed at prevState — so it read back as disposed.
  const after = wm.currentState.deterministic.arraySync();
  assert.deepEqual(after, before, "state must be unchanged after a caught throw");

  // A further step must still work normally — the model wasn't poisoned.
  wm.step(Action.Down, OBSERVATION, rng, true, false);

  wm.dispose();
});

test("WorldModel: a throw from forward() with train=false leaks no tensors (tensor-leak check across a caught throw, PR #40 review follow-up 2 — the train=true path also leaks nothing of ours, but tf.variableGrads' own internal tidy drops ~66 intermediates on its own error path regardless of branch, a pre-existing library-level cost outside step()'s control)", () => {
  const wm = new WorldModel({ rssm: CONFIG, observationSize: OBSERVATION_SIZE });
  const rng = new Rng(1);
  wm.step(Action.Up, OBSERVATION, rng, false, false);

  const before = tf.memory().numTensors;
  const wrongLengthObservation = OBSERVATION.slice(0, 3);
  assert.throws(() => wm.step(Action.Up, wrongLengthObservation, rng, false, false));
  const after = tf.memory().numTensors;

  assert.equal(after - before, 0, `expected 0 net tensor growth across a caught throw, got ${after - before}`);

  wm.dispose();
});

test("WorldModel: a throw from decoder.decode() — after the tf.keep() calls above, unlike every other throw case above — leaves this.state unchanged and leaks no tensors (PR #41 review non-blocking note 1: nextDeterministic?.dispose()/nextStochastic?.dispose() at the catch block was, until this test, exercised by zero tests)", () => {
  const wm = new WorldModel({ rssm: CONFIG, observationSize: OBSERVATION_SIZE });
  const rng = new Rng(1);
  wm.step(Action.Up, OBSERVATION, rng, true, false);

  wm.decoder.decode = (): tf.Tensor2D => {
    throw new Error("stubbed decode failure");
  };

  const before = wm.currentState.deterministic.arraySync();
  const beforeTensors = tf.memory().numTensors;

  // train=false, matching the other leak checks above, so the pre-existing
  // tf.variableGrads-internal ~66-tensor error-path cost (unrelated to this
  // fix, see the train=true throw test above) doesn't obscure the count.
  assert.throws(() => wm.step(Action.Down, OBSERVATION, rng, false, false), /stubbed decode failure/);

  const afterTensors = tf.memory().numTensors;
  assert.equal(
    afterTensors - beforeTensors,
    0,
    `expected 0 net tensor growth across a decode()-throw, got ${afterTensors - beforeTensors}`,
  );

  const after = wm.currentState.deterministic.arraySync();
  assert.deepEqual(after, before, "state must be unchanged after a caught throw from decode()");

  wm.dispose();
});

test("WorldModel: a NaN-valued observation with train=false throws WorldModelNaNError, leaves recurrent state and weights untouched and leaks no tensors, and a subsequent clean step recovers fully — the NaN-halt invariant (loop/GOAL.md priority 5, docs/explainers/0009)", () => {
  const wm = new WorldModel({ rssm: CONFIG, observationSize: OBSERVATION_SIZE });
  const rng = new Rng(1);
  wm.step(Action.Up, OBSERVATION, rng, true, false);
  const nanObservation = [NaN, ...OBSERVATION.slice(1)];

  const beforeState = wm.currentState.deterministic.arraySync();
  const beforeWeights = wm.cell.trainableWeights().map((w) => Array.from(w.dataSync()));
  const beforeTensors = tf.memory().numTensors;

  assert.throws(
    () => wm.step(Action.Up, nanObservation, rng, false, false),
    (err: unknown) =>
      err instanceof WorldModelNaNError && Number.isNaN(err.loss) && Number.isNaN(err.reconstructionLoss),
  );

  const afterState = wm.currentState.deterministic.arraySync();
  assert.deepEqual(afterState, beforeState, "recurrent state must be unchanged after a NaN-halt throw");
  const afterWeights = wm.cell.trainableWeights().map((w) => Array.from(w.dataSync()));
  assert.deepEqual(afterWeights, beforeWeights, "train=false must not move any weight, even on a NaN-halt throw");
  const afterTensors = tf.memory().numTensors;
  assert.equal(
    afterTensors - beforeTensors,
    0,
    `expected 0 net tensor growth across a NaN-halt throw, got ${afterTensors - beforeTensors}`,
  );

  const result = wm.step(Action.Down, OBSERVATION, rng, true, false);
  assert.ok(Number.isFinite(result.loss), "model must recover fully after a caught train=false NaN-halt throw");

  wm.dispose();
});

test("WorldModel: a NaN-valued observation with train=true throws WorldModelNaNError and leaves recurrent state unchanged and leak-free, but — documented limitation, docs/explainers/0009 — cannot undo the optimizer step forward() already applied before the throw, so weights are left NaN-corrupted", () => {
  const wm = new WorldModel({ rssm: CONFIG, observationSize: OBSERVATION_SIZE });
  const rng = new Rng(1);
  wm.step(Action.Up, OBSERVATION, rng, true, false);
  const nanObservation = [NaN, ...OBSERVATION.slice(1)];

  const beforeState = wm.currentState.deterministic.arraySync();
  const beforeTensors = tf.memory().numTensors;

  assert.throws(
    () => wm.step(Action.Up, nanObservation, rng, true, false),
    (err: unknown) =>
      err instanceof WorldModelNaNError && Number.isNaN(err.loss) && Number.isNaN(err.reconstructionLoss),
  );

  const afterState = wm.currentState.deterministic.arraySync();
  assert.deepEqual(afterState, beforeState, "recurrent state must be unchanged after a NaN-halt throw");
  const afterTensors = tf.memory().numTensors;
  assert.equal(
    afterTensors - beforeTensors,
    0,
    `expected 0 net tensor growth across a NaN-halt throw, got ${afterTensors - beforeTensors}`,
  );

  const afterWeights = wm.cell.trainableWeights().map((w) => Array.from(w.dataSync()));
  assert.ok(
    afterWeights.some((w) => w.some((v) => Number.isNaN(v))),
    "documented limitation: train=true's optimizer step already ran on the non-finite gradient before " +
      "the throw, so weights are left NaN — the halt stops further compounding, it doesn't undo this step",
  );

  wm.dispose();
});
