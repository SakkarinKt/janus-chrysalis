import { test } from "node:test";
import assert from "node:assert/strict";
import tf from "@tensorflow/tfjs-node";
import { WorldModel } from "../../src/model/worldModel.ts";
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
  wm.step(Action.Up, OBSERVATION, rng, true);
  const afterTrain = wm.currentState.deterministic.arraySync();
  assert.notDeepEqual(before, afterTrain, "state should change after a training step");

  wm.step(Action.Down, OBSERVATION, rng, false);
  const afterFrozen = wm.currentState.deterministic.arraySync();
  assert.notDeepEqual(afterTrain, afterFrozen, "state should still change after a frozen (eval-only) step");

  wm.dispose();
});

test("WorldModel: train=true changes at least one trainable weight; train=false leaves every weight bit-identical", () => {
  const wm = new WorldModel({ rssm: CONFIG, observationSize: OBSERVATION_SIZE, lossConfig: { freeBits: 0 } });
  const rng = new Rng(1);
  const weightsOf = () => wm.cell.trainableWeights().map((w) => Array.from(w.dataSync()));

  const beforeFrozen = weightsOf();
  wm.step(Action.Up, OBSERVATION, rng, false);
  const afterFrozen = weightsOf();
  assert.deepEqual(beforeFrozen, afterFrozen, "train=false must not move any weight");

  const beforeTrain = weightsOf();
  wm.step(Action.Up, OBSERVATION, rng, true);
  const afterTrain = weightsOf();
  assert.notDeepEqual(beforeTrain, afterTrain, "train=true must move at least one weight");

  wm.dispose();
});

test("WorldModel: reconstructionLoss + klLoss equals loss, every step", () => {
  const wm = new WorldModel({ rssm: CONFIG, observationSize: OBSERVATION_SIZE });
  const rng = new Rng(1);

  for (const train of [true, false, true]) {
    const result = wm.step(Action.Up, OBSERVATION, rng, train);
    assert.ok(
      Math.abs(result.reconstructionLoss + result.klLoss - result.loss) < 1e-4,
      `expected reconstructionLoss + klLoss ~= loss, got ${result.reconstructionLoss} + ${result.klLoss} != ${result.loss}`,
    );
  }

  wm.dispose();
});

test("WorldModel: train=true changes at least one decoder weight; train=false leaves every decoder weight bit-identical", () => {
  const wm = new WorldModel({ rssm: CONFIG, observationSize: OBSERVATION_SIZE, lossConfig: { freeBits: 0 } });
  const rng = new Rng(1);
  const decoderWeightsOf = () => wm.decoder.trainableWeights().map((w) => Array.from(w.dataSync()));

  const beforeFrozen = decoderWeightsOf();
  wm.step(Action.Up, OBSERVATION, rng, false);
  const afterFrozen = decoderWeightsOf();
  assert.deepEqual(beforeFrozen, afterFrozen, "train=false must not move any decoder weight");

  const beforeTrain = decoderWeightsOf();
  wm.step(Action.Up, OBSERVATION, rng, true);
  const afterTrain = decoderWeightsOf();
  assert.notDeepEqual(beforeTrain, afterTrain, "train=true must move at least one decoder weight");

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
    reconLosses.push(wm.step(Action.Up, OBSERVATION, rng, true).reconstructionLoss);
  }

  const firstTwenty = reconLosses.slice(0, 20).reduce((a, b) => a + b, 0) / 20;
  const lastTwenty = reconLosses.slice(-20).reduce((a, b) => a + b, 0) / 20;
  assert.ok(
    lastTwenty < firstTwenty,
    `expected mean reconstructionLoss to drop across training (first 20 avg ${firstTwenty}, last 20 avg ${lastTwenty})`,
  );

  wm.dispose();
});

test("WorldModel: reset() returns to a fresh zero-filled state and disposes the previous one", () => {
  const wm = new WorldModel({ rssm: CONFIG, observationSize: OBSERVATION_SIZE });
  const rng = new Rng(1);
  wm.step(Action.Up, OBSERVATION, rng, true);
  wm.step(Action.Down, OBSERVATION, rng, true);

  wm.reset();
  const { deterministic, stochastic } = wm.currentState;
  assert.ok(deterministic.arraySync().flat().every((v) => v === 0));
  assert.ok(stochastic.arraySync().flat().every((v) => v === 0));

  wm.dispose();
});

test("WorldModel: repeated identical-input training steps drive the combined (reconstruction + KL-balanced) loss down (freeBits: 0, isolating signal from the KL floor per test/model/losses.test.ts's convention — coarse 'does it learn' check, not a convergence guarantee)", () => {
  const wm = new WorldModel({ rssm: CONFIG, observationSize: OBSERVATION_SIZE, lossConfig: { freeBits: 0 } });
  const rng = new Rng(7);

  // 150 steps / first-20-vs-last-20: see the reconstructionLoss-specific version of this test
  // above for why this sub-increment widened the window from the original 40-step / first-5-
  // vs-last-5 check (this test predates the reconstruction term; adding it changed the
  // per-step variance enough that the narrower window stopped being reliable).
  const losses: number[] = [];
  for (let i = 0; i < 150; i++) {
    losses.push(wm.step(Action.Up, OBSERVATION, rng, true).loss);
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
  for (let i = 0; i < 3; i++) wm.step(i % 2 === 0 ? Action.Up : Action.Down, OBSERVATION, rng, true);

  const before = tf.memory().numTensors;
  for (let i = 0; i < 20; i++) {
    wm.step(i % 2 === 0 ? Action.Up : Action.Down, OBSERVATION, rng, i % 3 !== 0);
  }
  const after = tf.memory().numTensors;

  assert.equal(after - before, 0, `expected 0 net tensor growth across 20 mixed train/eval steps, got ${after - before}`);

  wm.dispose();
});
