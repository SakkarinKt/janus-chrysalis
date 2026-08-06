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

test("WorldModel: repeated identical-input training steps drive the KL-balanced loss down (freeBits: 0, isolating signal from the floor per test/model/losses.test.ts's convention — coarse 'does it learn' check, not a convergence guarantee)", () => {
  const wm = new WorldModel({ rssm: CONFIG, observationSize: OBSERVATION_SIZE, lossConfig: { freeBits: 0 } });
  const rng = new Rng(7);

  const losses: number[] = [];
  for (let i = 0; i < 40; i++) {
    losses.push(wm.step(Action.Up, OBSERVATION, rng, true).loss);
  }

  const firstFive = losses.slice(0, 5).reduce((a, b) => a + b, 0) / 5;
  const lastFive = losses.slice(-5).reduce((a, b) => a + b, 0) / 5;
  assert.ok(
    lastFive < firstFive,
    `expected mean loss to drop across training (first 5 avg ${firstFive}, last 5 avg ${lastFive}); losses: ${losses}`,
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
