import { test } from "node:test";
import assert from "node:assert/strict";
import { RSSMCell } from "../../src/model/rssm.ts";
import { Action } from "../../src/env/types.ts";

test("initialState: zeros, shaped [batch, deterministicSize]", () => {
  const rssm = new RSSMCell({ deterministicSize: 6 });
  const state = rssm.initialState(3);
  assert.deepEqual(state.deterministic.shape, [3, 6]);
  assert.deepEqual(state.deterministic.arraySync(), [
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
  ]);
});

test("step: output is shaped [batch, deterministicSize]", () => {
  const rssm = new RSSMCell({ deterministicSize: 5 });
  const state = rssm.initialState(2);
  const next = rssm.step(state, [Action.Up, Action.Left]);
  assert.deepEqual(next.deterministic.shape, [2, 5]);
});

test("step: deterministic given the same prevState and actions", () => {
  const rssm = new RSSMCell({ deterministicSize: 4 });
  const state = rssm.initialState(2);
  const actions = [Action.Right, Action.Down];

  const a = rssm.step(state, actions).deterministic.arraySync();
  const b = rssm.step(state, actions).deterministic.arraySync();
  assert.deepEqual(a, b);
});

test("step: different actions from the same prevState produce different states", () => {
  const rssm = new RSSMCell({ deterministicSize: 4 });
  const state = rssm.initialState(1);

  const stay = rssm.step(state, [Action.Stay]).deterministic.arraySync();
  const up = rssm.step(state, [Action.Up]).deterministic.arraySync();
  assert.notDeepEqual(stay, up);
});

test("step: each batch row is independent — one agent's action doesn't affect another's row", () => {
  const rssm = new RSSMCell({ deterministicSize: 4 });
  const batched = rssm.step(rssm.initialState(2), [Action.Up, Action.Down]).deterministic.arraySync();

  const singleUp = rssm.step(rssm.initialState(1), [Action.Up]).deterministic.arraySync();
  const singleDown = rssm.step(rssm.initialState(1), [Action.Down]).deterministic.arraySync();

  assert.deepEqual(batched[0], singleUp[0]);
  assert.deepEqual(batched[1], singleDown[0]);
});

test("step: chaining across multiple steps keeps a stable shape and accumulates a non-zero state", () => {
  const rssm = new RSSMCell({ deterministicSize: 4 });
  let state = rssm.initialState(1);
  const actionSequence = [Action.Up, Action.Up, Action.Right, Action.Stay];

  for (const action of actionSequence) {
    state = rssm.step(state, [action]);
    assert.deepEqual(state.deterministic.shape, [1, 4]);
  }
  const final = state.deterministic.arraySync()[0];
  assert.ok(final.some((v) => v !== 0), "expected the recurrent state to move away from zero after several steps");
});

test("step: two different action sequences from the same initial state diverge", () => {
  const rssm = new RSSMCell({ deterministicSize: 4 });
  const initial = rssm.initialState(1);

  let stateA = initial;
  for (const action of [Action.Up, Action.Up, Action.Up]) {
    stateA = rssm.step(stateA, [action]);
  }

  let stateB = initial;
  for (const action of [Action.Down, Action.Down, Action.Down]) {
    stateB = rssm.step(stateB, [action]);
  }

  assert.notDeepEqual(stateA.deterministic.arraySync(), stateB.deterministic.arraySync());
});
