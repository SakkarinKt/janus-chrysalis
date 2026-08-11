import type { Rng } from "../env/rng.ts";
import { Action } from "../env/types.ts";
import type { Observation } from "../env/types.ts";

const ACTION_VALUES = Object.values(Action) as Action[];

export interface Transition {
  observation: Observation;
  action: Action;
  reward: number;
  nextObservation: Observation;
  done: boolean;
}

export interface Policy {
  act(observation: Observation, rng: Rng): Action;
  /**
   * Optional learning hook. Absent for policies with nothing to learn (e.g.
   * RandomPolicy below). The freeze mechanism (src/experiment/freeze.ts)
   * gates calls to this, not act() — a frozen agent still has to act to
   * produce a trajectory, it just must not learn from it.
   */
  update?(transition: Transition): void;
}

/**
 * Uniform-random action selection. Stands in for a learned policy until one
 * exists (proposal 0001's world-model cell is still gated per
 * notes/adr-0002-js-ml-stack.md §7) — has no `update`, so freezing it is a
 * no-op today. Its purpose here is to let the freeze mechanism and rollout
 * wiring be built and tested against something that actually acts.
 */
export class RandomPolicy implements Policy {
  act(_observation: Observation, rng: Rng): Action {
    return ACTION_VALUES[rng.nextInt(ACTION_VALUES.length)];
  }
}

export interface QLearningConfig {
  /**
   * TD learning rate. Default `0.1` — a common small-tabular-MDP placeholder,
   * not tuned against this environment; see
   * docs/explainers/0008-tabular-q-learning-policy.md.
   */
  alpha?: number;
  /** Discount factor. Default `0.95`, same placeholder status as `alpha`. */
  gamma?: number;
  /** Epsilon-greedy exploration rate. Default `0.1`, same placeholder status. */
  epsilon?: number;
}

const DEFAULT_Q_LEARNING_CONFIG: Required<QLearningConfig> = {
  alpha: 0.1,
  gamma: 0.95,
  epsilon: 0.1,
};

function actionAt(index: number): Action {
  const action = ACTION_VALUES[index];
  if (action === undefined) {
    throw new Error(`QLearningPolicy: action index ${index} out of range`);
  }
  return action;
}

function qValueAt(row: number[], index: number): number {
  const value = row[index];
  if (value === undefined) {
    throw new Error(`QLearningPolicy: q-value index ${index} out of range`);
  }
  return value;
}

/** The highest-valued (index, value) pair in `row`, ties broken by lowest index. */
function argmaxRow(row: number[]): { index: number; value: number } {
  let bestIndex = 0;
  let bestValue = qValueAt(row, 0);
  for (let i = 1; i < row.length; i++) {
    const value = qValueAt(row, i);
    if (value > bestValue) {
      bestValue = value;
      bestIndex = i;
    }
  }
  return { index: bestIndex, value: bestValue };
}

/**
 * Tabular Q-learning over the discrete action space (`loop/GOAL.md` priority
 * 4's trainable-policy prerequisite — see
 * docs/explainers/0008-tabular-q-learning-policy.md for discretization,
 * tie-breaking, and freeze-interaction rationale). Discretizes `Observation`
 * by using its own values as a map key, exact rather than binned for
 * `CooperativeGridWorld`'s specific encoding (see the explainer).
 */
export class QLearningPolicy implements Policy {
  private readonly config: Required<QLearningConfig>;
  private readonly qTable = new Map<string, number[]>();

  constructor(config: QLearningConfig = {}) {
    this.config = { ...DEFAULT_Q_LEARNING_CONFIG, ...config };
  }

  private qRow(observation: Observation): number[] {
    const key = observation.join(",");
    let row = this.qTable.get(key);
    if (row === undefined) {
      row = new Array(ACTION_VALUES.length).fill(0);
      this.qTable.set(key, row);
    }
    return row;
  }

  act(observation: Observation, rng: Rng): Action {
    if (rng.next() < this.config.epsilon) {
      return actionAt(rng.nextInt(ACTION_VALUES.length));
    }
    return actionAt(argmaxRow(this.qRow(observation)).index);
  }

  update(transition: Transition): void {
    const row = this.qRow(transition.observation);
    const actionIndex = ACTION_VALUES.indexOf(transition.action);
    const current = qValueAt(row, actionIndex);
    // `transition.done` marks a time-limit truncation (CooperativeGridWorld.step:
    // `done = currentStep >= horizon`), not a terminal/absorbing state — the env
    // has none. Dropping the bootstrap on `done` therefore biased the last
    // transition of every episode; always bootstrap instead. See PR #43 review
    // (@SakkarinKt, 2026-08-11) and docs/explainers/0008.
    const nextMax = argmaxRow(this.qRow(transition.nextObservation)).value;
    row[actionIndex] =
      current + this.config.alpha * (transition.reward + this.config.gamma * nextMax - current);
  }
}
