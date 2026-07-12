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
