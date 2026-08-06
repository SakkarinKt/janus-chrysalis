import type { CooperativeGridWorld } from "../env/gridworld.ts";
import { NUM_AGENTS } from "../env/types.ts";
import type { Action, Observation } from "../env/types.ts";
import type { Policy } from "../agent/policy.ts";
import type { Rng } from "../env/rng.ts";
import type { WorldModel } from "../model/worldModel.ts";

/**
 * "intervention": only `frozenAgentIndex` stops updating at `freezeStep`;
 * the partner keeps training. "control": both agents stop updating at
 * `freezeStep` — proposal 0001's both-frozen condition, used to confirm the
 * measurement itself doesn't drift from noise alone (kill criterion #1).
 */
export type FreezeCondition = "intervention" | "control";

export interface FreezeConfig {
  /** Env step index (matches StepResult.step) at and after which freezing applies. */
  freezeStep: number;
  condition: FreezeCondition;
  /** Required when condition is "intervention"; ignored for "control" (both agents freeze there). */
  frozenAgentIndex?: number;
}

/** Whether `agentIndex` is frozen for the transition landing on env step `step`. */
export function isFrozen(agentIndex: number, step: number, config: FreezeConfig): boolean {
  if (config.condition === "intervention" && config.frozenAgentIndex === undefined) {
    throw new Error('FreezeConfig with condition "intervention" requires frozenAgentIndex');
  }
  if (step < config.freezeStep) {
    return false;
  }
  return config.condition === "control" ? true : agentIndex === config.frozenAgentIndex;
}

export interface EpisodeStepRecord {
  step: number;
  /** Per-agent observation each policy acted on to produce `actions` (the reset observation for step 1). */
  observations: Observation[];
  /** Per-agent observation after `actions` were applied — what `policy.update()` saw as `nextObservation`. */
  nextObservations: Observation[];
  actions: Action[];
  reward: number;
  done: boolean;
  /** Per-agent frozen status that gated this step's policy.update() call, indexed like actions. */
  frozen: boolean[];
  /**
   * Per-agent world-model loss this step (`WorldModel.step()`'s return),
   * `undefined` where `worldModels` wasn't given for that agent. Populated
   * every step regardless of frozen status — a frozen agent's world model
   * still gets evaluated (just not trained), per proposal 0001's "track the
   * frozen agent's... prediction error... over the following steps" —
   * raw per-step loss, not the freeze-vs-control diff itself (that's
   * `loop/GOAL.md` priority 4's metric plumbing, not computed here). See
   * docs/explainers/0005-world-model-rollout-wiring.md.
   */
  worldModelLoss: (number | undefined)[];
}

/**
 * Runs one full episode, calling each agent's policy to act every step and,
 * unless that agent is frozen for the step per `freezeConfig`, its
 * `update()` hook afterward. Omitting `freezeConfig` means no agent is ever
 * frozen (ordinary rollout). This is the backbone-agnostic half of proposal
 * 0001's freeze intervention — it does not itself compute the
 * drift-attributable prediction-error metric, only gates which agent
 * "learns" from which transition, which is what that metric depends on.
 *
 * `worldModels`, when given (indexed like `policies`, entries optional per
 * agent — Arm A: one independent `WorldModel` per agent, no sharing),
 * advances that agent's world model every step and gates *training* it
 * (not evaluating it) by the same `frozen[i]` this step already computes
 * for `policy.update()` — see docs/explainers/0005-world-model-rollout-wiring.md
 * for why a frozen agent's world model still has to be evaluated, just not
 * trained. Reset at the start of this episode via `WorldModel.reset()`.
 */
export function runEpisode(
  env: CooperativeGridWorld,
  policies: Policy[],
  rng: Rng,
  freezeConfig?: FreezeConfig,
  worldModels?: (WorldModel | undefined)[],
): EpisodeStepRecord[] {
  if (policies.length !== NUM_AGENTS) {
    throw new Error(`Expected ${NUM_AGENTS} policies, got ${policies.length}`);
  }

  worldModels?.forEach((worldModel) => worldModel?.reset());

  const records: EpisodeStepRecord[] = [];
  let observations = env.reset().observations;
  let done = false;

  while (!done) {
    const actions = observations.map((obs, i) => policies[i].act(obs, rng)) as Action[];
    const result = env.step(actions);
    const frozen = observations.map((_, i) =>
      freezeConfig ? isFrozen(i, result.step, freezeConfig) : false,
    );

    policies.forEach((policy, i) => {
      if (!frozen[i]) {
        policy.update?.({
          observation: observations[i],
          action: actions[i],
          reward: result.reward,
          nextObservation: result.observations[i],
          done: result.done,
        });
      }
    });

    const worldModelLoss = observations.map(
      (_, i) => worldModels?.[i]?.step(actions[i], result.observations[i], rng, !frozen[i]).loss,
    );

    records.push({
      step: result.step,
      observations,
      nextObservations: result.observations,
      actions,
      reward: result.reward,
      done: result.done,
      frozen,
      worldModelLoss,
    });

    observations = result.observations;
    done = result.done;
  }

  return records;
}
