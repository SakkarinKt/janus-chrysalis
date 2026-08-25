import { Rng } from "./rng.ts";
import { Action, DEFAULT_CONFIG, NUM_AGENTS } from "./types.ts";
import type { GridWorldConfig, Observation, Position, ResetResult, StepResult } from "./types.ts";

/**
 * Laptop-scale 2-agent cooperative-navigation grid world (proposal 0001's
 * Arm-A milestone environment). No sharing/comms machinery lives here —
 * that is Arm B/C/D territory and stays out of scope for this scaffold.
 */
export class CooperativeGridWorld {
  readonly config: GridWorldConfig;

  private rng: Rng;
  private agentPositions: Position[] = [];
  private landmarkPositions: Position[] = [];
  private currentStep = 0;
  /**
   * Independent override for the partner's `relativeEntry` gate, distinct
   * from `config.viewRadius` (which alone still gates every landmark's
   * `relativeEntry` call). `undefined` means "no override" — partner
   * visibility falls back to `config.viewRadius`, matching every version of
   * this class before `setPartnerViewRadius()` existed. Needed because
   * `setViewRadius()` moved both gates together, confounding a partner-only
   * visibility manipulation with a landmark-visibility change (proposal
   * 0001, PR #50 review, 2026-08-23: "`viewRadius` gates landmarks too ...
   * the sweep axis moves partner visibility and landmark observability
   * together").
   */
  private partnerViewRadiusOverride: number | undefined = undefined;

  constructor(config: Partial<GridWorldConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
    this.rng = new Rng(this.config.seed);
  }

  get observationLength(): number {
    return 2 + this.config.numLandmarks * 3 + 3;
  }

  reset(): ResetResult {
    this.rng = new Rng(this.config.seed);
    this.currentStep = 0;
    this.partnerViewRadiusOverride = undefined;
    const occupied = new Set<string>();
    this.agentPositions = Array.from({ length: NUM_AGENTS }, () => this.spawn(occupied));
    this.landmarkPositions = Array.from(
      { length: this.config.numLandmarks },
      () => this.spawn(occupied),
    );
    return { observations: this.observe(), step: this.currentStep };
  }

  step(actions: Action[]): StepResult {
    if (actions.length !== NUM_AGENTS) {
      throw new Error(`Expected ${NUM_AGENTS} actions, got ${actions.length}`);
    }
    if (this.currentStep >= this.config.horizon) {
      throw new Error("Episode already finished; call reset() before stepping further.");
    }

    this.agentPositions = this.agentPositions.map((pos, i) => this.applyAction(pos, actions[i]));
    this.currentStep += 1;

    const reward = this.computeReward();
    const done = this.currentStep >= this.config.horizon;
    return { observations: this.observe(), reward, done, step: this.currentStep };
  }

  /**
   * Mid-episode override for `config.viewRadius`, taking effect on every
   * subsequent `step()`'s `relativeEntry` gate — for a post-freeze-only
   * `viewRadius` manipulation (proposal 0001, PR #49 review follow-up,
   * 2026-08-23) that keeps pre-freeze training identical across a sweep
   * while varying only the post-freeze visibility window. Does not reset,
   * or otherwise touch, agent/landmark positions or `currentStep`.
   */
  setViewRadius(viewRadius: number): void {
    this.config.viewRadius = viewRadius;
  }

  /**
   * Mid-episode override for the partner's visibility gate only — every
   * landmark's `relativeEntry` call keeps using `config.viewRadius`
   * unaffected. Same call-once-before-`env.step()` usage as `setViewRadius`
   * (see `runEpisode`'s `postFreezeEnvMutation`, src/experiment/freeze.ts),
   * but decouples the partner-visibility manipulation from landmark
   * observability (proposal 0001, PR #50 review follow-up, 2026-08-24).
   * `reset()` clears the override (PR #51 review, 2026-08-25): it does not
   * survive across episodes on a reused instance, unlike `setViewRadius`'s
   * mutation of `config.viewRadius`, which does.
   */
  setPartnerViewRadius(viewRadius: number): void {
    this.partnerViewRadiusOverride = viewRadius;
  }

  /** Effective radius gating the partner's `relativeEntry` call — `config.viewRadius` unless
   * `setPartnerViewRadius()` has overridden it for this instance. */
  get partnerViewRadius(): number {
    return this.partnerViewRadiusOverride ?? this.config.viewRadius;
  }

  /** Defensive copy — for logging/replay/tests, not for mutating env state. */
  getAgentPositions(): Position[] {
    return this.agentPositions.map((p) => ({ ...p }));
  }

  /** Defensive copy — for logging/replay/tests, not for mutating env state. */
  getLandmarkPositions(): Position[] {
    return this.landmarkPositions.map((p) => ({ ...p }));
  }

  private spawn(occupied: Set<string>): Position {
    let pos: Position;
    let key: string;
    do {
      pos = { x: this.rng.nextInt(this.config.gridSize), y: this.rng.nextInt(this.config.gridSize) };
      key = `${pos.x},${pos.y}`;
    } while (occupied.has(key));
    occupied.add(key);
    return pos;
  }

  private applyAction(pos: Position, action: Action): Position {
    const { gridSize } = this.config;
    const clamp = (v: number) => Math.max(0, Math.min(gridSize - 1, v));
    switch (action) {
      case Action.Up:
        return { x: pos.x, y: clamp(pos.y - 1) };
      case Action.Down:
        return { x: pos.x, y: clamp(pos.y + 1) };
      case Action.Left:
        return { x: clamp(pos.x - 1), y: pos.y };
      case Action.Right:
        return { x: clamp(pos.x + 1), y: pos.y };
      case Action.Stay:
      default:
        return { x: pos.x, y: pos.y };
    }
  }

  private distance(a: Position, b: Position): number {
    return Math.abs(a.x - b.x) + Math.abs(a.y - b.y);
  }

  private computeReward(): number {
    const { gridSize, collisionPenalty } = this.config;
    let coverage = 0;
    for (const landmark of this.landmarkPositions) {
      coverage += Math.min(...this.agentPositions.map((p) => this.distance(p, landmark)));
    }
    let reward = -coverage / gridSize;
    const [a0, a1] = this.agentPositions;
    if (a0.x === a1.x && a0.y === a1.y) {
      reward -= collisionPenalty;
    }
    return reward;
  }

  private observe(): Observation[] {
    return this.agentPositions.map((self, i) => {
      const vec: number[] = [self.x / this.config.gridSize, self.y / this.config.gridSize];
      for (const landmark of this.landmarkPositions) {
        vec.push(...this.relativeEntry(self, landmark, this.config.viewRadius));
      }
      const other = this.agentPositions[1 - i];
      vec.push(...this.relativeEntry(self, other, this.partnerViewRadius));
      return vec;
    });
  }

  private relativeEntry(self: Position, other: Position, viewRadius: number): [number, number, number] {
    if (this.distance(self, other) > viewRadius) {
      return [0, 0, 0];
    }
    const { gridSize } = this.config;
    return [1, (other.x - self.x) / gridSize, (other.y - self.y) / gridSize];
  }
}
