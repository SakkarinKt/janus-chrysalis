export const NUM_AGENTS = 2;

// A const object rather than `enum`: Node's type-stripping only erases
// syntax, and TS `enum` compiles to runtime code, so it isn't supported
// without a build step (see package.json's zero-dependency test setup).
export const Action = {
  Stay: 0,
  Up: 1,
  Down: 2,
  Left: 3,
  Right: 4,
} as const;
export type Action = (typeof Action)[keyof typeof Action];

export interface Position {
  x: number;
  y: number;
}

export interface GridWorldConfig {
  /** Grid is gridSize x gridSize, origin at (0, 0). */
  gridSize: number;
  /** Chebyshev-distance radius within which another entity is observable. */
  viewRadius: number;
  /** Fixed episode length in steps; env is done after this many steps. */
  horizon: number;
  /** Number of landmarks agents must cover (cooperative-navigation style). */
  numLandmarks: number;
  /** Per-step penalty applied to the shared reward when both agents occupy the same cell. */
  collisionPenalty: number;
  seed: number;
}

export const DEFAULT_CONFIG: GridWorldConfig = {
  gridSize: 8,
  viewRadius: 2,
  horizon: 75,
  numLandmarks: 2,
  collisionPenalty: 0.5,
  seed: 0,
};

/**
 * Flat per-agent observation vector, fixed length for a given config:
 * [selfX, selfY,
 *  landmark_0_visible, landmark_0_dx, landmark_0_dy, ... (per landmark),
 *  otherAgent_visible, otherAgent_dx, otherAgent_dy]
 * Positions/offsets are normalized by gridSize; entities outside viewRadius
 * are masked (visible=0, dx=dy=0) rather than omitted, so vector length is
 * constant across steps and agents.
 */
export type Observation = number[];

export interface StepResult {
  observations: Observation[];
  /** Shared (cooperative) scalar reward, identical for both agents. */
  reward: number;
  done: boolean;
  step: number;
}

export interface ResetResult {
  observations: Observation[];
  step: number;
}
