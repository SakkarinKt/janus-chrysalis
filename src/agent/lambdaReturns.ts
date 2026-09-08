/**
 * Inputs to a λ-return computation over one trajectory segment. All arrays
 * are aligned by step index and must be the same length as `rewards`;
 * `values`/`continues` deliberately do **not** carry an extra
 * `rewards.length + 1`'th entry for the post-segment state — that value
 * lives in `bootstrapValue` instead, so a caller can't accidentally pass a
 * same-length array and have its last element silently misread as the
 * bootstrap (see `docs/explainers/0012-replay-buffer-lambda-returns-spec.md`
 * for why this indexing was chosen over the alternative).
 */
export interface LambdaReturnsInput {
  rewards: number[];
  /** Value-function estimate for the state *before* each step's transition. */
  values: number[];
  /**
   * Probability the episode continues past each step, in `[0, 1]` — not a
   * hard boolean `done` flag, so a future actor-critic can feed
   * `ContinueHead.predict()`'s sigmoid output (`src/model/continueHead.ts`)
   * directly when bootstrapping through imagined rollouts. A caller working
   * from ground-truth `done`s (e.g. `EpisodeStepRecord.done`,
   * `src/experiment/freeze.ts`) passes the degenerate hard case, `done ? 0 :
   * 1` per step — the same target-construction convention `WorldModel.step()`
   * already uses for its continue-head training target
   * (`src/model/worldModel.ts:184`, `docs/explainers/0011`).
   */
  continues: number[];
  /** Value estimate for the state immediately after the segment's last recorded step. */
  bootstrapValue: number;
  gamma: number;
  /** Interpolation weight between one-step TD (`lambda: 0`) and Monte-Carlo (`lambda: 1`). */
  lambda: number;
}

/**
 * SPEC ONLY — reserved for the human's Gate G2 role-flip implementation
 * (`loop/GOAL.md` priority 6; `PLAN.html`'s Gate G2 line). Throws
 * unconditionally; see
 * `docs/explainers/0012-replay-buffer-lambda-returns-spec.md` for why this
 * has no caller yet (no value-function-based policy exists in
 * `src/agent/` today) and for the boundary properties
 * `test/agent/lambdaReturns.test.ts`'s `{ todo }` skeletons pin down.
 *
 * Expected contract once implemented: returns one λ-return per input step
 * (same length as `input.rewards`), via the standard backward recursion
 *
 *   R_t = rewards[t] + gamma * continues[t] * ((1 - lambda) * V_{t+1} + lambda * R_{t+1})
 *
 * where `V_{t+1}` is `values[t + 1]` for `t < rewards.length - 1` and
 * `bootstrapValue` at the last step, and `R_{t+1}` is similarly seeded from
 * `bootstrapValue` one step past the end. Should throw on a
 * `rewards`/`values`/`continues` length mismatch rather than truncate or
 * pad — matches this codebase's existing convention for misaligned inputs
 * (e.g. `driftAttributableError`, `src/experiment/metrics.ts`).
 */
export function computeLambdaReturns(input: LambdaReturnsInput): number[] {
  void input;
  throw new Error(
    "computeLambdaReturns is a Gate G2 spec-only skeleton (loop/GOAL.md priority 6) — " +
      "implementation is reserved for the human; see docs/explainers/0012.",
  );
}
