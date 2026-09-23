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
   * directly when bootstrapping through imagined rollouts.
   *
   * **`0` means a true terminal state only — never a time-limit truncation.**
   * This project's `done` (`StepResult.done`, `EpisodeStepRecord.done`) is
   * *always* a truncation (`CooperativeGridWorld.step`: `done = currentStep >=
   * horizon`; the env has no absorbing states), so a caller working from
   * ground-truth `done`s passes `1` at every step and lets `bootstrapValue`
   * carry the value past the cut — the same "always bootstrap" rule
   * `QLearningPolicy.update` adopted after PR #43's review
   * (`src/agent/policy.ts:125-129`). Passing `done ? 0 : 1` would sever the
   * return at every episode's last step: the exact bias PR #43 removed.
   * (Session audit B2, 2026-09-23 — this comment previously recommended
   * `done ? 0 : 1`. `WorldModel.step()`'s continue-head target
   * (`continueTargetTensor`, `src/model/worldModel.ts`) still uses that
   * polarity and so learns the horizon; separating terminal from truncated
   * there is a Phase-3 prerequisite before any actor-critic bootstraps
   * through imagination.)
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
