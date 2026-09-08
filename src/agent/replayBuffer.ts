import type { Transition } from "./policy.ts";
import type { Rng } from "../env/rng.ts";

/**
 * One replay buffer entry: a `Transition` plus the bookkeeping a
 * recency-weighted sampling scheme needs "recency" to mean something
 * concrete over. `insertedAt` is a monotonically increasing sequence number
 * assigned by `ReplayBuffer.add()`, not a wall-clock timestamp — matches
 * this codebase's existing convention of measuring recency/age in step
 * counts, not real time (e.g. `postFreezeLossSeries`'s steps-since-freeze
 * array-index alignment, `src/experiment/metrics.ts`).
 */
export interface ReplayBufferEntry {
  transition: Transition;
  insertedAt: number;
}

export type ReplaySamplingStrategy = "uniform" | "recency-weighted";

export interface ReplayBufferConfig {
  /**
   * Maximum entries held at once. Required, not defaulted — proposal 0001's
   * laptop-scale budget (`docs/proposals/0001-direct-nonstationarity-measurement.md`,
   * "Estimated cost") means an unbounded buffer is never the intended
   * configuration for this project; forcing a value here means a caller
   * can't silently inherit an accidental unbounded buffer by omission.
   */
  capacity: number;
  /**
   * `"uniform"`: every entry equally likely to be sampled. `"recency-weighted"`:
   * proposal 0001's ablation 3, MATWM-style mechanism — newer entries
   * weighted higher (`docs/proposals/0001-direct-nonstationarity-measurement.md`,
   * ablation 3). `recencyHalfLife` is required when this is
   * `"recency-weighted"`; the exact weighting formula is left to the
   * implementation — see `docs/explainers/0012-replay-buffer-lambda-returns-spec.md`'s
   * "Open design questions", this interface only pins the configuration
   * surface, not the math.
   */
  sampling: ReplaySamplingStrategy;
  /**
   * Number of insertions after which a `"recency-weighted"` entry's sampling
   * weight has decayed to half its freshly-inserted value. Required when
   * `sampling === "recency-weighted"`, meaningless (and should be omitted)
   * for `"uniform"` — enforce this at construction, not just in a comment,
   * so a misconfigured buffer fails at startup rather than silently
   * sampling uniformly under a `"recency-weighted"` label.
   */
  recencyHalfLife?: number;
}

/**
 * SPEC ONLY — reserved for the human's Gate G2 role-flip implementation
 * (`loop/GOAL.md` priority 6; `PLAN.html`'s Gate G2 line: "the human has
 * personally implemented ≥2 modules (replay buffer, λ-returns) with Claude
 * reviewing"). Every method below throws unconditionally; this class exists
 * so the *interface* — what a caller can rely on once implemented — is
 * reviewable and typecheckable now, and so `test/agent/replayBuffer.test.ts`'s
 * `{ todo }` skeletons have a concrete type to import against. See
 * `docs/explainers/0012-replay-buffer-lambda-returns-spec.md` for the full
 * design rationale, the two consumers this interface is meant to serve
 * (ablation 3's prioritized replay and a future actor-critic's ordinary
 * experience replay), and the open questions the implementation still needs
 * to settle (eviction policy, exact recency-weighting formula, sampling
 * with/without replacement, per-agent vs. shared buffers).
 */
export class ReplayBuffer {
  constructor(config: ReplayBufferConfig) {
    void config;
    throw new Error(
      "ReplayBuffer is a Gate G2 spec-only skeleton (loop/GOAL.md priority 6) — " +
        "implementation is reserved for the human; see docs/explainers/0012.",
    );
  }

  /** Adds one transition. Expected to evict per the implementation's chosen policy once at capacity. */
  add(_transition: Transition): void {
    throw new Error("ReplayBuffer.add is a Gate G2 spec-only skeleton — see docs/explainers/0012.");
  }

  /**
   * Draws `batchSize` entries via `rng`, distributed per `config.sampling`.
   * Takes an explicit `Rng` rather than an internal random source — same
   * reproducibility reasoning `RandomPolicy.act`/`QLearningPolicy.act`
   * already establish (`src/agent/policy.ts`, `docs/explainers/0008`).
   */
  sample(_batchSize: number, _rng: Rng): ReplayBufferEntry[] {
    throw new Error("ReplayBuffer.sample is a Gate G2 spec-only skeleton — see docs/explainers/0012.");
  }

  /** Current entry count (`<= config.capacity`). */
  size(): number {
    throw new Error("ReplayBuffer.size is a Gate G2 spec-only skeleton — see docs/explainers/0012.");
  }
}
