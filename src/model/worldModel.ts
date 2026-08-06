import tf from "@tensorflow/tfjs-node";
import { RSSMCell } from "./rssm.ts";
import type { RSSMConfig, RSSMState } from "./rssm.ts";
import { klBalancedLoss } from "./losses.ts";
import type { KLBalancedLossConfig } from "./losses.ts";
import { Action } from "../env/types.ts";
import type { Observation } from "../env/types.ts";
import { Rng } from "../env/rng.ts";

export interface WorldModelConfig {
  rssm: RSSMConfig;
  /** Fixed length of this environment's `Observation` vector — see `src/env/types.ts`. */
  observationSize: number;
  /** Passed through to `klBalancedLoss` every step. Default: that function's own defaults. */
  lossConfig?: KLBalancedLossConfig;
  /**
   * Adam learning rate. Default `1e-3` (Adam's common default) — an
   * engineering placeholder, not a value derived from or validated against
   * DreamerV3 or this environment; see
   * docs/explainers/0005-world-model-rollout-wiring.md.
   */
  learningRate?: number;
}

export interface WorldModelStepResult {
  /** klBalancedLoss's batch-meaned `total` (dyn+rep only — no reconstruction term yet) as a plain number. */
  loss: number;
}

/**
 * Wraps one `RSSMCell` with an optimizer and the single persistent
 * `RSSMState` a rollout's recurrence carries across an episode — proposal
 * `0001` Arm-A's per-agent world model. See
 * docs/explainers/0005-world-model-rollout-wiring.md for the full design
 * (why `step()` always advances state but only sometimes trains, and the
 * tensor-lifecycle contract with `RSSMCell`).
 */
export class WorldModel {
  readonly cell: RSSMCell;
  private readonly optimizer: tf.Optimizer;
  private readonly lossConfig: KLBalancedLossConfig;
  private readonly trainableVars: tf.Variable[];
  private state: RSSMState;

  constructor(config: WorldModelConfig) {
    this.cell = new RSSMCell(config.rssm);
    this.lossConfig = config.lossConfig ?? {};
    this.optimizer = tf.train.adam(config.learningRate ?? 1e-3);
    this.state = this.cell.initialState(1);

    // Force every layer to build now, with a throwaway forward pass, so
    // `trainableVars` below is the complete set. Necessary because
    // tf.variableGrads with an implicit varList pulls in every other
    // RSSMCell's registered variables too, in the same process
    // (test/model/rssm.test.ts's `ownedVariablesAfter` comment) — every real
    // step() call below passes this explicit list instead.
    tf.tidy(() => {
      const warmupRng = { rng: new Rng(0) };
      const deterministic = this.cell.step(this.state, [Action.Stay]);
      this.cell.prior(deterministic, warmupRng);
      this.cell.posterior(deterministic, tf.zeros([1, config.observationSize]) as tf.Tensor2D, warmupRng);
    });
    this.trainableVars = this.cell.trainableWeights();
  }

  /** The current recurrent state — `deterministic`/`stochastic`, per `RSSMState`. */
  get currentState(): RSSMState {
    return this.state;
  }

  /** Resets to a fresh zero-filled state (episode boundary) — disposes the previous one. */
  reset(): void {
    this.state.deterministic.dispose();
    this.state.stochastic.dispose();
    this.state = this.cell.initialState(1);
  }

  dispose(): void {
    this.state.deterministic.dispose();
    this.state.stochastic.dispose();
  }

  /**
   * Advances one real transition: h_t = step(prevState, action); z_t ~
   * posterior(h_t, observation); loss = klBalancedLoss(prior(h_t), that
   * posterior). When `train` is true, applies one Adam step toward `loss`
   * before advancing; when false, still advances state and returns `loss`
   * (for post-freeze prediction-error tracking, per proposal 0001) but
   * leaves weights untouched. Always disposes the previous state.
   */
  step(action: Action, observation: Observation, rng: Rng, train: boolean): WorldModelStepResult {
    const prevState = this.state;
    const observationTensor = tf.tensor2d([observation]);

    let nextDeterministic!: tf.Tensor2D;
    let nextStochastic!: tf.Tensor2D;

    // tf.variableGrads(f, ...) internally wraps `f` in its own tf.tidy
    // (tfjs's Engine.gradients: `this.tidy('forward', f)`) and disposes
    // everything `f` created except its returned scalar as soon as `f`
    // returns — before control ever comes back to this function. So the two
    // tensors that must outlive this call (the next RSSMState) have to be
    // tf.keep()'d from *inside* forward(), not after variableGrads returns —
    // by then it's too late, they're already disposed. This is true in the
    // train=false branch too (forward() called directly, no variableGrads),
    // so keeping unconditionally inside forward() covers both.
    const forward = (): tf.Scalar => {
      const deterministic = this.cell.step(prevState, [action]);
      const priorDist = this.cell.prior(deterministic, { rng });
      const posteriorDist = this.cell.posterior(deterministic, observationTensor, { rng });
      nextDeterministic = deterministic;
      nextStochastic = posteriorDist.sample;
      tf.keep(nextDeterministic);
      tf.keep(nextStochastic);
      return klBalancedLoss(priorDist, posteriorDist, this.lossConfig).total;
    };

    // The whole tf.variableGrads call (forward *and* backward) runs inside
    // this one outer tf.tidy — not a tidy wrapped around `forward` alone,
    // which would dispose its intermediates before backward pass runs (see
    // RSSMCell's class doc comment and docs/explainers/0005). Everything
    // else forward()/klBalancedLoss create — priorDist's probs/sample,
    // posteriorDist's probs, the KL loss's own dyn/rep components, RSSMCell's
    // internal concat/dense intermediates — gets disposed once this tidy
    // returns (or, for the train=true path, once variableGrads' own internal
    // tidy calls return, which happens first).
    const lossValue = tf.tidy(() => {
      let lossTensor: tf.Scalar;
      if (train) {
        const { value, grads } = tf.variableGrads(forward, this.trainableVars);
        this.optimizer.applyGradients(grads);
        lossTensor = value;
      } else {
        lossTensor = forward();
      }
      return lossTensor.arraySync() as number;
    });

    observationTensor.dispose();
    prevState.deterministic.dispose();
    prevState.stochastic.dispose();
    this.state = { deterministic: nextDeterministic, stochastic: nextStochastic };

    return { loss: lossValue };
  }
}
