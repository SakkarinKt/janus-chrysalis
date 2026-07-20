import tf from "@tensorflow/tfjs-node";
import { Action } from "../env/types.ts";

const NUM_ACTIONS = Object.keys(Action).length;

export interface RSSMConfig {
  /** Size of the GRU-based deterministic recurrent state. */
  deterministicSize: number;
  /** Number of independent categorical variables in the stochastic latent (DreamerV3-style). */
  latentCategoricals: number;
  /** Number of classes per categorical variable. */
  latentClasses: number;
}

export interface RSSMState {
  /** GRU hidden state, shape [batch, deterministicSize]. */
  deterministic: tf.Tensor2D;
  /**
   * Straight-through sample of the categorical latent held by this state,
   * flattened to [batch, latentCategoricals * latentClasses]. Zero for the
   * initial state (no sample has been drawn yet); otherwise the output of
   * `prior()` or `posterior()` from the previous step.
   */
  stochastic: tf.Tensor2D;
}

/** A categorical latent distribution and a straight-through sample from it. */
export interface LatentDistribution {
  /** Softmax probabilities, [batch, latentCategoricals, latentClasses]. */
  probs: tf.Tensor3D;
  /** Straight-through sample, flattened to [batch, latentCategoricals * latentClasses]. */
  sample: tf.Tensor2D;
}

/**
 * The RSSM cell: a GRU-based deterministic recurrence plus a categorical
 * stochastic latent with prior/posterior heads (DreamerV3-style), per PR #14's
 * review ("split the cell as you proposed — struct/forward-pass, then STE +
 * gradient-check") and PR #19's reply resuming this deferred sub-increment.
 * The deterministic path composes `tf.layers.gruCell` via `tf.layers.rnn`
 * rather than hand-rolling GRU gate math, per
 * notes/rssm-vs-ssm-implementation-robustness.md §3's finding that TF.js's
 * own GRU layer is the mature, tested primitive to lean on.
 */
export class RSSMCell {
  readonly config: RSSMConfig;
  private readonly cell: ReturnType<typeof tf.layers.gruCell>;
  private readonly rnn: ReturnType<typeof tf.layers.rnn>;
  private readonly priorDense: ReturnType<typeof tf.layers.dense>;
  private readonly posteriorDense: ReturnType<typeof tf.layers.dense>;

  constructor(config: RSSMConfig) {
    this.config = config;
    this.cell = tf.layers.gruCell({ units: config.deterministicSize });
    this.rnn = tf.layers.rnn({ cell: this.cell, returnState: true, returnSequences: false });
    const stochasticSize = config.latentCategoricals * config.latentClasses;
    this.priorDense = tf.layers.dense({ units: stochasticSize });
    this.posteriorDense = tf.layers.dense({ units: stochasticSize });
  }

  initialState(batchSize: number): RSSMState {
    const stochasticSize = this.config.latentCategoricals * this.config.latentClasses;
    return {
      deterministic: tf.zeros([batchSize, this.config.deterministicSize]),
      stochastic: tf.zeros([batchSize, stochasticSize]),
    };
  }

  /**
   * One deterministic recurrent step:
   * h_t = GRU(h_{t-1}, [onehot(action_{t-1}), z_{t-1}]).
   * `actions` is one discrete action per batch element (the action each
   * agent took to produce the transition landing on this step); `prevState`
   * carries the previous step's stochastic sample `z_{t-1}` (zero for the
   * first step). Returns `h_t` only — callers combine it with `prior()` or
   * `posterior()` to get the full next `RSSMState`, since which one supplies
   * `z_t` depends on whether an observation is available.
   */
  step(prevState: RSSMState, actions: Action[]): tf.Tensor2D {
    const recurrentInput = tf.concat([oneHotActions(actions), prevState.stochastic], 1);
    const inputSeq = recurrentInput.reshape([actions.length, 1, recurrentInput.shape[1]]);
    const output = this.rnn.apply(inputSeq, { initialState: [prevState.deterministic] }) as tf.Tensor2D[];
    return output[0];
  }

  /** Prior p(z_t | h_t) — no observation, used for imagination rollouts. */
  prior(deterministic: tf.Tensor2D): LatentDistribution {
    const logits = this.reshapeLogits(this.priorDense.apply(deterministic) as tf.Tensor2D);
    return sampleStraightThrough(logits);
  }

  /** Posterior q(z_t | h_t, o_t) — conditions on a real observation, used for training. */
  posterior(deterministic: tf.Tensor2D, observation: tf.Tensor2D): LatentDistribution {
    const input = tf.concat([deterministic, observation], 1);
    const logits = this.reshapeLogits(this.posteriorDense.apply(input) as tf.Tensor2D);
    return sampleStraightThrough(logits);
  }

  private reshapeLogits(flat: tf.Tensor2D): tf.Tensor3D {
    return flat.reshape([flat.shape[0], this.config.latentCategoricals, this.config.latentClasses]);
  }
}

/**
 * One-hot encodes a batch of discrete actions to shape [batch, NUM_ACTIONS],
 * cast to float32 — `tf.oneHot` defaults to int32, which doesn't `tf.concat`
 * with the float32 stochastic latent `step()` now concatenates it against.
 */
function oneHotActions(actions: Action[]): tf.Tensor2D {
  return tf.oneHot(actions, NUM_ACTIONS).toFloat() as tf.Tensor2D;
}

/**
 * The straight-through categorical estimator, given logits and a fixed hard
 * sample drawn from them. Forward pass returns exactly `hard`. Backward pass,
 * gradients w.r.t. `logits` are the softmax Jacobian-vector product — i.e.
 * identical to differentiating `softmax(logits)` alone — and gradients w.r.t.
 * `hard` are zero (`hard` is a non-differentiable sample, not a function of
 * `logits`, as far as this op is concerned). Exported standalone so its
 * gradient can be checked directly (see test/model/rssm.test.ts) against a
 * fixed `hard`, independent of `sampleHard`'s randomness.
 *
 * Implemented with `tf.customGrad`, not the textbook
 * `hard - stopGradient(soft) + soft` composition: `tf.stopGradient` does not
 * exist in this project's pinned `@tensorflow/tfjs-node@4.22.0` (confirmed by
 * running it — `tf.stopGradient is not a function` on both `tfjs-node` and
 * plain `tfjs`'s public API) despite
 * `notes/adr-0002-js-ml-stack.md` §4 recording it as "a native, stable TF.js
 * primitive... confirmed present." That confirmation doesn't hold up under
 * actually exercising the API in this pinned version — worth a correction
 * pass on that note. `tf.customGrad` (confirmed present) is the documented
 * alternative for exactly this "custom backward, given a forward value" case.
 *
 * Also note: naive finite-difference checking *cannot* validate this
 * function's gradient against its own forward pass, because the forward
 * value (`hard`) is piecewise-constant in `logits` by construction — a
 * discrete tensor supplied independently of `logits`'s continuous value, so
 * `(f(x+e) - f(x-e)) / 2e` is identically zero regardless of whether the
 * custom gradient is correct. The finite-difference check instead has to
 * target `softmax(logits)` directly (see test/model/rssm.test.ts), which is
 * what this straight-through gradient is defined to reproduce.
 */
export const straightThroughEstimator = tf.customGrad(
  (logits: tf.Tensor, hard: tf.Tensor, save: (tensors: tf.Tensor[]) => void) => {
    const soft = tf.softmax(logits, -1);
    save([soft]);
    return {
      value: hard,
      gradFunc: (dy: tf.Tensor, saved: tf.Tensor[]) => {
        const [savedSoft] = saved;
        const dot = tf.sum(tf.mul(dy, savedSoft), -1, true);
        const gradLogits = tf.mul(savedSoft, tf.sub(dy, dot));
        return [gradLogits, tf.zerosLike(hard)];
      },
    };
  },
) as (logits: tf.Tensor, hard: tf.Tensor) => tf.Tensor;

/**
 * Draws one hard categorical sample per [.., category] row of `logits` via
 * `tf.multinomial`, cast to float32 (`tf.oneHot` defaults to int32, but this
 * flows into `straightThroughEstimator`'s output, which `RSSMCell.step()`
 * later `tf.concat`s against the float32 recurrent state — see
 * `oneHotActions`'s equivalent cast for the same reason).
 */
export function sampleHard(logits: tf.Tensor3D): tf.Tensor3D {
  const [batch, categoricals, classes] = logits.shape;
  const flatLogits = logits.reshape([batch * categoricals, classes]) as tf.Tensor2D;
  const indices = (tf.multinomial(flatLogits, 1) as tf.Tensor2D).reshape([batch * categoricals]);
  return tf.oneHot(indices, classes).toFloat().reshape([batch, categoricals, classes]) as tf.Tensor3D;
}

/** Samples a categorical latent from `logits` via the straight-through estimator. */
export function sampleStraightThrough(logits: tf.Tensor3D): LatentDistribution {
  const probs = tf.softmax(logits, -1) as tf.Tensor3D;
  const hard = sampleHard(logits);
  const sample = straightThroughEstimator(logits, hard) as tf.Tensor3D;
  return { probs, sample: sample.reshape([logits.shape[0], -1]) as tf.Tensor2D };
}
