import tf from "@tensorflow/tfjs-node";

export interface RewardHeadConfig {
  /**
   * Seeds the dense layer's kernel initializer — same convention as
   * `ContinueHeadConfig.seed` (`src/model/continueHead.ts`). Omit to fall
   * back to tfjs's unseeded global initializer.
   */
  seed?: number;
}

/**
 * Predicts the scalar transition reward from the RSSM state `(h_t, z_t)` —
 * the reward-prediction half of a DreamerV3-style world model's `L_pred`
 * heads, alongside `ObservationDecoder` and `ContinueHead`. See
 * docs/explainers/0014-reward-head-spec.md for why this closes
 * docs/explainers/0013's blocking dependency, and why no `symlog` transform
 * is used here (this environment's reward is already bounded to a fixed
 * small range, same reasoning `reconstructionLoss`'s doc comment gives for
 * `Observation`).
 *
 * One linear dense layer (`units: 1`, no activation) — same minimalism
 * convention as `ContinueHead`, but unlike `ContinueHead`'s logit (squashed
 * downstream by `sigmoid`), this output is the reward prediction itself,
 * with no further transform.
 */
export class RewardHead {
  private readonly dense: ReturnType<typeof tf.layers.dense>;

  constructor(config: RewardHeadConfig = {}) {
    this.dense = tf.layers.dense({
      units: 1,
      ...(config.seed !== undefined && {
        kernelInitializer: tf.initializers.glorotNormal({ seed: config.seed }),
      }),
    });
  }

  /** Scalar reward prediction, `[batch, 1]` — differentiable, no activation. */
  predict(deterministic: tf.Tensor2D, stochastic: tf.Tensor2D): tf.Tensor2D {
    const input = tf.concat([deterministic, stochastic], 1);
    return this.dense.apply(input) as tf.Tensor2D;
  }

  /**
   * Every trainable `tf.Variable` in this head's dense layer — for
   * `tf.variableGrads(fn, varList)` callers (`src/model/worldModel.ts`).
   * Only meaningful after the layer has built (its first `predict()` call)
   * — same caveat as `ContinueHead.trainableWeights()`.
   */
  trainableWeights(): tf.Variable[] {
    // tfjs-layers' LayerVariable.val is `protected`, not part of its public
    // type — same access pattern `ContinueHead.trainableWeights()` already uses.
    const layer = this.dense as unknown as { trainableWeights: Array<{ val: tf.Variable }> };
    return layer.trainableWeights.map((w) => w.val);
  }
}
