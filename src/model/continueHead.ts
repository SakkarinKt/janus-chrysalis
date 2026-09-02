import tf from "@tensorflow/tfjs-node";

export interface ContinueHeadConfig {
  /**
   * Seeds the dense layer's kernel initializer — see `RSSMConfig.seed`
   * (`src/model/rssm.ts`) for why this exists and what it does and doesn't
   * cover. Omit to fall back to tfjs's unseeded global initializer, as
   * before this field existed.
   */
  seed?: number;
}

/**
 * Predicts whether the episode continues past a transition, from the RSSM
 * state `(h_t, z_t)` — the continuation half of a DreamerV3-style world
 * model's `L_pred` heads. Kept as its own network, matching
 * `ObservationDecoder`'s reasoning: the RSSM (`src/model/rssm.ts`) is the
 * dynamics core only, with decoder/reward/continuation heads reading off
 * `(h_t, z_t)` separately. See
 * docs/explainers/0011-continue-termination-head.md — including why this
 * sub-increment's target is horizon-deterministic scaffolding, not a
 * genuine prediction task, in Arm-A's gridworld.
 *
 * One linear dense layer producing a raw logit (`units: 1`, no activation)
 * — matches `ObservationDecoder`'s single-dense-layer minimalism, and
 * `continueLoss` (`src/model/losses.ts`) expects a logit, not a
 * pre-squashed probability.
 */
export class ContinueHead {
  private readonly dense: ReturnType<typeof tf.layers.dense>;

  constructor(config: ContinueHeadConfig = {}) {
    this.dense = tf.layers.dense({
      units: 1,
      ...(config.seed !== undefined && {
        kernelInitializer: tf.initializers.glorotNormal({ seed: config.seed }),
      }),
    });
  }

  /** Raw continuation logit, `[batch, 1]` — `sigmoid(logit)` is P(episode continues past this transition). */
  predict(deterministic: tf.Tensor2D, stochastic: tf.Tensor2D): tf.Tensor2D {
    const input = tf.concat([deterministic, stochastic], 1);
    return this.dense.apply(input) as tf.Tensor2D;
  }

  /**
   * Every trainable `tf.Variable` in this head's dense layer — for
   * `tf.variableGrads(fn, varList)` callers (`src/model/worldModel.ts`).
   * Only meaningful after the layer has built (its first `predict()` call)
   * — same caveat as `ObservationDecoder.trainableWeights()`.
   */
  trainableWeights(): tf.Variable[] {
    // tfjs-layers' LayerVariable.val is `protected`, not part of its public
    // type — same access pattern `ObservationDecoder.trainableWeights()`
    // already uses.
    const layer = this.dense as unknown as { trainableWeights: Array<{ val: tf.Variable }> };
    return layer.trainableWeights.map((w) => w.val);
  }
}
