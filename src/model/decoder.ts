import tf from "@tensorflow/tfjs-node";

export interface DecoderConfig {
  /** Fixed length of this environment's `Observation` vector — see `src/env/types.ts`. */
  observationSize: number;
  /**
   * Seeds the dense layer's kernel initializer — see `RSSMConfig.seed`
   * (`src/model/rssm.ts`) for why this exists and what it does and doesn't
   * cover. Omit to fall back to tfjs's unseeded global initializer, as
   * before this field existed.
   */
  seed?: number;
}

/**
 * Reconstructs the observation vector from an RSSM state `(h_t, z_t)` — the
 * decoder half of a DreamerV3-style world model. Kept as its own network
 * rather than a method on `RSSMCell`: the paper's RSSM is the dynamics core
 * (deterministic recurrence + prior/posterior over the stochastic latent)
 * only, with the decoder (and reward/continuation heads, not built yet)
 * reading off `(h_t, z_t)` separately. See
 * docs/explainers/0006-observation-reconstruction-loss.md.
 *
 * One linear dense layer, not a deeper MLP — matches `RSSMCell`'s
 * `priorDense`/`posteriorDense` minimalism; this environment's `Observation`
 * is a small, pre-normalized, hand-engineered vector, not raw pixels, so
 * there's no established need for decoder depth yet.
 */
export class ObservationDecoder {
  private readonly dense: ReturnType<typeof tf.layers.dense>;

  constructor(config: DecoderConfig) {
    this.dense = tf.layers.dense({
      units: config.observationSize,
      ...(config.seed !== undefined && {
        kernelInitializer: tf.initializers.glorotNormal({ seed: config.seed }),
      }),
    });
  }

  /** Reconstructs `[batch, observationSize]` from `concat([deterministic, stochastic], 1)`. */
  decode(deterministic: tf.Tensor2D, stochastic: tf.Tensor2D): tf.Tensor2D {
    const input = tf.concat([deterministic, stochastic], 1);
    return this.dense.apply(input) as tf.Tensor2D;
  }

  /**
   * Every trainable `tf.Variable` in the decoder's dense layer — for
   * `tf.variableGrads(fn, varList)` callers (`src/model/worldModel.ts`).
   * Only meaningful after the layer has built (its first `decode()` call) —
   * same caveat as `RSSMCell.trainableWeights()`.
   */
  trainableWeights(): tf.Variable[] {
    // tfjs-layers' LayerVariable.val is `protected`, not part of its public
    // type — same access pattern `RSSMCell.trainableWeights()` already uses.
    const layer = this.dense as unknown as { trainableWeights: Array<{ val: tf.Variable }> };
    return layer.trainableWeights.map((w) => w.val);
  }
}
