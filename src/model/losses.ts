import tf from "@tensorflow/tfjs-node";
import type { LatentDistribution } from "./rssm.ts";

/**
 * Floor for `categoricalKL`'s `log` calls — `probs` comes from `tf.softmax`
 * so it's mathematically `> 0`, but float32 can underflow a low-probability
 * class to exactly `0.0`, and `0 * log(0)` is `NaN` (not `0`) in IEEE float
 * arithmetic, same failure class as `sampleHard`'s `GUMBEL_UNIFORM_EPSILON`
 * guard (`src/model/rssm.ts`) — see docs/explainers/0004-kl-balanced-world-model-loss.md.
 */
const LOG_PROB_EPSILON = 1e-7;

/** DreamerV3 (arXiv:2301.04104) defaults — see docs/explainers/0004-kl-balanced-world-model-loss.md
 * for the confidence tag on these specific values (medium, self_checked — primary-source fetch
 * 403'd this run, cross-checked via WebSearch only). */
const DEFAULT_FREE_BITS = 1;
const DEFAULT_BETA_DYN = 1.0;
const DEFAULT_BETA_REP = 0.1;

/**
 * Identity forward pass, all-zero gradient — this project's `tf.stopGradient`
 * substitute. `tf.stopGradient` does not exist on this project's pinned
 * `@tensorflow/tfjs-node@4.22.0` (same finding as `straightThroughEstimator`
 * in `src/model/rssm.ts`; confirmed by calling it, not by search).
 *
 * `value` is `x.clone()`, not `x` itself: returning the identical input
 * tensor object confirmed empirically to make tfjs's engine skip this op's
 * gradient node entirely during backprop (the *ordinary* gradient of
 * whatever consumes the output leaks through, as if `customGrad` weren't
 * applied at all) — `straightThroughEstimator`'s `value: hard` doesn't hit
 * this because `hard` is never the tensor being differentiated there, only
 * `logits` is. `clone()` is a distinct tensor with identical values, so the
 * aliasing doesn't occur.
 */
export const stopGradient = tf.customGrad((x: tf.Tensor, save: (tensors: tf.Tensor[]) => void) => {
  save([]);
  return {
    value: x.clone(),
    gradFunc: () => [tf.zerosLike(x)],
  };
}) as (x: tf.Tensor) => tf.Tensor;

/**
 * Categorical KL divergence KL[p || q], for `p`/`q` shaped
 * `[batch, categoricals, classes]` (independent categoricals per DreamerV3-
 * style latents — see `RSSMConfig` in `src/model/rssm.ts`). Sums over
 * `classes` per categorical, then over `categoricals`, returning one scalar
 * per batch row (`[batch]`) — deliberately *not* reduced to a single scalar
 * here, since `klBalancedLoss`'s free-bits floor clips per row before
 * averaging (see docs/explainers/0004-kl-balanced-world-model-loss.md).
 */
export function categoricalKL(p: tf.Tensor3D, q: tf.Tensor3D): tf.Tensor1D {
  return tf.tidy(() => {
    const logP = tf.log(tf.add(p, LOG_PROB_EPSILON));
    const logQ = tf.log(tf.add(q, LOG_PROB_EPSILON));
    const term = tf.mul(p, tf.sub(logP, logQ));
    return tf.sum(term, [1, 2]) as tf.Tensor1D;
  });
}

/**
 * Observation reconstruction loss `L_pred` — sum of squared error between
 * `predicted`/`target` (each `[batch, observationSize]`) per row, then
 * batch-meaned, mirroring `categoricalKL`'s "reduce over the per-example
 * axes, then batch-mean" shape. Equal (up to an additive constant and a
 * `0.5` factor that doesn't change what minimizes it) to the negative
 * log-likelihood of `target` under a unit-variance isotropic Gaussian
 * centered at `predicted` — the standard simplification of DreamerV3's
 * `L_pred = -log p(o_t | h_t, z_t)`, without that paper's `symlog` transform
 * (this environment's `Observation` is already normalized to a fixed small
 * range, so there's no multi-order-of-magnitude spread for `symlog` to
 * compress — see docs/explainers/0006-observation-reconstruction-loss.md
 * for the full reasoning and confidence tag).
 */
export function reconstructionLoss(predicted: tf.Tensor2D, target: tf.Tensor2D): tf.Scalar {
  return tf.tidy(() => {
    const squaredError = tf.sum(tf.square(tf.sub(predicted, target)), 1);
    return tf.mean(squaredError) as tf.Scalar;
  });
}

export interface KLBalancedLossConfig {
  /** Nats floor each of dynLoss/repLoss is clipped to, per batch row, before the batch mean. Default 1. */
  freeBits?: number;
  /** Weight on dynLoss in `total`. Default 1.0. */
  betaDyn?: number;
  /** Weight on repLoss in `total`. Default 0.1. */
  betaRep?: number;
}

export interface KLBalancedLoss {
  /** max(freeBits, KL[sg(posterior) || prior]), batch-meaned. Gradients flow into `prior` only. */
  dynLoss: tf.Scalar;
  /** max(freeBits, KL[posterior || sg(prior)]), batch-meaned. Gradients flow into `posterior` only. */
  repLoss: tf.Scalar;
  /** betaDyn * dynLoss + betaRep * repLoss. */
  total: tf.Scalar;
}

/**
 * DreamerV3-style KL-balanced dynamics/representation loss between one
 * timestep's prior and posterior latent distributions. See
 * docs/explainers/0004-kl-balanced-world-model-loss.md for the formula,
 * defaults, and what this deliberately does not cover yet (reconstruction
 * loss, sequence/time batching, rollout wiring).
 */
export function klBalancedLoss(
  prior: LatentDistribution,
  posterior: LatentDistribution,
  config: KLBalancedLossConfig = {},
): KLBalancedLoss {
  const freeBits = config.freeBits ?? DEFAULT_FREE_BITS;
  const betaDyn = config.betaDyn ?? DEFAULT_BETA_DYN;
  const betaRep = config.betaRep ?? DEFAULT_BETA_REP;

  return tf.tidy(() => {
    const priorProbs = prior.probs;
    const posteriorProbs = posterior.probs;

    const dynKL = categoricalKL(stopGradient(posteriorProbs) as tf.Tensor3D, priorProbs);
    const repKL = categoricalKL(posteriorProbs, stopGradient(priorProbs) as tf.Tensor3D);

    const dynLoss = tf.mean(tf.maximum(dynKL, freeBits)) as tf.Scalar;
    const repLoss = tf.mean(tf.maximum(repKL, freeBits)) as tf.Scalar;
    const total = tf.add(tf.mul(betaDyn, dynLoss), tf.mul(betaRep, repLoss)) as tf.Scalar;

    return { dynLoss, repLoss, total };
  });
}
