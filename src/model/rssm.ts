import tf from "@tensorflow/tfjs-node";
import { Action } from "../env/types.ts";
import type { Rng } from "../env/rng.ts";

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
 * The deterministic path calls `tf.layers.gruCell.call()` directly, one
 * timestep at a time — **not** through a `tf.layers.rnn` wrapper — per the
 * week-3 stack-validation spike (2026-07-22, processing PR #22's review):
 * chaining `tf.layers.rnn.apply()` calls across >=2 timesteps and
 * differentiating through the chain crashes `tf.variableGrads` inside
 * tfjs-layers' own RNN backward pass (`tensorflow/tfjs#1529`, `#3550`) —
 * root-caused to `rnn()`'s internal `tfc.unstack`/`tfc.stack` bookkeeping,
 * which runs even for a length-1 sequence. `GRUCell.call()` itself is plain
 * matmul/split/activation ops with no stack/unstack anywhere, so calling it
 * directly (bypassing the wrapper's sequence machinery) avoids the bug
 * entirely, confirmed by a from-scratch gradient-check across up to 4
 * chained steps (`test/model/rssm.test.ts`). `notes/adr-0002-js-ml-stack.md`
 * §9 has the full investigation; this supersedes §8's "likely tripped" kill-
 * criterion read — the criterion did not fire, no custom autograd needed.
 * `GRUCell` is still the mature, tested primitive per
 * notes/rssm-vs-ssm-implementation-robustness.md §3 — only the wrapper
 * around it changed.
 */
export class RSSMCell {
  readonly config: RSSMConfig;
  private readonly cell: ReturnType<typeof tf.layers.gruCell>;
  private readonly priorDense: ReturnType<typeof tf.layers.dense>;
  private readonly posteriorDense: ReturnType<typeof tf.layers.dense>;

  constructor(config: RSSMConfig) {
    this.config = config;
    this.cell = tf.layers.gruCell({ units: config.deterministicSize });
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
   *
   * Calls `this.cell.call()` directly rather than going through a
   * `tf.layers.rnn`-wrapped `.apply()` — see the class doc comment for why:
   * chaining the wrapper across timesteps within one differentiation trace
   * crashes `tf.variableGrads`, but the cell's own `call()` is plain
   * differentiable tensor ops with no such wrapper machinery in the way.
   * Builds the cell's weights lazily on the first call (mirroring
   * `tf.layers.rnn`'s own build-on-first-`apply()` behavior, and every other
   * layer in this class), against the actual recurrent-input width rather
   * than one computed ahead of time from config.
   */
  step(prevState: RSSMState, actions: Action[]): tf.Tensor2D {
    const recurrentInput = tf.concat([oneHotActions(actions), prevState.stochastic], 1);
    if (!this.cell.built) this.cell.build([null, recurrentInput.shape[1]]);
    const [output] = this.cell.call([recurrentInput, prevState.deterministic], { training: false }) as tf.Tensor2D[];
    return output;
  }

  /**
   * Prior p(z_t | h_t) — no observation, used for imagination rollouts.
   * `fixedHard`, when given, is used in place of a freshly-drawn sample (see
   * `sampleStraightThrough`) — needed by the end-to-end gradient-check test,
   * since a fresh `sampleHard` draw on every forward call isn't a function of
   * the weights being perturbed, which breaks finite-differencing.
   * `rng`: required whenever `fixedHard` is omitted — see `sampleHard`'s doc
   * comment for why (quality-pass finding #1,
   * `reports/quality/2026-08-03-quality-pass.md`).
   */
  prior(deterministic: tf.Tensor2D, fixedHard?: tf.Tensor3D, rng?: Rng): LatentDistribution {
    const logits = this.reshapeLogits(this.priorDense.apply(deterministic) as tf.Tensor2D);
    return sampleStraightThrough(logits, fixedHard, rng);
  }

  /**
   * Posterior q(z_t | h_t, o_t) — conditions on a real observation, used for
   * training. `fixedHard`, `rng`: see `prior()`.
   */
  posterior(
    deterministic: tf.Tensor2D,
    observation: tf.Tensor2D,
    fixedHard?: tf.Tensor3D,
    rng?: Rng,
  ): LatentDistribution {
    const input = tf.concat([deterministic, observation], 1);
    const logits = this.reshapeLogits(this.posteriorDense.apply(input) as tf.Tensor2D);
    return sampleStraightThrough(logits, fixedHard, rng);
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
 * Floor for the uniform draws `sampleHard`'s Gumbel noise is built from —
 * `-Math.log(-Math.log(u))` is `-Infinity`/`NaN` at `u`'s domain edges
 * (`u = 0` or `u = 1`), and `Rng.next()`'s documented range `[0, 1)` already
 * touches the lower edge.
 */
const GUMBEL_UNIFORM_EPSILON = 1e-7;

/**
 * Draws one hard categorical sample per [.., category] row of `logits`,
 * cast to float32 (`tf.oneHot` defaults to int32, but this flows into
 * `straightThroughEstimator`'s output, which `RSSMCell.step()` later
 * `tf.concat`s against the float32 recurrent state — see `oneHotActions`'s
 * equivalent cast for the same reason).
 *
 * `rng` is required. Every other source of randomness in this repo threads
 * through the project's `Rng` (environment spawn positions, `RandomPolicy.act`);
 * this was the one exception, silently non-reproducible per-seed until fixed
 * (quality-pass finding #1, `reports/quality/2026-08-03-quality-pass.md`: two
 * fresh `RSSMCell`s, same config/state/action, drew different unseeded
 * samples).
 *
 * Implemented as a Gumbel-max sample (`argmax(logits + Gumbel noise)`,
 * equivalent to sampling `Categorical(softmax(logits))`) with the Gumbel
 * noise drawn from `rng` via inverse-CDF (`-log(-log(u))`, `u ~ rng.next()`),
 * **not** `tf.multinomial(logits, 1, seed)` despite tfjs's own `seed` param
 * looking like the obvious fit: on this project's pinned
 * `@tensorflow/tfjs-node@4.22.0`, two `tf.multinomial` calls with the
 * identical numeric seed do not reliably reproduce the same draw — confirmed
 * directly:
 * ```js
 * const logits = tf.tensor2d([[1, 2, 3], [0, 0, 0]]);
 * tf.multinomial(logits, 1, 42).arraySync(); // [[2], [2]]
 * tf.multinomial(logits, 1, 42).arraySync(); // [[2], [0]] — same seed, second row differs
 * ```
 * (native TF's `Multinomial` op only guarantees determinism when a
 * graph-level seed is also set, which tfjs's `multinomial()` binding has no
 * way to do — op-level `seed`/`seed2` alone still pull fresh entropy per
 * call). Routing the draw through `rng` directly, instead of asking a
 * TF-native op to honor a seed it doesn't reliably honor, sidesteps that
 * gap entirely and was verified reproducible the same way (same `Rng` seed,
 * two fresh calls, identical output) plus a distributional sanity check
 * (3000 draws from skewed logits landed within 1% of the softmax
 * proportions).
 */
export function sampleHard(logits: tf.Tensor3D, rng: Rng): tf.Tensor3D {
  return tf.tidy(() => {
    const [batch, categoricals, classes] = logits.shape;
    const size = batch * categoricals * classes;
    const gumbelNoise = new Float32Array(size);
    for (let i = 0; i < size; i++) {
      const u = Math.min(Math.max(rng.next(), GUMBEL_UNIFORM_EPSILON), 1 - GUMBEL_UNIFORM_EPSILON);
      gumbelNoise[i] = -Math.log(-Math.log(u));
    }
    const noise = tf.tensor(gumbelNoise, [batch, categoricals, classes]);
    const indices = tf.argMax(tf.add(logits, noise), -1);
    return tf.oneHot(indices, classes).toFloat() as tf.Tensor3D;
  });
}

/**
 * Samples a categorical latent from `logits` via the straight-through
 * estimator. `fixedHard`, when given, replaces the internal `sampleHard`
 * draw — see `RSSMCell.prior()`'s doc comment for why. `rng` is required
 * whenever `fixedHard` is omitted (throws otherwise) — see `sampleHard`'s
 * doc comment.
 */
export function sampleStraightThrough(logits: tf.Tensor3D, fixedHard?: tf.Tensor3D, rng?: Rng): LatentDistribution {
  const probs = tf.softmax(logits, -1) as tf.Tensor3D;
  let hard: tf.Tensor3D;
  if (fixedHard) {
    hard = fixedHard;
  } else {
    if (!rng) {
      throw new Error("sampleStraightThrough: rng is required when fixedHard is not given (see sampleHard's doc comment)");
    }
    hard = sampleHard(logits, rng);
  }
  const sample = straightThroughEstimator(logits, hard) as tf.Tensor3D;
  return { probs, sample: sample.reshape([logits.shape[0], -1]) as tf.Tensor2D };
}
