import tf from "@tensorflow/tfjs-node";
import { Action } from "../env/types.ts";

const NUM_ACTIONS = Object.keys(Action).length;

export interface RSSMConfig {
  /** Size of the GRU-based deterministic recurrent state. */
  deterministicSize: number;
}

export interface RSSMState {
  /**
   * GRU hidden state, shape [batch, deterministicSize]. The stochastic
   * categorical latent (prior/posterior, straight-through sampling) is not
   * part of this increment — see docs/explainers/0003-rssm-world-model-cell.md
   * for why it's deferred to the next sub-increment.
   */
  deterministic: tf.Tensor2D;
}

/**
 * The RSSM's deterministic recurrent path only (struct/forward-pass
 * sub-increment, per PR #14 review: "split the cell as you proposed —
 * struct/forward-pass, then STE + gradient-check"). Composes
 * `tf.layers.gruCell` via `tf.layers.rnn` rather than hand-rolling GRU gate
 * math, per notes/rssm-vs-ssm-implementation-robustness.md §3's finding that
 * TF.js's own GRU layer is the mature, tested primitive to lean on. The
 * stochastic categorical latent, its straight-through estimator
 * (`tf.stopGradient`), and the finite-difference gradient-check test §5
 * recommends are the explicitly deferred next sub-increment.
 */
export class RSSMCell {
  readonly config: RSSMConfig;
  private readonly cell: ReturnType<typeof tf.layers.gruCell>;
  private readonly rnn: ReturnType<typeof tf.layers.rnn>;

  constructor(config: RSSMConfig) {
    this.config = config;
    this.cell = tf.layers.gruCell({ units: config.deterministicSize });
    this.rnn = tf.layers.rnn({ cell: this.cell, returnState: true, returnSequences: false });
  }

  initialState(batchSize: number): RSSMState {
    return { deterministic: tf.zeros([batchSize, this.config.deterministicSize]) };
  }

  /**
   * One deterministic recurrent step: h_t = GRU(h_{t-1}, onehot(action_{t-1})).
   * `actions` is one discrete action per batch element (the action each
   * agent took to produce the transition landing on this step).
   */
  step(prevState: RSSMState, actions: Action[]): RSSMState {
    const inputSeq = oneHotActions(actions).reshape([actions.length, 1, NUM_ACTIONS]);
    const output = this.rnn.apply(inputSeq, { initialState: [prevState.deterministic] }) as tf.Tensor2D[];
    return { deterministic: output[0] };
  }
}

/** One-hot encodes a batch of discrete actions to shape [batch, NUM_ACTIONS]. */
function oneHotActions(actions: Action[]): tf.Tensor2D {
  return tf.oneHot(actions, NUM_ACTIONS) as tf.Tensor2D;
}
