import tf from "@tensorflow/tfjs-node";
import { RSSMCell } from "./rssm.ts";
import type { RSSMConfig, RSSMState } from "./rssm.ts";
import { klBalancedLoss, reconstructionLoss, continueLoss } from "./losses.ts";
import type { KLBalancedLossConfig } from "./losses.ts";
import { ObservationDecoder } from "./decoder.ts";
import { ContinueHead } from "./continueHead.ts";
import { Action } from "../env/types.ts";
import type { Observation } from "../env/types.ts";
import { deriveSeed, Rng } from "../env/rng.ts";

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
  /**
   * Seeds this model's weight initialization — `deriveSeed`d into one
   * stream for the `RSSMCell` (salt 0) and one for the `ObservationDecoder`
   * (salt 1), so the two don't draw correlated initial weights from the
   * same raw seed. Takes precedence over a `seed` set directly on `rssm`.
   * Omit for tfjs's unseeded global initializer, as before this field
   * existed — see `RSSMConfig.seed`'s doc comment (`src/model/rssm.ts`)
   * for what this does and doesn't cover.
   */
  seed?: number;
}

/**
 * Thrown by `WorldModel.step()` when the computed loss, or any of its
 * `reconstructionLoss`/`klLoss`/`continueLoss` components, is non-finite
 * (`NaN` or `±Infinity`) — the "NaN → graceful halt" invariant
 * (`loop/GOAL.md` priority 5). See `docs/explainers/0009-worldmodel-nan-halt.md`
 * for what this does and doesn't guarantee: in particular, for a `train: true`
 * step it cannot undo the optimizer update `forward()` already applied before
 * the non-finite value was ever read back — it only stops that corruption
 * from silently propagating into every later step() call.
 */
export class WorldModelNaNError extends Error {
  readonly loss: number;
  readonly reconstructionLoss: number;
  readonly klLoss: number;
  readonly continueLoss: number;

  // No TS parameter-property shorthand: Node runs these .ts files with
  // type-stripping only (no real compile step, package.json's "test"
  // script), and `constructor(readonly x: number)` isn't valid stripped
  // JavaScript — same constraint `src/env/types.ts`'s `Action` const-object
  // comment documents for `enum`.
  constructor(loss: number, reconstructionLoss: number, klLoss: number, continueLoss: number) {
    super(
      `WorldModel.step(): non-finite loss (loss=${loss}, reconstructionLoss=${reconstructionLoss}, ` +
        `klLoss=${klLoss}, continueLoss=${continueLoss}) — halting rather than risk silently ` +
        `continuing on a corrupted state.`,
    );
    this.name = "WorldModelNaNError";
    this.loss = loss;
    this.reconstructionLoss = reconstructionLoss;
    this.klLoss = klLoss;
    this.continueLoss = continueLoss;
  }
}

export interface WorldModelStepResult {
  /** reconstructionLoss + klBalancedLoss's `total` (recon + dyn+rep) + continueLoss, as a plain number. */
  loss: number;
  /** This step's reconstructionLoss alone, as a plain number — see docs/explainers/0006. */
  reconstructionLoss: number;
  /** This step's klBalancedLoss.total alone, as a plain number. */
  klLoss: number;
  /** This step's continueLoss alone, as a plain number — see docs/explainers/0011. */
  continueLoss: number;
}

/**
 * Wraps one `RSSMCell`, one `ObservationDecoder`, and one `ContinueHead` with
 * a shared optimizer and the single persistent `RSSMState` a rollout's
 * recurrence carries across an episode — proposal `0001` Arm-A's per-agent
 * world model. See docs/explainers/0005-world-model-rollout-wiring.md for the
 * rollout-wiring design (why `step()` always advances state but only
 * sometimes trains, the tensor-lifecycle contract with `RSSMCell`, and the
 * BPTT-horizon-1 amendment), docs/explainers/0006-observation-reconstruction-loss.md
 * for the decoder/reconstruction-loss piece, and
 * docs/explainers/0011-continue-termination-head.md for the continuation head.
 */
export class WorldModel {
  readonly cell: RSSMCell;
  readonly decoder: ObservationDecoder;
  readonly continueHead: ContinueHead;
  private readonly optimizer: tf.Optimizer;
  private readonly lossConfig: KLBalancedLossConfig;
  private readonly trainableVars: tf.Variable[];
  private state: RSSMState;

  constructor(config: WorldModelConfig) {
    const rssmConfig = config.seed !== undefined ? { ...config.rssm, seed: deriveSeed(config.seed, 0) } : config.rssm;
    this.cell = new RSSMCell(rssmConfig);
    this.decoder = new ObservationDecoder({
      observationSize: config.observationSize,
      ...(config.seed !== undefined && { seed: deriveSeed(config.seed, 1) }),
    });
    this.continueHead = new ContinueHead({
      ...(config.seed !== undefined && { seed: deriveSeed(config.seed, 2) }),
    });
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
      const posterior = this.cell.posterior(
        deterministic,
        tf.zeros([1, config.observationSize]) as tf.Tensor2D,
        warmupRng,
      );
      this.decoder.decode(deterministic, posterior.sample);
      this.continueHead.predict(deterministic, posterior.sample);
    });
    this.trainableVars = [
      ...this.cell.trainableWeights(),
      ...this.decoder.trainableWeights(),
      ...this.continueHead.trainableWeights(),
    ];
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
   * posterior(h_t, observation); predicted_o_t = decoder(h_t, z_t); loss =
   * reconstructionLoss(predicted_o_t, o_t) + klBalancedLoss(prior(h_t), that
   * posterior).total + continueLoss(continueHead(h_t, z_t), target) — the
   * last term added by docs/explainers/0011 (`done` is this transition's
   * env-reported outcome, `StepResult.done` polarity: target is `0` when
   * `done`, `1` otherwise). When `train` is true, applies one Adam step
   * toward `loss` before advancing; when false, still advances state and
   * returns `loss` (for post-freeze prediction-error tracking, per proposal
   * 0001) but leaves weights untouched. On success, disposes the previous
   * state and the observation tensor. On a throw from `forward()` (e.g. a
   * shape mismatch), `this.state` is left exactly as it was — `prevState` is
   * untouched, remains `this.state`, and stays usable — and only what
   * `forward()` itself allocated is cleaned up: the observation tensor plus,
   * if the throw happened after the `tf.keep()` calls below (e.g. inside
   * `decoder.decode()`/`reconstructionLoss()`), the two already-escaped
   * next-state tensors (PR #40 review: an earlier version disposed
   * `prevState`'s tensors unconditionally in a `finally`, which on a throw
   * left `this.state` — still pointing at `prevState`, since it's only
   * reassigned below on success — referencing disposed tensors).
   */
  step(action: Action, observation: Observation, rng: Rng, train: boolean, done: boolean): WorldModelStepResult {
    const prevState = this.state;
    const observationTensor = tf.tensor2d([observation]);
    const continueTargetTensor = tf.tensor2d([[done ? 0 : 1]]);

    let nextDeterministic: tf.Tensor2D | undefined;
    let nextStochastic: tf.Tensor2D | undefined;
    let reconstructionLossValue!: number;
    let klLossValue!: number;
    let continueLossValue!: number;

    // tf.variableGrads(f, ...) internally wraps `f` in its own tf.tidy
    // (tfjs's Engine.gradients: `this.tidy('forward', f)`) and disposes
    // everything `f` created except its returned scalar as soon as `f`
    // returns — before control ever comes back to this function. So the two
    // tensors that must outlive this call (the next RSSMState) have to be
    // tf.keep()'d from *inside* forward(), not after variableGrads returns —
    // by then it's too late, they're already disposed. This is true in the
    // train=false branch too (forward() called directly, no variableGrads),
    // so keeping unconditionally inside forward() covers both. The
    // reconstruction/KL/continue breakdown values are read out as plain
    // numbers here too (arraySync doesn't disturb the gradient tape), since
    // only the combined scalar below is what tf.variableGrads differentiates.
    const forward = (): tf.Scalar => {
      const deterministic = this.cell.step(prevState, [action]);
      const priorDist = this.cell.prior(deterministic, { rng });
      const posteriorDist = this.cell.posterior(deterministic, observationTensor, { rng });
      nextDeterministic = deterministic;
      nextStochastic = posteriorDist.sample;
      tf.keep(nextDeterministic);
      tf.keep(nextStochastic);

      const predictedObservation = this.decoder.decode(deterministic, nextStochastic);
      const recon = reconstructionLoss(predictedObservation, observationTensor);
      const kl = klBalancedLoss(priorDist, posteriorDist, this.lossConfig).total;
      const continueLogit = this.continueHead.predict(deterministic, nextStochastic);
      const cont = continueLoss(continueLogit, continueTargetTensor);
      reconstructionLossValue = recon.arraySync() as number;
      klLossValue = kl.arraySync() as number;
      continueLossValue = cont.arraySync() as number;
      return tf.add(tf.add(recon, kl), cont) as tf.Scalar;
    };

    // The whole tf.variableGrads call (forward *and* backward) runs inside
    // this one outer tf.tidy — not a tidy wrapped around `forward` alone,
    // which would dispose its intermediates before backward pass runs (see
    // RSSMCell's class doc comment and docs/explainers/0005). Everything
    // else forward()/klBalancedLoss/reconstructionLoss create — priorDist's
    // probs/sample, posteriorDist's probs, the decoder's reconstruction, the
    // KL loss's own dyn/rep components, RSSMCell's internal concat/dense
    // intermediates — gets disposed once this tidy returns (or, for the
    // train=true path, once variableGrads' own internal tidy calls return,
    // which happens first).
    let lossValue: number;
    try {
      lossValue = tf.tidy(() => {
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
    } catch (err) {
      // forward() threw before returning. this.state still equals prevState
      // (never reassigned below) and prevState's tensors were never
      // disposed, so the model stays usable. Only clean up what forward()
      // itself created for this failed attempt: the observation tensor, the
      // continuation-target tensor (same outside-forward()-but-consumed-by-it
      // lifecycle as observationTensor), and — if the throw happened after
      // the tf.keep() calls above, e.g. from decoder.decode() or
      // reconstructionLoss() — the two next-state tensors that already
      // escaped the inner tf.tidy and would otherwise leak (nothing else
      // will ever reference or dispose them).
      observationTensor.dispose();
      continueTargetTensor.dispose();
      nextDeterministic?.dispose();
      nextStochastic?.dispose();
      throw err;
    }

    if (
      !Number.isFinite(lossValue) ||
      !Number.isFinite(reconstructionLossValue) ||
      !Number.isFinite(klLossValue) ||
      !Number.isFinite(continueLossValue)
    ) {
      // forward() returned normally — this isn't the catch block's "threw
      // before completing" case, it's a value that computed successfully
      // and is NaN/±Infinity. When train, tf.variableGrads already called
      // applyGradients above; that update cannot be undone from here. What
      // this check prevents is compounding: without it, this.state would be
      // reassigned to next{Deterministic,Stochastic} — themselves computed
      // from the same non-finite forward pass — and every later step() call
      // would keep producing more non-finite losses that read to a caller as
      // ordinary (if extreme) numbers, e.g. surfacing as a spurious
      // drift-attributable-error reading instead of the training collapse it
      // actually is. Throwing here surfaces the failure at its origin
      // instead. Same state-preservation contract as the catch block above —
      // this.state is never reassigned below, so prevState stays exactly as
      // it was and remains usable — and only what this forward() pass itself
      // allocated and tf.keep()'d gets disposed here.
      observationTensor.dispose();
      continueTargetTensor.dispose();
      nextDeterministic!.dispose();
      nextStochastic!.dispose();
      throw new WorldModelNaNError(lossValue, reconstructionLossValue, klLossValue, continueLossValue);
    }

    observationTensor.dispose();
    continueTargetTensor.dispose();
    prevState.deterministic.dispose();
    prevState.stochastic.dispose();
    this.state = { deterministic: nextDeterministic!, stochastic: nextStochastic! };
    return {
      loss: lossValue,
      reconstructionLoss: reconstructionLossValue,
      klLoss: klLossValue,
      continueLoss: continueLossValue,
    };
  }
}
