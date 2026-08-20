import type { EpisodeStepRecord } from "./freeze.ts";

/**
 * Extracts one agent's world-model loss at every post-freeze step ("steps
 * since freeze", 0-indexed by array position — not by absolute env `step`)
 * from a completed episode's records — proposal 0001's "frozen-agent
 * one-step prediction error ... on newly collected transitions over the
 * following steps" (docs/proposals/0001-direct-nonstationarity-measurement.md).
 *
 * `records` is expected in ascending `step` order, matching `runEpisode`'s
 * output (src/experiment/freeze.ts) — this is not re-sorted.
 *
 * Throws if `freezeStep` never occurs in `records` (e.g. set past the
 * episode's horizon — an empty result would otherwise silently look like a
 * valid zero-length series to a caller), or if `agentIndex`'s
 * `worldModelLoss` entry is `undefined` at any post-freeze step (no
 * `WorldModel` was wired in for that agent for this episode — see
 * docs/explainers/0005-world-model-rollout-wiring.md).
 */
export function postFreezeLossSeries(
  records: EpisodeStepRecord[],
  freezeStep: number,
  agentIndex: number,
): number[] {
  const postFreeze = records.filter((record) => record.step >= freezeStep);
  if (postFreeze.length === 0) {
    throw new Error(`freezeStep ${freezeStep} never occurs in records (episode horizon too short)`);
  }
  return postFreeze.map((record) => {
    const loss = record.worldModelLoss[agentIndex];
    if (loss === undefined) {
      throw new Error(
        `worldModelLoss[${agentIndex}] is undefined at step ${record.step} — ` +
          "no WorldModel was wired in for this agent for this episode",
      );
    }
    return loss;
  });
}

/**
 * Proposal 0001's primary metric: drift-attributable world-model prediction
 * error. The elementwise difference (frozen-agent post-freeze loss under the
 * freeze *intervention*, partner still training) minus (frozen-agent
 * post-freeze loss under the both-frozen *control*, per
 * `postFreezeLossSeries` above for each), aligned by steps-since-freeze
 * (array index — index 0 is each run's first post-freeze step), not by
 * absolute env step. `intervention` and `control` come from two separate
 * `runEpisode` calls (ordinarily different seeds, per the milestone's 3-seed
 * validation design), so their absolute step numbering only needs to agree
 * on `freezeStep`, not on matching env states past it — see
 * docs/explainers/0007-drift-attributable-error-metric.md for why this
 * alignment, not a raw per-episode trajectory pairing, is the correct
 * comparison, and for how this feeds the milestone's two-sided gate ((a)
 * control stays flat, (b) intervention rises measurably above it,
 * proposal 0001 "L2 promotion request").
 *
 * Requires equal-length inputs. A length mismatch means the two runs used a
 * different horizon or `freezeStep` — every element after the shorter run
 * ends would silently compare unrelated steps-since-freeze offsets, which is
 * a caller bug to surface, not a value to paper over by truncating.
 */
/**
 * The instrument's power metric, per PR #46's review (@SakkarinKt, 2026-08-13): a per-seed count
 * of post-freeze steps at which `control` and `intervention` picked different joint actions,
 * aligned by steps-since-freeze (array index — index 0 is each run's first post-freeze step), not
 * by absolute env step, matching `driftAttributableError`'s alignment convention above.
 *
 * A `diffMean` of exactly `0` (as seed 1003 showed in the 2026-08-13 paired-init validation) is
 * ambiguous on its own — it's consistent both with "the freeze intervention has no effect at this
 * horizon" and with "the post-freeze window was too short/unexplored for the two conditions'
 * policies to ever pick different actions in the first place," which would make a `0` diffMean
 * uninformative rather than a null result. `divergentSteps === 0` directly distinguishes the
 * latter case: if no post-freeze action ever differed, no downstream loss *could* have differed
 * either, independent of whether the freeze mechanism itself works.
 *
 * Requires equal post-freeze step counts (same `freezeStep` and horizon in both runs, and same
 * `NUM_AGENTS`-length `actions` per step) — surfaced as a thrown error rather than truncated, same
 * rationale as `driftAttributableError`.
 */
export function postFreezeActionDivergenceCount(
  control: EpisodeStepRecord[],
  intervention: EpisodeStepRecord[],
  freezeStep: number,
): { postFreezeSteps: number; divergentSteps: number } {
  const postFreeze = (records: EpisodeStepRecord[]) => records.filter((record) => record.step >= freezeStep);
  const c = postFreeze(control);
  const i = postFreeze(intervention);
  if (c.length !== i.length) {
    throw new Error(
      `control has ${c.length} post-freeze steps, intervention has ${i.length} — same freezeStep and horizon in both runs`,
    );
  }
  if (c.length === 0) {
    throw new Error(`freezeStep ${freezeStep} never occurs in control's records (episode horizon too short)`);
  }

  let divergentSteps = 0;
  for (let idx = 0; idx < c.length; idx++) {
    const cActions = c[idx]!.actions;
    const iActions = i[idx]!.actions;
    if (cActions.length !== iActions.length) {
      throw new Error(
        `step-since-freeze ${idx}: control has ${cActions.length} agents' actions, intervention has ${iActions.length}`,
      );
    }
    if (JSON.stringify(cActions) !== JSON.stringify(iActions)) {
      divergentSteps++;
    }
  }

  return { postFreezeSteps: c.length, divergentSteps };
}

export function driftAttributableError(interventionLosses: number[], controlLosses: number[]): number[] {
  if (interventionLosses.length !== controlLosses.length) {
    throw new Error(
      `interventionLosses (length ${interventionLosses.length}) and controlLosses ` +
        `(length ${controlLosses.length}) must be the same length — same freezeStep and horizon in both runs`,
    );
  }
  return interventionLosses.map((loss, i) => {
    const control = controlLosses[i];
    // Unreachable given the length check above (every i < interventionLosses.length
    // is also < controlLosses.length) — noUncheckedIndexedAccess can't see that
    // correlation, so this narrows explicitly rather than asserting past it.
    if (control === undefined) {
      throw new Error(`internal error: controlLosses[${i}] is undefined despite the equal-length check`);
    }
    return loss - control;
  });
}
