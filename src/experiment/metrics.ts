import type { EpisodeStepRecord } from "./freeze.ts";
import type { Observation } from "../env/types.ts";

/**
 * Extracts one agent's world-model loss at every post-freeze step ("steps
 * since freeze", 0-indexed by array position — not by absolute env `step`)
 * from a completed episode's records — proposal 0001's "frozen-agent
 * one-step prediction error ... on newly collected transitions over the
 * following steps" (docs/proposals/0001-direct-nonstationarity-measurement.md).
 *
 * Reads `reconstructionLoss + klLoss` from `EpisodeStepRecord.worldModelLossBreakdown`,
 * **excluding `continueLoss`** — not the summed `worldModelLoss` total. Per PR #63's
 * review (@SakkarinKt, 2026-09-02): since the continue head landed (docs/explainers/0011),
 * `worldModelLoss` mixes in `continueLoss`, and that term doesn't cancel between conditions —
 * post-freeze the frozen arm's continue term goes static while control's keeps training, against a
 * target that is near-constant `1` for the whole post-freeze window (it flips to `0` only at the
 * episode's very last step) — so it would contaminate this instrument's comparison. See
 * docs/explainers/0007's addendum.
 *
 * `records` is expected in ascending `step` order, matching `runEpisode`'s
 * output (src/experiment/freeze.ts) — this is not re-sorted.
 *
 * Throws if `freezeStep` never occurs in `records` (e.g. set past the
 * episode's horizon — an empty result would otherwise silently look like a
 * valid zero-length series to a caller), or if `agentIndex`'s
 * `worldModelLossBreakdown` entry is `undefined` at any post-freeze step (no
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
    const breakdown = record.worldModelLossBreakdown[agentIndex];
    if (breakdown === undefined) {
      throw new Error(
        `worldModelLossBreakdown[${agentIndex}] is undefined at step ${record.step} — ` +
          "no WorldModel was wired in for this agent for this episode",
      );
    }
    return breakdown.reconstructionLoss + breakdown.klLoss;
  });
}

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

/**
 * A single agent's own observation-divergence count, per PR #47's review (@SakkarinKt,
 * 2026-08-20 merge comment, "land an agent-0 observation-divergence (or partner-visibility)
 * counter beside this one"): a per-seed count of post-freeze steps at which `agentIndex`'s
 * post-step observation (`EpisodeStepRecord.nextObservations[agentIndex]` — the observation
 * `WorldModel.step()` is actually evaluated against, per `runEpisode` in
 * src/experiment/freeze.ts) differs between `control` and `intervention`, aligned by
 * steps-since-freeze (array index), same convention as `postFreezeActionDivergenceCount` above.
 *
 * `postFreezeActionDivergenceCount` showed that a post-freeze action difference at some step
 * doesn't imply the *frozen* agent's own loss could have seen anything different — a joint
 * divergence driven entirely by the other agent's action, with `agentIndex` outside
 * `viewRadius` (src/env/gridworld.ts's `relativeEntry`) of the partner both before and after,
 * leaves `agentIndex`'s observation identical across control and intervention despite the
 * actions differing. This metric isolates that — with one caveat, per the PR #48 review:
 * `WorldModel` threads recurrent state across the rollout, so an observation divergence at step
 * `s` can leave the loss diverging at steps after `s` even once observations resync (contaminated
 * through carried-forward state), which means "divergent at step t" is *not* itself a necessary
 * condition for `postFreezeLossSeries` to differ at step t — only "divergent at or before step t"
 * is. The claim that survives per-step: an agent whose observation never diverges across the
 * *entire* post-freeze window (`divergentSteps === 0`) has no step at which its world-model loss
 * *could* diverge, independent of whether its own actions ever diverged.
 *
 * Requires equal post-freeze step counts (same `freezeStep` and horizon in both runs) — surfaced
 * as a thrown error rather than truncated, same rationale as `postFreezeActionDivergenceCount`.
 */
export function postFreezeObservationDivergenceCount(
  control: EpisodeStepRecord[],
  intervention: EpisodeStepRecord[],
  freezeStep: number,
  agentIndex: number,
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
    const cObs = c[idx]!.nextObservations[agentIndex];
    const iObs = i[idx]!.nextObservations[agentIndex];
    if (cObs === undefined || iObs === undefined) {
      throw new Error(`nextObservations[${agentIndex}] is undefined at step-since-freeze ${idx}`);
    }
    if (JSON.stringify(cObs) !== JSON.stringify(iObs)) {
      divergentSteps++;
    }
  }

  return { postFreezeSteps: c.length, divergentSteps };
}

/**
 * Per-seed count of post-freeze steps at which the partner is visible to
 * `agentIndex` — the observation's otherAgent-visible flag (`relativeEntry`
 * in src/env/gridworld.ts, always the third-from-last element of the flat
 * `Observation` vector: `[..., otherAgent_visible, otherAgent_dx,
 * otherAgent_dy]`, src/env/types.ts) — in *either* `control` or
 * `intervention`. This is the denominator the PR #48/#49 reviews computed
 * informally ("partner-visible steps") to read how much of the post-freeze
 * window even offers `postFreezeObservationDivergenceCount` and
 * `driftAttributableError` anything to detect, independent of whether
 * control and intervention actually differ. Landed as its own metric per
 * the PR #49 review (@SakkarinKt, 2026-08-22 merge comment): "landing the
 * standalone visibility tally from #48 would have sharpened this."
 *
 * Union ("in either"), not intersection ("in both") or "in control alone":
 * a step where the partner is visible under one condition but not the
 * other is itself a step where `postFreezeObservationDivergenceCount` could
 * register a difference, so the union is the right denominator for "how
 * many steps had anything to see."
 *
 * Requires equal post-freeze step counts (same `freezeStep` and horizon in
 * both runs) — surfaced as a thrown error rather than truncated, same
 * rationale as `postFreezeActionDivergenceCount`.
 */
export function postFreezePartnerVisibleCount(
  control: EpisodeStepRecord[],
  intervention: EpisodeStepRecord[],
  freezeStep: number,
  agentIndex: number,
): { postFreezeSteps: number; visibleSteps: number } {
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

  const isPartnerVisible = (obs: Observation): boolean => {
    const visibleFlag = obs[obs.length - 3];
    if (visibleFlag === undefined) {
      throw new Error(`observation too short (length ${obs.length}) to contain an otherAgent-visible flag`);
    }
    return visibleFlag === 1;
  };

  let visibleSteps = 0;
  for (let idx = 0; idx < c.length; idx++) {
    const cObs = c[idx]!.nextObservations[agentIndex];
    const iObs = i[idx]!.nextObservations[agentIndex];
    if (cObs === undefined || iObs === undefined) {
      throw new Error(`nextObservations[${agentIndex}] is undefined at step-since-freeze ${idx}`);
    }
    if (isPartnerVisible(cObs) || isPartnerVisible(iObs)) {
      visibleSteps++;
    }
  }

  return { postFreezeSteps: c.length, visibleSteps };
}

/**
 * The landmark analogue of `postFreezePartnerVisibleCount` above, per the PR #50 review
 * (@SakkarinKt, 2026-08-23 merge comment): "`viewRadius` gates landmarks too ... the sweep axis
 * moves partner visibility and landmark observability together, and `postFreezePartnerVisibleCount`
 * only tallies the partner half." Same union-over-control-and-intervention convention, but checks
 * every landmark's visible flag rather than the partner's: a step counts as `visibleSteps` if
 * *any* of `numLandmarks` landmarks is visible to `agentIndex` in `control` or `intervention` (see
 * the flat `Observation` layout, src/env/types.ts: `[selfX, selfY, landmark_0_visible, ...,
 * landmark_{numLandmarks-1}_visible, ..., otherAgent_visible, ...]` — each landmark's visible flag
 * sits at index `2 + 3*k`).
 *
 * With `CooperativeGridWorld.setPartnerViewRadius()` (src/env/gridworld.ts, PR #50 review
 * follow-up, 2026-08-24) decoupling the partner gate from the landmark gate, a run that only calls
 * `setPartnerViewRadius()` post-freeze should see this tally track position drift alone, not the
 * radius sweep directly — unlike `postFreezePartnerVisibleCount`, which is expected to respond to
 * the swept radius by construction.
 *
 * Requires equal post-freeze step counts (same `freezeStep` and horizon in both runs) — surfaced
 * as a thrown error rather than truncated, same rationale as `postFreezeActionDivergenceCount`.
 */
export function postFreezeLandmarkVisibleCount(
  control: EpisodeStepRecord[],
  intervention: EpisodeStepRecord[],
  freezeStep: number,
  agentIndex: number,
  numLandmarks: number,
): { postFreezeSteps: number; visibleSteps: number } {
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

  const isAnyLandmarkVisible = (obs: Observation): boolean => {
    for (let k = 0; k < numLandmarks; k++) {
      const visibleFlag = obs[2 + k * 3];
      if (visibleFlag === undefined) {
        throw new Error(`observation too short (length ${obs.length}) to contain landmark ${k}'s visible flag`);
      }
      if (visibleFlag === 1) {
        return true;
      }
    }
    return false;
  };

  let visibleSteps = 0;
  for (let idx = 0; idx < c.length; idx++) {
    const cObs = c[idx]!.nextObservations[agentIndex];
    const iObs = i[idx]!.nextObservations[agentIndex];
    if (cObs === undefined || iObs === undefined) {
      throw new Error(`nextObservations[${agentIndex}] is undefined at step-since-freeze ${idx}`);
    }
    if (isAnyLandmarkVisible(cObs) || isAnyLandmarkVisible(iObs)) {
      visibleSteps++;
    }
  }

  return { postFreezeSteps: c.length, visibleSteps };
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
