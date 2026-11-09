/**
 * Card NNNN — <title>. Run with `npm run exp -- NNNN`.
 *
 * Copy this directory to `experiments/cards/proposed/NNNN-short-slug/`, fill in `card.json`
 * first, then write this file to do exactly what the card says. `../README.md` has the rules.
 */
import { writeFileSync } from "node:fs";
import { join } from "node:path";
import type { RunContext, RunResult } from "../../../src/experiment/card.ts";

/** A threshold from the card, or a thrown error naming the one that is missing. */
function threshold(context: RunContext, name: string): number {
  const value = context.card.decisionRule.thresholds[name];
  if (value === undefined) {
    throw new Error(`card decisionRule.thresholds.${name} is missing`);
  }
  return value;
}

export function run(context: RunContext): RunResult {
  const required = threshold(context, "threshold");
  const k = threshold(context, "k");

  // One row per seed. Replace `measure` with the experiment; keep the shape.
  const measure = (seed: number): number => seed * 0;
  const rows = context.seeds.map((seed) => ({ seed, value: measure(seed) }));
  for (const row of rows) {
    context.log(`seed ${row.seed}: ${row.value}`);
  }

  // Summaries are committed and are what `npm run repro` compares. Full precision, no rounding.
  writeFileSync(
    join(context.outDir, "per-seed.summary.csv"),
    `seed,value\n${rows.map((row) => `${row.seed},${row.value}`).join("\n")}\n`,
  );

  const passing = rows.filter((row) => row.value >= required).length;
  return {
    primary: { name: context.card.primaryMetric, value: passing },
    verdict: passing >= k ? "pass" : "fail",
    resultSummary: { passing, of: rows.length, perSeed: rows },
  };
}
