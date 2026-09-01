/**
 * Processes PR #54's review (@SakkarinKt, posted as an issue comment after merge). The review
 * found that 2026-08-26's radius-4-vs-radius-6 sign comparison mixed cells of very different
 * statistical power: seed 1008's radius-4 `diffMean` is exactly 0 because its
 * `postFreezePartnerVisibleCount` is 0/38 (the partner was never visible, so control and
 * intervention are bit-identical *by construction* — a zero-power cell, not evidence about sign),
 * and two more seeds (1006: 2/38, 1003: 11/38) are similarly low-power at radius 4 while every seed
 * has 27-38/38 visibility at radius 6. The unfiltered "radius 4 looks nothing like radius 6"
 * conclusion (docs/proposals/0001-direct-nonstationarity-measurement.md's now-corrected 2026-08-27
 * update) overclaimed scope as a result.
 *
 * This is a pure re-analysis of already-collected, already-committed data — no new environment or
 * world-model training runs, so no `manifest.json` per seed (nothing new was run; this script reads
 * three prior runs' artifacts and writes one derived summary CSV for reproducibility, per the
 * review's "condition the sign analysis on partner visibility" directive).
 *
 *   node experiments/2026-08-27-power-aware-radius-analysis/analyze.ts
 *
 * Reads:
 *   - artifacts/2026-08-26-radius6-more-seeds/pooled-radius6.summary.csv (diffMean, partnerVisibleSteps @ r6, all 9 seeds)
 *   - artifacts/2026-08-26-radius4-matched-seeds/radius4-vs-radius6.summary.csv (diffMean @ r4 and r6, all 9 seeds)
 *   - artifacts/2026-08-26-radius4-matched-seeds/seed-<seed>-intervention/manifest.json (partnerVisibleSteps @ r4, per seed — not in the CSV)
 *
 * Computes, for the full n=9 and for the subset with partner-visible >=27/38 at BOTH radii (r6 is
 * already >=27/38 for every seed, so this reduces to filtering on r4's visibility, matching the
 * review's own threshold): sign split, one-sided binomial p for same-sign-at-both-radii count, and
 * the continuous paired t-test on (diffMean(r4) - diffMean(r6)).
 */
import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { binomialUpperTail } from "../../src/experiment/statistics.ts";

const RUN_ID = "2026-08-27-power-aware-radius-analysis";
const VISIBILITY_THRESHOLD = 27; // Matches the PR #54 review's own choice, out of 38 post-freeze steps.

const artifactsDir = fileURLToPath(new URL(`../../artifacts/${RUN_ID}/`, import.meta.url));
const radius6CsvPath = fileURLToPath(new URL("../../artifacts/2026-08-26-radius6-more-seeds/pooled-radius6.summary.csv", import.meta.url));
const radius4CsvPath = fileURLToPath(new URL("../../artifacts/2026-08-26-radius4-matched-seeds/radius4-vs-radius6.summary.csv", import.meta.url));
const radius4ManifestDir = fileURLToPath(new URL("../../artifacts/2026-08-26-radius4-matched-seeds/", import.meta.url));

interface Row {
  seed: number;
  diffMeanR4: number;
  diffMeanR6: number;
  partnerVisibleR4: number;
  partnerVisibleR6: number;
}

function mean(values: number[]): number {
  return values.reduce((a, b) => a + b, 0) / values.length;
}

function sampleStddev(values: number[]): number {
  const m = mean(values);
  const sumSquares = values.reduce((acc, v) => acc + (v - m) ** 2, 0);
  return Math.sqrt(sumSquares / (values.length - 1));
}

function readRows(): Row[] {
  const r6Csv = readFileSync(radius6CsvPath, "utf8").trim().split("\n");
  const r6Header = r6Csv[0]!.split(",");
  const r6SeedIdx = r6Header.indexOf("seed");
  const r6VisibleIdx = r6Header.indexOf("partnerVisibleSteps");
  const r6Visibility = new Map<number, number>();
  for (const line of r6Csv.slice(1)) {
    const cols = line.split(",");
    r6Visibility.set(Number(cols[r6SeedIdx]), Number(cols[r6VisibleIdx]));
  }

  const r4Csv = readFileSync(radius4CsvPath, "utf8").trim().split("\n");
  const r4Header = r4Csv[0]!.split(",");
  const seedIdx = r4Header.indexOf("seed");
  const diffMeanR4Idx = r4Header.indexOf("diffMeanR4");
  const diffMeanR6Idx = r4Header.indexOf("diffMeanR6");

  return r4Csv.slice(1).map((line) => {
    const cols = line.split(",");
    const seed = Number(cols[seedIdx]);
    const manifest = JSON.parse(readFileSync(`${radius4ManifestDir}seed-${seed}-intervention/manifest.json`, "utf8"));
    const partnerFinding = manifest.findings.find((f: { postFreezePartnerVisibleCount?: unknown }) => f.postFreezePartnerVisibleCount)!
      .postFreezePartnerVisibleCount as { visibleSteps: number };
    return {
      seed,
      diffMeanR4: Number(cols[diffMeanR4Idx]),
      diffMeanR6: Number(cols[diffMeanR6Idx]),
      partnerVisibleR4: partnerFinding.visibleSteps,
      partnerVisibleR6: r6Visibility.get(seed)!,
    };
  });
}

function report(label: string, rows: Row[]): void {
  const n = rows.length;
  const negR4 = rows.filter((r) => r.diffMeanR4 < 0).length;
  const posR4 = rows.filter((r) => r.diffMeanR4 > 0).length;
  const sameSign = rows.filter((r) => Math.sign(r.diffMeanR4) === Math.sign(r.diffMeanR6)).length;
  const paired = rows.map((r) => r.diffMeanR4 - r.diffMeanR6);
  const pairedMean = mean(paired);
  const pairedStddev = n > 1 ? sampleStddev(paired) : NaN;
  const t = n > 1 ? pairedMean / (pairedStddev / Math.sqrt(n)) : NaN;
  const p = binomialUpperTail(sameSign, n);

  console.log(`\n--- ${label} (n=${n}) ---`);
  console.log(`seeds: ${rows.map((r) => r.seed).join(", ")}`);
  console.log(`radius 4 sign split: ${negR4} negative, ${posR4} positive, ${n - negR4 - posR4} zero`);
  console.log(`same-sign-at-both-radii: ${sameSign}/${n} (one-sided binomial p=${p.toFixed(4)} under independent-coin-flip null)`);
  console.log(`paired diff (r4 - r6): mean=${pairedMean.toFixed(4)}, sample stddev=${pairedStddev.toFixed(4)}, t(${n - 1})=${t.toFixed(4)}`);
}

function main(): void {
  mkdirSync(artifactsDir, { recursive: true });
  const rows = readRows();

  report("full n=9 (unfiltered)", rows);

  const highPower = rows.filter((r) => r.partnerVisibleR4 >= VISIBILITY_THRESHOLD && r.partnerVisibleR6 >= VISIBILITY_THRESHOLD);
  report(`filtered to partner-visible >=${VISIBILITY_THRESHOLD}/38 at both radii`, highPower);

  console.log("\nseed | diffMean(r4) | diffMean(r6) | visible(r4) | visible(r6) | high power?");
  for (const r of rows) {
    const highPowerFlag = r.partnerVisibleR4 >= VISIBILITY_THRESHOLD && r.partnerVisibleR6 >= VISIBILITY_THRESHOLD;
    console.log(
      `${r.seed} | ${r.diffMeanR4.toFixed(4)} | ${r.diffMeanR6.toFixed(4)} | ${r.partnerVisibleR4}/38 | ${r.partnerVisibleR6}/38 | ${highPowerFlag}`,
    );
  }

  const csvHeader = "seed,diffMeanR4,diffMeanR6,partnerVisibleR4,partnerVisibleR6,highPower\n";
  const csvBody = rows
    .map((r) => {
      const highPowerFlag = r.partnerVisibleR4 >= VISIBILITY_THRESHOLD && r.partnerVisibleR6 >= VISIBILITY_THRESHOLD;
      return `${r.seed},${r.diffMeanR4},${r.diffMeanR6},${r.partnerVisibleR4},${r.partnerVisibleR6},${highPowerFlag}`;
    })
    .join("\n");
  writeFileSync(`${artifactsDir}power-annotated-radius4-vs-radius6.summary.csv`, csvHeader + csvBody + "\n");
}

main();
