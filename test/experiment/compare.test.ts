import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { compareCsv, compareSummaryDirs, difference } from "../../src/experiment/compare.ts";

function dirWith(files: Record<string, string>): string {
  const dir = mkdtempSync(join(tmpdir(), "janus-compare-"));
  for (const [path, content] of Object.entries(files)) {
    mkdirSync(join(dir, path, ".."), { recursive: true });
    writeFileSync(join(dir, path), content);
  }
  return dir;
}

const CSV = "seed,agent,loss\n1001,0,2.5275290473302205\n1001,1,1.6510124254226684\n";

test("difference: absolute below magnitude 1, relative above", () => {
  assert.equal(difference(0.5, 0.25), 0.25);
  assert.equal(difference(1e-9, 2e-9), 1e-9);
  assert.equal(difference(100, 101), 1 / 101);
  assert.equal(difference(-3, -3), 0);
});

test("compareCsv: identical text is identical, matches at tolerance 0, and has zero difference", () => {
  const result = compareCsv("a.summary.csv", CSV, CSV, 0);
  assert.deepEqual(
    { identical: result.identical, match: result.match, maxDifference: result.maxDifference, mismatchCount: result.mismatchCount },
    { identical: true, match: true, maxDifference: 0, mismatchCount: 0 },
  );
});

test("compareCsv: a last-digit float difference fails at tolerance 0 and passes at 1e-9, and is located", () => {
  const drifted = CSV.replace("2.5275290473302205", "2.5275290473302210");
  const strict = compareCsv("a.summary.csv", CSV, drifted, 0);
  assert.equal(strict.identical, false);
  assert.equal(strict.match, false);
  assert.ok(strict.maxDifference > 0 && strict.maxDifference < 1e-12);
  assert.deepEqual(
    strict.mismatches.map((mismatch) => [mismatch.line, mismatch.column]),
    [[2, 3]],
  );

  const tolerant = compareCsv("a.summary.csv", CSV, drifted, 1e-9);
  assert.equal(tolerant.match, true);
  assert.equal(tolerant.identical, false, "a tolerated difference is still not byte-identical");
  assert.equal(tolerant.maxDifference, strict.maxDifference);
});

test("compareCsv: a different seeded stream is far outside any float tolerance", () => {
  const otherStream = CSV.replace("2.5275290473302205", "2.4911");
  assert.equal(compareCsv("a.summary.csv", CSV, otherStream, 1e-6).match, false);
});

test("compareCsv: non-numeric cells must be equal as text, whatever the tolerance", () => {
  const expected = "seed,pass\n1,true\n";
  const result = compareCsv("gate.summary.csv", expected, "seed,pass\n1,false\n", 1);
  assert.equal(result.match, false);
  assert.match(result.mismatches[0]?.detail ?? "", /expected "true", got "false"/);
  assert.equal(compareCsv("gate.summary.csv", expected, "seed,passed\n1,true\n", 1).match, false, "header change");
});

test("compareCsv: a missing row or column is a structural mismatch", () => {
  assert.match(compareCsv("a", CSV, "seed,agent,loss\n1001,0,2.5275290473302205\n", 1).mismatches[0]?.detail ?? "", /expected 4 lines, got 3/);
  assert.match(compareCsv("a", CSV, CSV.replace("1001,1,", "1001,"), 1).mismatches[0]?.detail ?? "", /expected 3 columns, got 2/);
});

test("compareCsv: reports at most 10 mismatches but counts them all", () => {
  const rows = Array.from({ length: 25 }, (_, index) => `${index},1`).join("\n");
  const changed = Array.from({ length: 25 }, (_, index) => `${index},2`).join("\n");
  const result = compareCsv("a", `i,v\n${rows}\n`, `i,v\n${changed}\n`, 0);
  assert.equal(result.mismatches.length, 10);
  assert.equal(result.mismatchCount, 25);
});

test("compareSummaryDirs: compares every reference *.summary.csv, including nested ones, and ignores other files", () => {
  const reference = dirWith({ "loss.summary.csv": CSV, "seed-1/gate.summary.csv": CSV, "seed-1/manifest.json": "{}", "notes.txt": "x" });
  const rerun = dirWith({ "loss.summary.csv": CSV, "seed-1/gate.summary.csv": CSV, "extra.summary.csv": "new,file\n" });
  const result = compareSummaryDirs(reference, rerun, 0);
  assert.equal(result.match, true);
  assert.equal(result.identical, true);
  assert.equal(result.filesCompared, 2);
  rmSync(reference, { recursive: true, force: true });
  rmSync(rerun, { recursive: true, force: true });
});

test("compareSummaryDirs: a reference summary missing from the re-run is a mismatch", () => {
  const reference = dirWith({ "loss.summary.csv": CSV, "gate.summary.csv": CSV });
  const rerun = dirWith({ "loss.summary.csv": CSV });
  const result = compareSummaryDirs(reference, rerun, 0);
  assert.equal(result.match, false);
  assert.deepEqual(result.mismatches.map((mismatch) => [mismatch.file, mismatch.detail]), [["gate.summary.csv", "missing from the re-run"]]);
  rmSync(reference, { recursive: true, force: true });
  rmSync(rerun, { recursive: true, force: true });
});

test("compareSummaryDirs: comparing nothing is a mismatch, never a match", () => {
  const reference = dirWith({ "manifest.json": "{}" });
  const rerun = dirWith({ "manifest.json": "{}" });
  const result = compareSummaryDirs(reference, rerun, 0);
  assert.equal(result.match, false);
  assert.equal(result.filesCompared, 0);
  rmSync(reference, { recursive: true, force: true });
  rmSync(rerun, { recursive: true, force: true });
});
