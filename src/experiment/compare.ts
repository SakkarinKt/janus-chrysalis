import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { listFiles } from "./provenance.ts";

/**
 * Comparing a re-run's summary CSVs against a reference. Used by `scripts/repro.ts` and by
 * `repro` cards. The unit of comparison is the committed `*.summary.csv`: per-step telemetry is
 * gitignored, so a summary is the only output a later reader can check.
 */

export interface CsvMismatch {
  file: string;
  /** 1-based line number, header included. 0 for a file-level problem. */
  line: number;
  /** 1-based column number. 0 for a line- or file-level problem. */
  column: number;
  detail: string;
}

export interface Comparison {
  /** Every compared file is byte-for-byte equal to its reference. */
  identical: boolean;
  /** Largest numeric difference found, in the units of `difference()`. 0 when identical. */
  maxDifference: number;
  /** Structural problems and numeric differences above `tolerance`. Capped at `MAX_REPORTED`. */
  mismatches: CsvMismatch[];
  /** True count of mismatches, which may exceed `mismatches.length`. */
  mismatchCount: number;
  filesCompared: number;
  tolerance: number;
  /** No structural problem and no numeric difference above `tolerance`. */
  match: boolean;
}

const MAX_REPORTED = 10;

/**
 * `|a − b| / max(1, |a|, |b|)`: an absolute difference for values below 1 in magnitude and a
 * relative one above. A purely relative measure would call 1e-9 vs 2e-9 a 50% difference; a
 * purely absolute one would wave through a 1% change in a loss of 100.
 */
export function difference(a: number, b: number): number {
  return Math.abs(a - b) / Math.max(1, Math.abs(a), Math.abs(b));
}

/** A cell counts as numeric only if it is a complete finite number, so `""` and `"true"` do not. */
function numericCell(cell: string): number | null {
  if (cell.trim() === "") {
    return null;
  }
  const value = Number(cell);
  return Number.isFinite(value) ? value : null;
}

/**
 * Compares two CSV texts cell by cell. The summaries this project writes are plain
 * comma-joined rows with no quoting, and this parser assumes exactly that.
 */
export function compareCsv(file: string, expected: string, actual: string, tolerance: number): Comparison {
  const mismatches: CsvMismatch[] = [];
  let mismatchCount = 0;
  let maxDifference = 0;
  const report = (line: number, column: number, detail: string): void => {
    mismatchCount += 1;
    if (mismatches.length < MAX_REPORTED) {
      mismatches.push({ file, line, column, detail });
    }
  };

  if (expected !== actual) {
    const expectedLines = expected.split("\n");
    const actualLines = actual.split("\n");
    if (expectedLines.length !== actualLines.length) {
      report(0, 0, `expected ${expectedLines.length} lines, got ${actualLines.length}`);
    }
    const lineCount = Math.min(expectedLines.length, actualLines.length);
    for (let index = 0; index < lineCount; index++) {
      const expectedLine = expectedLines[index] ?? "";
      const actualLine = actualLines[index] ?? "";
      if (expectedLine === actualLine) {
        continue;
      }
      const expectedCells = expectedLine.split(",");
      const actualCells = actualLine.split(",");
      if (expectedCells.length !== actualCells.length) {
        report(index + 1, 0, `expected ${expectedCells.length} columns, got ${actualCells.length}`);
        continue;
      }
      expectedCells.forEach((expectedCell, column) => {
        const actualCell = actualCells[column] ?? "";
        if (expectedCell === actualCell) {
          return;
        }
        const expectedNumber = numericCell(expectedCell);
        const actualNumber = numericCell(actualCell);
        if (expectedNumber === null || actualNumber === null) {
          report(index + 1, column + 1, `expected "${expectedCell}", got "${actualCell}"`);
          return;
        }
        const delta = difference(expectedNumber, actualNumber);
        maxDifference = Math.max(maxDifference, delta);
        if (delta > tolerance) {
          report(index + 1, column + 1, `expected ${expectedCell}, got ${actualCell} (difference ${delta.toExponential(3)})`);
        }
      });
    }
  }

  return {
    identical: expected === actual,
    maxDifference,
    mismatches,
    mismatchCount,
    filesCompared: 1,
    tolerance,
    match: mismatchCount === 0,
  };
}

/** Folds per-file comparisons into one. */
export function mergeComparisons(parts: readonly Comparison[], tolerance: number): Comparison {
  const mismatches = parts.flatMap((part) => part.mismatches).slice(0, MAX_REPORTED);
  const mismatchCount = parts.reduce((sum, part) => sum + part.mismatchCount, 0);
  return {
    identical: parts.every((part) => part.identical),
    maxDifference: parts.reduce((max, part) => Math.max(max, part.maxDifference), 0),
    mismatches,
    mismatchCount,
    filesCompared: parts.reduce((sum, part) => sum + part.filesCompared, 0),
    tolerance,
    match: mismatchCount === 0,
  };
}

const SUMMARY_SUFFIX = ".summary.csv";

/**
 * Compares every `*.summary.csv` under `expectedDir` with the file at the same relative path
 * under `actualDir`. A reference summary missing from the re-run is a mismatch; an extra file in
 * the re-run is not (a newer script may write more). Finding no reference summaries at all is a
 * mismatch too: a comparison that compared nothing must not report a match.
 */
export function compareSummaryDirs(expectedDir: string, actualDir: string, tolerance: number): Comparison {
  const summaries = listFiles(expectedDir).filter((path) => path.endsWith(SUMMARY_SUFFIX));
  if (summaries.length === 0) {
    return {
      identical: false,
      maxDifference: 0,
      mismatches: [{ file: expectedDir, line: 0, column: 0, detail: `no *${SUMMARY_SUFFIX} files to compare` }],
      mismatchCount: 1,
      filesCompared: 0,
      tolerance,
      match: false,
    };
  }
  const parts = summaries.map((path): Comparison => {
    const actualPath = join(actualDir, path);
    if (!existsSync(actualPath)) {
      return {
        identical: false,
        maxDifference: 0,
        mismatches: [{ file: path, line: 0, column: 0, detail: "missing from the re-run" }],
        mismatchCount: 1,
        filesCompared: 1,
        tolerance,
        match: false,
      };
    }
    return compareCsv(path, readFileSync(join(expectedDir, path), "utf8"), readFileSync(actualPath, "utf8"), tolerance);
  });
  return mergeComparisons(parts, tolerance);
}
