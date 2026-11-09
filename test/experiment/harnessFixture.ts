import { execFileSync } from "node:child_process";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { Card } from "../../src/experiment/card.ts";

// Throwaway git repositories laid out like this one (experiments/cards/, experiments/ledger.jsonl,
// artifacts/), with cards whose run.ts needs nothing but Node. The harness is bookkeeping around
// git state and a child process, so its tests run it for real against these.

const GIT_ENV = {
  ...process.env,
  GIT_CONFIG_GLOBAL: "/dev/null",
  GIT_CONFIG_SYSTEM: "/dev/null",
  GIT_AUTHOR_NAME: "test",
  GIT_AUTHOR_EMAIL: "test@example.invalid",
  GIT_COMMITTER_NAME: "test",
  GIT_COMMITTER_EMAIL: "test@example.invalid",
};

export interface FixtureRepo {
  root: string;
  git: (...args: string[]) => string;
  /** Stages everything and commits. Returns the new commit's SHA. */
  commitAll: (message: string) => string;
  /** Writes `experiments/cards/<dirName>/{card.json,run.ts}`. Does not commit. */
  writeCard: (dirName: string, card: Partial<Card>, runSource?: string) => void;
  write: (relativePath: string, content: string) => void;
  dispose: () => void;
}

/**
 * A card run that writes one summary CSV (`seed,value` with value = seed × config.scale) and
 * passes when the total reaches `thresholds.minTotal`. Its resultSummary lists the dependency
 * ids it was handed, so a test can check they arrived.
 */
export const SCALE_RUN = `
import { writeFileSync } from "node:fs";
import { join } from "node:path";

export function run(context) {
  const scale = context.card.config.scale;
  const rows = context.seeds.map((seed) => seed + "," + seed * scale);
  writeFileSync(join(context.outDir, "values.summary.csv"), "seed,value\\n" + rows.join("\\n") + "\\n");
  writeFileSync(join(context.outDir, "telemetry.jsonl"), JSON.stringify({ seeds: context.seeds }) + "\\n");
  const total = context.seeds.reduce((sum, seed) => sum + seed * scale, 0);
  return {
    primary: { name: "total", value: total },
    verdict: total >= context.card.decisionRule.thresholds.minTotal ? "pass" : "fail",
    resultSummary: { total, dependencies: context.dependencies },
  };
}
`;

export const THROWING_RUN = `export function run() { throw new Error("boom"); }\n`;
export const HANGING_RUN = `export function run() { for (;;) {} }\n`;
export const MALFORMED_RUN = `export function run() { return { verdict: "maybe" }; }\n`;

export function validCard(overrides: Partial<Card> = {}): Card {
  return {
    id: "0001",
    title: "Fixture card",
    hypothesis: "The total reaches the threshold.",
    kind: "confirmatory",
    config: { scale: 2 },
    seeds: [1, 2, 3],
    primaryMetric: "total",
    secondaryMetrics: [],
    decisionRule: { text: "pass if total >= minTotal", thresholds: { minTotal: 10 } },
    onFail: "Nothing depends on this fixture.",
    dependsOn: [],
    budget: { maxWallMinutes: 1 },
    supersedes: null,
    ...overrides,
  };
}

export function makeFixtureRepo(): FixtureRepo {
  const root = mkdtempSync(join(tmpdir(), "janus-harness-"));
  const git = (...args: string[]): string => execFileSync("git", args, { cwd: root, env: GIT_ENV, encoding: "utf8" }).trim();
  const write = (relativePath: string, content: string): void => {
    const target = join(root, relativePath);
    mkdirSync(join(target, ".."), { recursive: true });
    writeFileSync(target, content);
  };
  git("init", "--quiet", "--initial-branch=main");
  // Same rule as the real repository: run outputs are ignored except manifests and summaries.
  write(
    ".gitignore",
    "artifacts/**\n!artifacts/**/\n!artifacts/**/manifest.json\n!artifacts/**/*.summary.csv\nartifacts/_scratch/\n",
  );
  write("experiments/ledger.jsonl", "");
  return {
    root,
    git,
    write,
    commitAll: (message) => {
      git("add", "--all");
      git("commit", "--quiet", "--allow-empty", "-m", message);
      return git("rev-parse", "HEAD");
    },
    writeCard: (dirName, card, runSource = SCALE_RUN) => {
      const id = dirName.slice(0, 4);
      write(`experiments/cards/${dirName}/card.json`, JSON.stringify(validCard({ id, ...card }), null, 2) + "\n");
      write(`experiments/cards/${dirName}/run.ts`, runSource);
    },
    dispose: () => rmSync(root, { recursive: true, force: true }),
  };
}
