import { test } from "node:test";
import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { listCards } from "../../src/experiment/card.ts";
import type { LoadedCard } from "../../src/experiment/card.ts";
import { HarnessError, executeCard, harnessPaths } from "../../src/experiment/harness.ts";
import type { ExecuteOutcome, HarnessPaths } from "../../src/experiment/harness.ts";
import { readLedger } from "../../src/experiment/ledger.ts";
import { reproduce } from "../../src/experiment/repro.ts";
import type { ReproOptions, ReproOutcome } from "../../src/experiment/repro.ts";
import { makeFixtureRepo } from "./harnessFixture.ts";
import type { FixtureRepo } from "./harnessFixture.ts";

// Reproduction, end to end, against throwaway repositories. Nothing here needs tfjs.

/**
 * Like SCALE_RUN, but every value is shifted by the JANUS_TEST_DRIFT environment variable.
 * That is how these tests make a re-run differ without touching the card (which would change
 * its hash): it stands in for drift from a changed `src/`, platform or dependency.
 */
const DRIFTING_RUN = `
import { writeFileSync } from "node:fs";
import { join } from "node:path";

export function run(context) {
  const drift = Number(process.env.JANUS_TEST_DRIFT ?? "0");
  const values = context.seeds.map((seed) => seed * context.card.config.scale + drift);
  writeFileSync(join(context.outDir, "values.summary.csv"), "seed,value\\n" + context.seeds.map((seed, i) => seed + "," + values[i]).join("\\n") + "\\n");
  const total = values.reduce((sum, value) => sum + value, 0);
  return {
    primary: { name: "total", value: total },
    verdict: total >= context.card.decisionRule.thresholds.minTotal ? "pass" : "fail",
    resultSummary: { total },
  };
}
`;

interface Session {
  repo: FixtureRepo;
  paths: HarnessPaths;
  exp: (id: string) => ExecuteOutcome;
  repro: (target: string, options?: Partial<ReproOptions>) => ReproOutcome;
}

function withSession(body: (session: Session) => void): void {
  const repo = makeFixtureRepo();
  const paths = harnessPaths(repo.root);
  const previousDrift = process.env.JANUS_TEST_DRIFT;
  try {
    body({
      repo,
      paths,
      exp: (id) => {
        const cards = listCards(paths.cardsDir);
        const loaded = cards.find((card) => card.card.id === id) as LoadedCard;
        return executeCard(loaded, cards, readLedger(paths.ledgerPath), paths, { mode: "result", command: "exp", stdio: "ignore" });
      },
      repro: (target, options = {}) =>
        reproduce(target, listCards(paths.cardsDir), readLedger(paths.ledgerPath), paths, {
          tolerance: 0,
          atCommit: false,
          scratch: false,
          budgetMinutes: 1,
          command: `npm run repro -- ${target}`,
          stdio: "ignore",
          ...options,
        }),
    });
  } finally {
    if (previousDrift === undefined) {
      delete process.env.JANUS_TEST_DRIFT;
    } else {
      process.env.JANUS_TEST_DRIFT = previousDrift;
    }
    repo.dispose();
  }
}

function assertRefused(action: () => unknown, expected: RegExp): void {
  assert.throws(action, (error: unknown) => {
    assert.ok(error instanceof HarnessError, `expected a HarnessError, got ${String(error)}`);
    assert.match(error.message, expected);
    return true;
  });
}

function readManifest(outDir: string): Record<string, unknown> {
  return JSON.parse(readFileSync(join(outDir, "manifest.json"), "utf8")) as Record<string, unknown>;
}

// --- Harness (v2) results -------------------------------------------------------------------

test("a deterministic card reproduces byte-identically, by ledger id and by manifest path", () => {
  withSession(({ repo, paths, exp, repro }) => {
    repo.writeCard("0001-fixture", {}, DRIFTING_RUN);
    repo.commitAll("add card");
    const original = exp("0001");
    repo.commitAll("record result");

    const byId = repro(original.manifest.runId);
    assert.equal(byId.of, original.manifest.runId);
    assert.deepEqual(
      { match: byId.comparison.match, identical: byId.comparison.identical, files: byId.comparison.filesCompared, max: byId.comparison.maxDifference },
      { match: true, identical: true, files: 1, max: 0 },
    );
    assert.equal(byId.ledgerEntry?.kind, "repro");
    assert.equal(byId.ledgerEntry?.reproOf, original.manifest.runId);
    assert.equal(byId.ledgerEntry?.match, true);
    assert.ok(byId.outDir.startsWith(join(repo.root, "artifacts", "repro", "0001-fixture")));
    assert.equal(readManifest(byId.outDir).reproOf, original.manifest.runId);

    repo.commitAll("record reproduction");
    const byPath = repro(`artifacts/0001-fixture/${original.manifest.runId}/manifest.json`);
    assert.equal(byPath.comparison.match, true);

    // The card still has exactly one result; reproductions are separate lines.
    assert.deepEqual(readLedger(paths.ledgerPath).map((entry) => entry.kind), ["confirmatory", "repro", "repro"]);
  });
});

test("a re-run that drifts is recorded as a mismatch, with the differing cells and the primary value located", () => {
  withSession(({ repo, exp, repro }) => {
    repo.writeCard("0001-fixture", {}, DRIFTING_RUN);
    repo.commitAll("add card");
    const original = exp("0001");
    repo.commitAll("record result");

    process.env.JANUS_TEST_DRIFT = "0.25";
    const outcome = repro(original.manifest.runId);

    assert.equal(outcome.comparison.match, false);
    assert.equal(outcome.ledgerEntry?.match, false);
    // values 2,4,6 → 2.25,4.25,6.25; total 12 → 12.75. difference() is relative above magnitude 1,
    // so the largest is on the smallest cell: 0.25 / 2.25 (the total's is 0.75 / 12.75).
    assert.equal(outcome.comparison.maxDifference, 0.25 / 2.25);
    assert.deepEqual(
      outcome.comparison.mismatches.map((mismatch) => `${mismatch.file}:${mismatch.line}:${mismatch.column}`),
      ["values.summary.csv:2:2", "values.summary.csv:3:2", "values.summary.csv:4:2", "manifest.json:0:0"],
    );
    assert.match(outcome.comparison.mismatches.at(-1)?.detail ?? "", /total: expected 12, got 12\.75/);
  });
});

test("a tolerance accepts drift below it and still reports that the outputs are not identical", () => {
  withSession(({ repo, exp, repro }) => {
    repo.writeCard("0001-fixture", {}, DRIFTING_RUN);
    repo.commitAll("add card");
    const original = exp("0001");
    repo.commitAll("record result");

    process.env.JANUS_TEST_DRIFT = "0.000001";
    const tolerant = repro(original.manifest.runId, { tolerance: 1e-5 });
    assert.equal(tolerant.comparison.match, true);
    assert.equal(tolerant.comparison.identical, false);
    assert.ok(tolerant.comparison.maxDifference > 0);
  });
});

test("a re-run that flips the verdict is a mismatch even inside the numeric tolerance", () => {
  withSession(({ repo, exp, repro }) => {
    // total 12 against minTotal 12: passes exactly, so any negative drift flips it.
    repo.writeCard("0001-fixture", { decisionRule: { text: "pass if total >= minTotal", thresholds: { minTotal: 12 } } }, DRIFTING_RUN);
    repo.commitAll("add card");
    const original = exp("0001");
    assert.equal(original.manifest.verdict, "pass");
    repo.commitAll("record result");

    process.env.JANUS_TEST_DRIFT = "-0.000001";
    const outcome = repro(original.manifest.runId, { tolerance: 1e-3 });
    assert.equal(outcome.comparison.match, false);
    assert.deepEqual(outcome.comparison.mismatches.map((mismatch) => mismatch.detail), ['verdict: expected "pass", got "fail"']);
  });
});

test("refuses a card that changed since the run, a dirty tree, a scratch manifest, and --at-commit on a harness manifest", () => {
  withSession(({ repo, paths, exp, repro }) => {
    repo.writeCard("0001-fixture", {}, DRIFTING_RUN);
    repo.commitAll("add card");
    const original = exp("0001");
    repo.commitAll("record result");
    const id = original.manifest.runId;

    assertRefused(() => repro(id, { atCommit: true }), /--at-commit is for pre-harness/);

    repo.write("untracked.txt", "dirty");
    assertRefused(() => repro(id), /uncommitted or untracked files/);
    const scratch = repro(id, { scratch: true });
    assert.equal(scratch.comparison.match, true, "a scratch reproduction still compares");
    assert.equal(scratch.ledgerEntry, null);
    assert.ok(scratch.outDir.startsWith(join(repo.root, "artifacts", "_scratch")));
    assertRefused(() => repro(join(scratch.outDir, "manifest.json"), { scratch: true }), /is itself a reproduction record/);
    repo.commitAll("commit the stray file");

    repo.write("experiments/cards/0001-fixture/run.ts", DRIFTING_RUN + "// edited\n");
    repo.commitAll("edit the card after its run");
    assertRefused(() => repro(id), /card 0001-fixture has changed since run/);

    assertRefused(() => repro("no-such-id"), /neither a ledger id nor a path to an existing manifest\.json/);
    assert.equal(readLedger(paths.ledgerPath).length, 1, "none of the refusals recorded anything");
  });
});

// --- Pre-harness (v1) runs ------------------------------------------------------------------

const RUN = "2026-01-01-demo";

/** A pre-harness script: writes its summary straight into artifacts/<run>/, like the real ones. */
function legacyScript(value: string): string {
  return `
import { mkdirSync, writeFileSync } from "node:fs";
mkdirSync("artifacts/${RUN}/seed-1", { recursive: true });
writeFileSync("artifacts/${RUN}/values.summary.csv", "seed,value\\n1,${value}\\n");
`;
}

/**
 * Lays out history the way the real repository has it: the script is committed, the run's
 * manifest (citing that commit) and summary are committed later, and then the script is changed
 * at HEAD so that it no longer produces the committed numbers.
 */
function legacyHistory(repo: FixtureRepo, committedValue: string): { runCommit: string; manifest: string } {
  repo.write(`experiments/${RUN}/run.ts`, legacyScript("0.5"));
  const runCommit = repo.commitAll("script as it ran");
  repo.write(`artifacts/${RUN}/values.summary.csv`, `seed,value\n1,${committedValue}\n`);
  repo.write(`artifacts/${RUN}/seed-1/manifest.json`, JSON.stringify({ runId: RUN, gitCommit: runCommit, nodeVersion: "v22.22.2" }));
  repo.commitAll("commit the run's artifacts");
  repo.write(`experiments/${RUN}/run.ts`, legacyScript("999"));
  repo.commitAll("the script drifts at HEAD");
  return { runCommit, manifest: `artifacts/${RUN}/seed-1/manifest.json` };
}

test("a pre-harness run is reproduced at its recorded commit, not at HEAD, and the worktree is cleaned up", () => {
  withSession(({ repo, paths, repro }) => {
    const { runCommit, manifest } = legacyHistory(repo, "0.5");
    const head = repo.git("rev-parse", "HEAD");

    const outcome = repro(manifest, { atCommit: true });

    assert.equal(outcome.of, `legacy:${RUN}`);
    assert.deepEqual(
      { match: outcome.comparison.match, identical: outcome.comparison.identical, files: outcome.comparison.filesCompared },
      { match: true, identical: true, files: 1 },
    );
    assert.equal(outcome.outDir.startsWith(join(repo.root, "artifacts", "repro", `legacy-${RUN}`)), true);

    const recorded = readManifest(outcome.outDir);
    assert.equal(recorded.cardId, `legacy:${RUN}`);
    assert.equal(recorded.kind, "repro");
    assert.equal(recorded.match, true);
    assert.equal((recorded.env as { gitCommit: string }).gitCommit, head, "env records the commit the harness ran from");
    assert.deepEqual(recorded.config, { script: `experiments/${RUN}/run.ts`, commit: runCommit, tolerance: 0 });
    assert.equal((recorded.resultSummary as { reproducedCommit: string }).reproducedCommit, runCommit);

    const [entry] = readLedger(paths.ledgerPath);
    assert.deepEqual(
      { cardId: entry?.cardId, kind: entry?.kind, verdict: entry?.verdict, match: entry?.match, reproOf: entry?.reproOf, gitCommit: entry?.gitCommit },
      { cardId: `legacy:${RUN}`, kind: "repro", verdict: "pass", match: true, reproOf: `legacy:${RUN}`, gitCommit: head },
    );

    // The committed reference was not overwritten, and no worktree or stray file is left.
    assert.equal(readFileSync(join(repo.root, "artifacts", RUN, "values.summary.csv"), "utf8"), "seed,value\n1,0.5\n");
    assert.equal(repo.git("worktree", "list").split("\n").length, 1);
    assert.equal(repo.git("rev-parse", "HEAD"), head);
  });
});

test("a pre-harness run whose committed numbers the script cannot regenerate is recorded as not reproducing", () => {
  withSession(({ repo, paths, repro }) => {
    // The script writes 0.5 at its commit, but the committed summary says 0.75.
    const { manifest } = legacyHistory(repo, "0.75");
    const outcome = repro(manifest, { atCommit: true });

    assert.equal(outcome.comparison.match, false);
    assert.equal(outcome.comparison.maxDifference, 0.25);
    assert.match(outcome.comparison.mismatches[0]?.detail ?? "", /expected 0\.75, got 0\.5/);
    const [entry] = readLedger(paths.ledgerPath);
    assert.deepEqual({ verdict: entry?.verdict, match: entry?.match }, { verdict: "fail", match: false });
  });
});

test("a pre-harness manifest needs --at-commit, a script that exists at that commit, and a clean tree unless scratch", () => {
  withSession(({ repo, paths, repro }) => {
    const { manifest } = legacyHistory(repo, "0.5");

    assertRefused(() => repro(manifest), /pre-harness manifest.*add --at-commit/s);
    assertRefused(() => repro(manifest, { atCommit: true, script: "experiments/elsewhere/run.ts" }), /does not exist at [0-9a-f]{7}/);

    repo.write("untracked.txt", "dirty");
    assertRefused(() => repro(manifest, { atCommit: true }), /uncommitted or untracked files/);
    const scratch = repro(manifest, { atCommit: true, scratch: true });
    assert.equal(scratch.comparison.match, true);
    assert.equal(scratch.ledgerEntry, null);
    assert.equal(existsSync(join(repo.root, "artifacts", "repro")), false);
    assert.equal(readLedger(paths.ledgerPath).length, 0);
  });
});

test("a pre-harness script that fails at its own commit records nothing and leaves no worktree behind", () => {
  withSession(({ repo, paths, repro }) => {
    repo.write(`experiments/${RUN}/run.ts`, "process.exit(3);\n");
    const runCommit = repo.commitAll("a script that fails");
    repo.write(`artifacts/${RUN}/values.summary.csv`, "seed,value\n1,0.5\n");
    repo.write(`artifacts/${RUN}/seed-1/manifest.json`, JSON.stringify({ gitCommit: runCommit }));
    repo.commitAll("artifacts");

    assertRefused(() => repro(`artifacts/${RUN}/seed-1/manifest.json`, { atCommit: true }), /exited 3 at its own commit/);
    assert.equal(readLedger(paths.ledgerPath).length, 0);
    assert.equal(repo.git("worktree", "list").split("\n").length, 1);
    assert.equal(repo.git("status", "--porcelain"), "");
  });
});
