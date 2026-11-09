import { test } from "node:test";
import assert from "node:assert/strict";
import { existsSync, readFileSync, readdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { listCards } from "../../src/experiment/card.ts";
import type { LoadedCard } from "../../src/experiment/card.ts";
import { HarnessError, executeCard, harnessPaths, makeRunId } from "../../src/experiment/harness.ts";
import type { ExecuteOptions, ExecuteOutcome, HarnessPaths, RunManifest } from "../../src/experiment/harness.ts";
import { readLedger } from "../../src/experiment/ledger.ts";
import { canonicalJson, hashFile, sha256 } from "../../src/experiment/provenance.ts";
import { HANGING_RUN, MALFORMED_RUN, THROWING_RUN, makeFixtureRepo } from "./harnessFixture.ts";
import type { FixtureRepo } from "./harnessFixture.ts";

// End to end: a real git repository, real card directories, and the real child process
// (scripts/exp-runner.ts). Only the cards are fixtures, and they need nothing but Node.

interface Session {
  repo: FixtureRepo;
  paths: HarnessPaths;
  /** Reloads cards and ledger from disk, then runs card `id`. */
  run: (id: string, options?: Partial<ExecuteOptions>) => ExecuteOutcome;
}

function withSession(body: (session: Session) => void): void {
  const repo = makeFixtureRepo();
  const paths = harnessPaths(repo.root);
  try {
    body({
      repo,
      paths,
      run: (id, options = {}) => {
        const cards = listCards(paths.cardsDir);
        const loaded = cards.find((card) => card.card.id === id) as LoadedCard;
        return executeCard(loaded, cards, readLedger(paths.ledgerPath), paths, {
          mode: "result",
          command: `npm run exp -- ${id}`,
          stdio: "ignore",
          ...options,
        });
      },
    });
  } finally {
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

test("makeRunId: timestamp then short commit, sortable as text", () => {
  assert.equal(makeRunId("2026-11-09T14:22:33.123Z", "e95ce68aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"), "20261109T142233123Z-e95ce68");
  assert.ok(makeRunId("2026-11-09T14:22:33.123Z", "ffffffff") < makeRunId("2026-11-10T00:00:00.000Z", "00000000"));
});

test("a recorded run writes a v2 manifest next to its outputs and appends exactly one ledger line", () => {
  withSession(({ repo, paths, run }) => {
    repo.writeCard("0001-fixture", {});
    const commit = repo.commitAll("add card 0001");
    const [loaded] = listCards(paths.cardsDir);

    const outcome = run("0001");
    const { manifest } = outcome;

    // seeds [1, 2, 3] × scale 2 → total 12 ≥ minTotal 10.
    assert.equal(manifest.verdict, "pass");
    assert.deepEqual(manifest.primary, { name: "total", value: 12 });
    assert.equal(manifest.schemaVersion, 2);
    assert.equal(manifest.cardId, "0001");
    assert.equal(manifest.cardDir, "0001-fixture");
    assert.equal(manifest.cardSha256, loaded?.sha256);
    assert.equal(manifest.kind, "confirmatory");
    assert.equal(manifest.scratch, false);
    assert.equal(manifest.command, "npm run exp -- 0001");
    assert.deepEqual(manifest.seeds, [1, 2, 3]);
    assert.deepEqual(manifest.config, { scale: 2 });
    assert.equal(manifest.configSha256, sha256(canonicalJson({ scale: 2 })));
    assert.deepEqual(manifest.decisionRule, { text: "pass if total >= minTotal", thresholds: { minTotal: 10 } });
    assert.equal(manifest.reproOf, null);
    assert.equal(manifest.match, null);
    assert.ok(manifest.wallSeconds > 0);

    assert.equal(manifest.env.gitCommit, commit);
    assert.equal(manifest.env.gitDirty, false);
    assert.equal(manifest.env.nodeVersion, process.version);
    assert.equal(manifest.env.platform, process.platform);
    assert.equal(manifest.env.arch, process.arch);
    assert.equal(manifest.env.lockfileSha256, null, "the fixture repository has no package-lock.json");

    // Outputs live in artifacts/<card>/<runId>/, and every one is hashed, including the
    // gitignored telemetry. The manifest does not list itself.
    assert.equal(outcome.outDir, join(repo.root, "artifacts", "0001-fixture", manifest.runId));
    assert.deepEqual(readdirSync(outcome.outDir).sort(), ["manifest.json", "telemetry.jsonl", "values.summary.csv"]);
    assert.deepEqual(manifest.files, [
      { path: "telemetry.jsonl", sha256: hashFile(join(outcome.outDir, "telemetry.jsonl")) },
      { path: "values.summary.csv", sha256: hashFile(join(outcome.outDir, "values.summary.csv")) },
    ]);
    assert.equal(readFileSync(join(outcome.outDir, "values.summary.csv"), "utf8"), "seed,value\n1,2\n2,4\n3,6\n");
    assert.deepEqual(JSON.parse(readFileSync(join(outcome.outDir, "manifest.json"), "utf8")), manifest);

    const ledger = readLedger(paths.ledgerPath);
    assert.equal(ledger.length, 1);
    assert.deepEqual(ledger[0], outcome.ledgerEntry);
    assert.deepEqual(ledger[0], {
      ledgerId: manifest.runId,
      cardId: "0001",
      cardSha256: manifest.cardSha256,
      kind: "confirmatory",
      gitCommit: commit,
      configSha256: manifest.configSha256,
      manifestPath: `artifacts/0001-fixture/${manifest.runId}/manifest.json`,
      manifestSha256: hashFile(join(outcome.outDir, "manifest.json")),
      primary: { name: "total", value: 12 },
      verdict: "pass",
      createdAt: manifest.env.createdAt,
      reproOf: null,
      match: null,
    });

    // Nothing is left behind in the staging area.
    assert.deepEqual(readdirSync(join(repo.root, "artifacts", "_scratch", "0001-fixture")), []);
  });
});

test("the verdict is the card's decision rule applied to the run, and a fail is recorded like a pass", () => {
  withSession(({ repo, paths, run }) => {
    repo.writeCard("0001-fixture", { decisionRule: { text: "pass if total >= minTotal", thresholds: { minTotal: 13 } } });
    repo.commitAll("add card 0001");
    assert.equal(run("0001").manifest.verdict, "fail");
    assert.equal(readLedger(paths.ledgerPath)[0]?.verdict, "fail");
  });
});

test("a card with a verdict is never run again, even with a clean tree", () => {
  withSession(({ repo, paths, run }) => {
    repo.writeCard("0001-fixture", {});
    repo.commitAll("add card 0001");
    run("0001");
    repo.commitAll("record result");

    assertRefused(() => run("0001"), /already has its verdict \(pass, .*A pre-registered card runs once/s);
    assert.equal(readLedger(paths.ledgerPath).length, 1);
  });
});

test("a recorded run is refused on a dirty tree, and nothing is written", () => {
  withSession(({ repo, paths, run }) => {
    repo.writeCard("0001-fixture", {});
    repo.commitAll("add card 0001");
    repo.write("experiments/cards/0001-fixture/run.ts", "export function run() { return 'edited after commit'; }\n");

    assertRefused(() => run("0001"), /uncommitted or untracked files/);
    assert.equal(readLedger(paths.ledgerPath).length, 0);
    assert.equal(existsSync(join(repo.root, "artifacts")), false);
  });
});

test("several runs can be recorded from one commit: a run's own not-yet-committed outputs do not make the tree dirty", () => {
  withSession(({ repo, paths, run }) => {
    repo.writeCard("0001-first", {});
    repo.writeCard("0002-second", { dependsOn: [{ card: "0001", verdict: "pass" }] });
    const commit = repo.commitAll("add cards");

    const first = run("0001");
    const second = run("0002"); // nothing committed in between

    for (const outcome of [first, second]) {
      assert.equal(outcome.manifest.env.gitCommit, commit);
      assert.equal(outcome.manifest.env.gitDirty, false);
    }
    assert.equal(readLedger(paths.ledgerPath).length, 2);
  });
});

test("editing a committed artifact does make the tree dirty: results are exempt only while they are new", () => {
  withSession(({ repo, run }) => {
    repo.writeCard("0001-first", {});
    repo.writeCard("0002-second", {});
    repo.commitAll("add cards");
    const first = run("0001");
    repo.commitAll("commit the first result");

    writeFileSync(join(first.outDir, "values.summary.csv"), "seed,value\n1,999\n");
    assertRefused(() => run("0002"), /uncommitted or untracked files \(other than recorded results/);
  });
});

test("a scratch run works on a dirty tree, may override seeds, records the dirty flag, and never touches the ledger", () => {
  withSession(({ repo, paths, run }) => {
    repo.writeCard("0001-fixture", {});
    repo.commitAll("add card 0001");
    repo.write("untracked.txt", "dirty");

    const outcome = run("0001", { mode: "scratch", seeds: [10] });
    assert.equal(outcome.ledgerEntry, null);
    assert.equal(outcome.manifest.scratch, true);
    assert.equal(outcome.manifest.env.gitDirty, true);
    assert.deepEqual(outcome.manifest.seeds, [10]);
    assert.deepEqual(outcome.manifest.primary, { name: "total", value: 20 });
    assert.equal(outcome.outDir, join(repo.root, "artifacts", "_scratch", "0001-fixture", outcome.manifest.runId));
    assert.equal(readFileSync(paths.ledgerPath, "utf8"), "");
    assert.equal(existsSync(join(repo.root, "artifacts", "0001-fixture")), false);

    // The scratch area is gitignored, so a scratch run does not itself dirty the tree further.
    assert.equal(repo.git("status", "--porcelain"), "?? untracked.txt");
  });
});

test("seeds cannot be overridden in a recorded run", () => {
  withSession(({ repo, run }) => {
    repo.writeCard("0001-fixture", {});
    repo.commitAll("add card 0001");
    assertRefused(() => run("0001", { seeds: [10] }), /seeds can only be overridden in a scratch run/);
  });
});

test("a run that throws, returns a malformed result, or exceeds its budget records nothing and can be retried", () => {
  withSession(({ repo, paths, run }) => {
    repo.writeCard("0001-throws", {}, THROWING_RUN);
    repo.writeCard("0002-malformed", {}, MALFORMED_RUN);
    repo.writeCard("0003-hangs", { budget: { maxWallMinutes: 0.02 } }, HANGING_RUN);
    repo.commitAll("add failing cards");

    assertRefused(() => run("0001"), /failed with exit code 1\. Nothing was recorded/);
    assertRefused(() => run("0002"), /malformed result/);
    assertRefused(() => run("0003"), /exceeded its budget of 0\.02 minutes and was stopped\. Nothing was recorded/);

    assert.equal(readFileSync(paths.ledgerPath, "utf8"), "");
    assert.equal(repo.git("status", "--porcelain"), "", "no stray committable files");
    for (const dirName of ["0001-throws", "0002-malformed", "0003-hangs"]) {
      assert.equal(existsSync(join(repo.root, "artifacts", dirName)), false, `${dirName} must have no recorded artifacts`);
    }

    // Fix the card that threw: because nothing was recorded, the same card id runs normally.
    repo.writeCard("0001-throws", {});
    repo.commitAll("fix card 0001");
    assert.equal(run("0001").manifest.verdict, "pass");
  });
});

test("a dependent card waits, then receives its dependency's recorded result", () => {
  withSession(({ repo, run }) => {
    repo.writeCard("0001-calibration", { kind: "calibration" });
    repo.writeCard("0002-dependent", { dependsOn: [{ card: "0001", verdict: "pass", reads: ["total"] }] });
    repo.commitAll("add cards");

    assertRefused(() => run("0002"), /cannot run: waiting on 0001/);

    const first = run("0001");
    repo.commitAll("record 0001");
    const second = run("0002");

    const handed = (second.manifest.resultSummary as { dependencies: Record<string, unknown> }).dependencies;
    assert.deepEqual(handed, {
      "0001": {
        ledgerId: first.manifest.runId,
        manifestPath: `artifacts/0001-calibration/${first.manifest.runId}/manifest.json`,
        verdict: "pass",
        resultSummary: first.manifest.resultSummary,
      },
    });
  });
});

test("a card whose dependency reached the wrong verdict never runs, not even as scratch", () => {
  withSession(({ repo, run }) => {
    repo.writeCard("0001-control", { decisionRule: { text: "pass if total >= minTotal", thresholds: { minTotal: 99 } } });
    repo.writeCard("0002-retest", { dependsOn: [{ card: "0001", verdict: "pass" }] });
    repo.commitAll("add cards");
    assert.equal(run("0001").manifest.verdict, "fail");
    repo.commitAll("record 0001");

    assertRefused(() => run("0002"), /cannot run: will not run: needs card 0001 to pass, but it reached "fail"/);
    assertRefused(() => run("0002", { mode: "scratch" }), /will not run/);
  });
});

test("a dependency's manifest edited after it was ledgered is refused", () => {
  withSession(({ repo, run }) => {
    repo.writeCard("0001-calibration", { kind: "calibration" });
    repo.writeCard("0002-dependent", { dependsOn: [{ card: "0001", verdict: "pass" }] });
    repo.commitAll("add cards");
    const first = run("0001");
    const manifestPath = join(first.outDir, "manifest.json");
    const tampered = JSON.parse(readFileSync(manifestPath, "utf8")) as RunManifest;
    tampered.resultSummary = { total: 9999 };
    writeFileSync(manifestPath, JSON.stringify(tampered, null, 2) + "\n");
    repo.commitAll("record 0001, with an edited manifest");

    assertRefused(() => run("0002"), /no longer matches the hash in the ledger/);
  });
});

test("a repro card runs any number of times; each run is ledgered as a reproduction with match = (verdict is pass)", () => {
  withSession(({ repo, paths, run }) => {
    repo.writeCard("0000-canary", { kind: "repro", config: { scale: 2, reproOf: "legacy:some-run" } });
    repo.commitAll("add canary");

    const first = run("0000");
    repo.commitAll("record canary run 1");
    const second = run("0000");

    assert.notEqual(first.manifest.runId, second.manifest.runId);
    assert.deepEqual(
      readLedger(paths.ledgerPath).map((entry) => [entry.kind, entry.reproOf, entry.match, entry.verdict]),
      [
        ["repro", "legacy:some-run", true, "pass"],
        ["repro", "legacy:some-run", true, "pass"],
      ],
    );
  });
});

test("repro mode re-runs a card that has its verdict, records the comparison, and files it under artifacts/repro/", () => {
  withSession(({ repo, paths, run }) => {
    repo.writeCard("0001-fixture", {});
    repo.commitAll("add card 0001");
    const original = run("0001");
    repo.commitAll("record 0001");

    const seen: string[] = [];
    const rerun = run("0001", {
      mode: "repro",
      command: "npm run repro -- x",
      repro: {
        of: original.manifest.runId,
        judge: ({ outDir, result }) => {
          seen.push(readFileSync(join(outDir, "values.summary.csv"), "utf8"), String(result.primary.value));
          return { identical: true, maxDifference: 0, mismatches: [], mismatchCount: 0, filesCompared: 1, tolerance: 0, match: true };
        },
      },
    });

    assert.deepEqual(seen, ["seed,value\n1,2\n2,4\n3,6\n", "12"], "the judge sees the re-run's outputs before they are moved");
    assert.equal(rerun.outDir, join(repo.root, "artifacts", "repro", "0001-fixture", rerun.manifest.runId));
    assert.equal(rerun.manifest.kind, "repro");
    assert.equal(rerun.manifest.reproOf, original.manifest.runId);
    assert.equal(rerun.manifest.match, true);
    assert.equal(rerun.manifest.comparison?.filesCompared, 1);

    const ledger = readLedger(paths.ledgerPath);
    assert.deepEqual(ledger.map((entry) => entry.kind), ["confirmatory", "repro"]);
    assert.equal(ledger[1]?.match, true);
  });
});
