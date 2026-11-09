import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync, rmSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { listCards } from "../../src/experiment/card.ts";
import type { LoadedCard } from "../../src/experiment/card.ts";
import { executeCard, harnessPaths } from "../../src/experiment/harness.ts";
import type { ExecuteOutcome, HarnessPaths } from "../../src/experiment/harness.ts";
import { readLedger } from "../../src/experiment/ledger.ts";
import { checkPreregistration } from "../../src/experiment/preregistration.ts";
import type { PreregistrationProblem } from "../../src/experiment/preregistration.ts";
import { hashDir, hashDirAtCommit, hashFile } from "../../src/experiment/provenance.ts";
import { reproduce } from "../../src/experiment/repro.ts";
import { makeFixtureRepo } from "./harnessFixture.ts";
import type { FixtureRepo } from "./harnessFixture.ts";

// The pre-registration rules, each exercised by actually doing the thing they forbid in a
// throwaway repository and running the real check. "base" plays the role of origin/main.

interface Session {
  repo: FixtureRepo;
  paths: HarnessPaths;
  exp: (id: string) => ExecuteOutcome;
  /** Rule slugs the check reports against `base`, in order. */
  rules: (base: string) => string[];
  problems: (base: string) => PreregistrationProblem[];
}

function withSession(body: (session: Session) => void): void {
  const repo = makeFixtureRepo();
  const paths = harnessPaths(repo.root);
  try {
    const problems = (base: string): PreregistrationProblem[] => checkPreregistration(repo.root, base).problems;
    body({
      repo,
      paths,
      exp: (id) => {
        const cards = listCards(paths.cardsDir);
        const loaded = cards.find((card) => card.card.id === id) as LoadedCard;
        return executeCard(loaded, cards, readLedger(paths.ledgerPath), paths, { mode: "result", command: "exp", stdio: "ignore" });
      },
      problems,
      rules: (base) => problems(base).map((problem) => problem.rule),
    });
  } finally {
    repo.dispose();
  }
}

/** Rewrites the ledger file through `edit`, line by line. */
function editLedger(paths: HarnessPaths, edit: (lines: string[]) => string[]): void {
  const lines = readFileSync(paths.ledgerPath, "utf8").split("\n").filter((line) => line !== "");
  writeFileSync(paths.ledgerPath, edit(lines).map((line) => `${line}\n`).join(""));
}

test("hashDirAtCommit agrees with hashDir for a committed directory, ignores dotfiles, and is null for a missing one", () => {
  withSession(({ repo }) => {
    repo.writeCard("0001-fixture", {});
    repo.write("experiments/cards/0001-fixture/.DS_Store", "junk that got committed");
    repo.write("experiments/cards/0001-fixture/helpers/util.ts", "export const x = 1;\n");
    const commit = repo.commitAll("add card");
    const dir = "experiments/cards/0001-fixture";

    assert.equal(hashDirAtCommit(commit, dir, repo.root), hashDir(join(repo.root, dir)));
    assert.equal(hashDirAtCommit(commit, `${dir}/`, repo.root), hashDir(join(repo.root, dir)), "a trailing slash is accepted");
    assert.equal(hashDirAtCommit(commit, "experiments/cards/0002-nope", repo.root), null);

    // The hash at the old commit is unaffected by later edits.
    repo.write(`${dir}/run.ts`, "export function run() {}\n");
    repo.commitAll("edit card");
    assert.notEqual(hashDirAtCommit("HEAD", dir, repo.root), hashDirAtCommit(commit, dir, repo.root));
    // A sibling directory sharing the prefix is not swept in.
    repo.write("experiments/cards/0001-fixture-two/card.json", "{}");
    repo.commitAll("sibling");
    assert.equal(hashDirAtCommit("HEAD", dir, repo.root), hashDir(join(repo.root, dir)));
  });
});

test("the intended flow passes: card merged to base, run from base, result committed afterwards", () => {
  withSession(({ repo, exp, problems }) => {
    repo.writeCard("0001-fixture", {});
    const base = repo.commitAll("card merged to base");
    exp("0001");
    repo.commitAll("result PR");

    assert.deepEqual(problems(base), []);
    const report = checkPreregistration(repo.root, base);
    assert.deepEqual({ lines: report.ledgerLines, fresh: report.newLines, cards: report.cardsOnBase }, { lines: 1, fresh: 1, cards: 1 });
    // And once that PR is itself the base, there is nothing new to check.
    assert.equal(checkPreregistration(repo.root, "HEAD").newLines, 0);
  });
});

test("a card added and run in the same change is rejected: the run's commit was never on base", () => {
  withSession(({ repo, exp, problems }) => {
    const base = repo.commitAll("base, before the card exists");
    repo.writeCard("0001-fixture", {});
    repo.commitAll("add the card on the PR branch");
    exp("0001");
    repo.commitAll("and its result, in the same PR");

    const found = problems(base);
    assert.deepEqual(found.map((problem) => problem.rule), ["run-on-base"]);
    assert.match(found[0]?.detail ?? "", /is not on .* merge the card first, then run it from the base branch/);
  });
});

test("editing a merged card is rejected, and so is deleting it; adding a new card is fine", () => {
  withSession(({ repo, rules, problems }) => {
    repo.writeCard("0001-fixture", {});
    const base = repo.commitAll("card merged to base");

    repo.writeCard("0002-new", {});
    assert.deepEqual(rules(base), [], "a new card directory is allowed");

    repo.writeCard("0001-fixture", { decisionRule: { text: "pass if total >= minTotal", thresholds: { minTotal: 1 } } });
    const edited = problems(base);
    assert.deepEqual(edited.map((problem) => [problem.rule, problem.where]), [["card-immutable", "experiments/cards/0001-fixture"]]);
    assert.match(edited[0]?.detail ?? "", /Add a new card with "supersedes" instead/);

    rmSync(join(repo.root, "experiments/cards/0001-fixture"), { recursive: true });
    assert.match(problems(base)[0]?.detail ?? "", /was removed/);
  });
});

test("TEMPLATE and proposed/ are not cards: they can change freely", () => {
  withSession(({ repo, rules }) => {
    repo.write("experiments/cards/TEMPLATE/card.json", "{}");
    repo.write("experiments/cards/proposed/0009-idea/card.json", "{}");
    const base = repo.commitAll("base");
    repo.write("experiments/cards/TEMPLATE/card.json", '{ "edited": true }');
    repo.write("experiments/cards/proposed/0009-idea/card.json", '{ "edited": true }');
    assert.deepEqual(rules(base), []);
  });
});

test("the ledger only grows: removing, altering or reordering a base line is rejected", () => {
  withSession(({ repo, paths, exp, rules }) => {
    repo.writeCard("0001-first", {});
    repo.writeCard("0002-second", {});
    repo.commitAll("cards");
    exp("0001");
    repo.commitAll("result 1");
    exp("0002");
    const base = repo.commitAll("result 2");
    assert.deepEqual(rules(base), []);
    const original = readFileSync(paths.ledgerPath, "utf8");

    editLedger(paths, (lines) => lines.slice(1));
    assert.ok(rules(base).includes("ledger-append-only"), "removed a line");

    writeFileSync(paths.ledgerPath, original);
    editLedger(paths, (lines) => [(lines[0] ?? "").replace('"verdict":"pass"', '"verdict":"fail"'), ...lines.slice(1)]);
    assert.ok(rules(base).includes("ledger-append-only"), "altered a line");

    writeFileSync(paths.ledgerPath, original);
    editLedger(paths, (lines) => [...lines].reverse());
    assert.ok(rules(base).includes("ledger-append-only"), "reordered");

    writeFileSync(paths.ledgerPath, original);
    assert.deepEqual(rules(base), [], "restored");
  });
});

test("an unreadable ledger is reported, not treated as empty", () => {
  withSession(({ repo, paths, rules }) => {
    const base = repo.commitAll("base");
    writeFileSync(paths.ledgerPath, "{ not json\n");
    assert.deepEqual(rules(base), ["ledger-readable"]);
  });
});

test("editing a recorded manifest is rejected, including one recorded long ago", () => {
  withSession(({ repo, exp, problems }) => {
    repo.writeCard("0001-fixture", {});
    repo.commitAll("card");
    const outcome = exp("0001");
    const base = repo.commitAll("result, now part of base");

    const manifestPath = join(outcome.outDir, "manifest.json");
    const relativeManifest = `artifacts/0001-fixture/${outcome.manifest.runId}/manifest.json`;
    writeFileSync(manifestPath, readFileSync(manifestPath, "utf8").replace('"verdict": "pass"', '"verdict": "fail"'));
    // Caught twice over: its hash no longer matches the ledger, and a committed artifact changed.
    assert.deepEqual(problems(base).map((problem) => [problem.rule, problem.where]), [
      ["manifest-immutable", outcome.manifest.runId],
      ["artifact-immutable", relativeManifest],
    ]);

    rmSync(manifestPath);
    assert.match(problems(base)[0]?.detail ?? "", /is missing/);
  });
});

test("a committed artifact cannot be modified or deleted; new artifacts are fine", () => {
  withSession(({ repo, rules, problems }) => {
    const summary = "artifacts/2026-01-01-demo/values.summary.csv";
    repo.write(summary, "seed,value\n1,0.5\n");
    const base = repo.commitAll("an artifact on base");

    repo.write("artifacts/2026-01-02-new/values.summary.csv", "seed,value\n1,0.1\n");
    assert.deepEqual(rules(base), [], "an untracked new artifact");
    repo.commitAll("commit the new artifact");
    assert.deepEqual(rules(base), [], "a committed new artifact");

    repo.write(summary, "seed,value\n1,0.75\n");
    const modified = problems(base);
    assert.deepEqual(modified.map((problem) => [problem.rule, problem.where]), [["artifact-immutable", summary]]);
    assert.match(modified[0]?.detail ?? "", /was modified; an artifact on .* is evidence and is never rewritten/);

    rmSync(join(repo.root, summary));
    assert.match(problems(base)[0]?.detail ?? "", /was deleted/);
  });
});

test("a ledger line whose card hash, commit or clean-tree claim does not hold up is rejected", () => {
  withSession(({ repo, paths, exp, problems }) => {
    repo.writeCard("0001-fixture", {});
    const base = repo.commitAll("card merged to base");
    exp("0001");
    const honest = readFileSync(paths.ledgerPath, "utf8");
    assert.deepEqual(problems(base), []);

    // Claim a different card hash than the card had at the run's commit.
    writeFileSync(paths.ledgerPath, honest.replace(/"cardSha256":"[0-9a-f]{64}"/, `"cardSha256":"${"0".repeat(64)}"`));
    assert.match(problems(base)[0]?.detail ?? "", /does not hash to the cardSha256 the ledger records/);

    // Claim a different commit than the manifest recorded.
    writeFileSync(paths.ledgerPath, honest.replace(/"gitCommit":"[0-9a-f]{40}"/, `"gitCommit":"${"a".repeat(40)}"`));
    const wrongCommit = problems(base).map((problem) => problem.detail).join("\n");
    assert.match(wrongCommit, /differs from the manifest's/);
    assert.match(wrongCommit, /is not on/);
  });
});

test("a manifest that records a dirty tree or a scratch run cannot be ledgered", () => {
  withSession(({ repo, paths, exp, problems }) => {
    repo.writeCard("0001-fixture", {});
    const base = repo.commitAll("card merged to base");
    const outcome = exp("0001");

    // Forge both the manifest and the ledger's hash of it, as someone hand-writing a line would.
    const manifestPath = join(outcome.outDir, "manifest.json");
    const forged = readFileSync(manifestPath, "utf8").replace('"gitDirty": false', '"gitDirty": true').replace('"scratch": false', '"scratch": true');
    writeFileSync(manifestPath, forged);
    writeFileSync(
      paths.ledgerPath,
      readFileSync(paths.ledgerPath, "utf8").replace(/"manifestSha256":"[0-9a-f]{64}"/, `"manifestSha256":"${hashFile(manifestPath)}"`),
    );

    const details = problems(base).map((problem) => `${problem.rule}: ${problem.detail}`);
    assert.equal(details.length, 2, details.join("\n"));
    assert.match(details[0] ?? "", /run-on-base: its manifest is a scratch run/);
    assert.match(details[1] ?? "", /run-on-base: its manifest does not record a clean working tree/);
  });
});

test("a card has one result: a second non-repro line for the same card is rejected", () => {
  withSession(({ repo, paths, exp, problems }) => {
    repo.writeCard("0001-fixture", {});
    const base = repo.commitAll("card merged to base");
    exp("0001");
    repo.commitAll("result");

    // The harness refuses a second run, so forge one by duplicating the line under a new id.
    editLedger(paths, (lines) => [...lines, (lines[0] ?? "").replace(/"ledgerId":"[^"]+"/, '"ledgerId":"forged-second-run"')]);
    const found = problems(base);
    assert.deepEqual(found.map((problem) => [problem.rule, problem.where]), [["run-once", "forged-second-run"]]);
  });
});

test("a result whose dependency was not met by an earlier line is rejected", () => {
  withSession(({ repo, paths, exp, rules }) => {
    repo.writeCard("0001-control", {});
    repo.writeCard("0002-retest", { dependsOn: [{ card: "0001", verdict: "pass" }] });
    const base = repo.commitAll("cards merged to base");
    // Both runs execute at the base commit: the first run's uncommitted outputs do not dirty the
    // tree, so one change can carry a dependency and its dependent.
    exp("0001");
    exp("0002");
    assert.deepEqual(rules(base), [], "the honest order passes, uncommitted");
    repo.commitAll("both results in one change");
    assert.deepEqual(rules(base), [], "and committed");

    editLedger(paths, (lines) => [...lines].reverse());
    assert.deepEqual(rules(base), ["dependency"], "the dependent's line comes before its dependency's");

    editLedger(paths, (lines) => lines.filter((line) => line.includes('"cardId":"0002"')));
    assert.deepEqual(rules(base), ["dependency"], "the dependency has no line at all");
  });
});

test("reproductions are checked for provenance but exempt from run-once and dependencies", () => {
  withSession(({ repo, paths, exp, rules }) => {
    repo.writeCard("0001-fixture", {});
    repo.commitAll("card");
    const original = exp("0001");
    const base = repo.commitAll("result, now part of base");

    const repro = (): void => {
      reproduce(original.manifest.runId, listCards(paths.cardsDir), readLedger(paths.ledgerPath), paths, {
        tolerance: 0,
        atCommit: false,
        scratch: false,
        budgetMinutes: 1,
        command: "repro",
        stdio: "ignore",
      });
    };
    repro();
    repro();
    repo.commitAll("two reproductions in one change");

    assert.deepEqual(readLedger(paths.ledgerPath).map((entry) => entry.kind), ["confirmatory", "repro", "repro"]);
    assert.deepEqual(rules(base), []);
  });
});

test("a run made after committing results locally is rejected: its commit is not on base", () => {
  withSession(({ repo, exp, rules }) => {
    repo.writeCard("0001-first", {});
    repo.writeCard("0002-second", {});
    const base = repo.commitAll("cards merged to base");
    exp("0001");
    repo.commitAll("commit result 1 on the PR branch"); // HEAD is now ahead of base
    exp("0002");
    repo.commitAll("commit result 2");
    assert.deepEqual(rules(base), ["run-on-base"]);
  });
});

test("an unresolvable base ref throws instead of passing", () => {
  withSession(({ repo }) => {
    repo.commitAll("base");
    assert.throws(() => checkPreregistration(repo.root, "origin/does-not-exist"), /does not resolve to a commit/);
  });
});
