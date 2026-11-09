import { test } from "node:test";
import assert from "node:assert/strict";
import { execFileSync, spawnSync } from "node:child_process";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";

// scripts/check-evidence-reachable.mjs is exercised end to end against throwaway git repositories:
// the property under test is git ancestry, so mocking git would test nothing.

const SCRIPT = resolve(import.meta.dirname, "../../scripts/check-evidence-reachable.mjs");

/** Isolated from the developer's and CI's git config (signing, hooks, default branch). */
const GIT_ENV = {
  ...process.env,
  GIT_CONFIG_GLOBAL: "/dev/null",
  GIT_CONFIG_SYSTEM: "/dev/null",
  GIT_AUTHOR_NAME: "test",
  GIT_AUTHOR_EMAIL: "test@example.invalid",
  GIT_COMMITTER_NAME: "test",
  GIT_COMMITTER_EMAIL: "test@example.invalid",
  GITHUB_BASE_REF: "",
};

interface Repo {
  dir: string;
  git: (...args: string[]) => string;
  commit: (message: string) => string;
  writeManifest: (runDir: string, manifest: unknown) => void;
  check: (...args: string[]) => { status: number | null; stdout: string; stderr: string };
}

function makeRepo(): Repo {
  const dir = mkdtempSync(join(tmpdir(), "evidence-reachable-"));
  const git = (...args: string[]): string =>
    execFileSync("git", args, { cwd: dir, env: GIT_ENV, encoding: "utf8" }).trim();
  git("init", "--quiet", "--initial-branch=main");
  return {
    dir,
    git,
    commit: (message) => {
      git("commit", "--quiet", "--allow-empty", "-m", message);
      return git("rev-parse", "HEAD");
    },
    writeManifest: (runDir, manifest) => {
      const target = join(dir, "artifacts", runDir);
      mkdirSync(target, { recursive: true });
      writeFileSync(join(target, "manifest.json"), JSON.stringify(manifest));
    },
    check: (...args) => {
      const result = spawnSync(process.execPath, [SCRIPT, ...args], { cwd: dir, env: GIT_ENV, encoding: "utf8" });
      return { status: result.status, stdout: result.stdout, stderr: result.stderr };
    },
  };
}

/** A commit on a side branch that is then deleted: still in the object store, on no ref. */
function orphanCommit(repo: Repo): string {
  repo.git("checkout", "--quiet", "-b", "side");
  const sha = repo.commit("run commit on a branch that gets squash-merged");
  repo.git("checkout", "--quiet", "main");
  repo.git("branch", "--quiet", "-D", "side");
  return sha;
}

function withRepo(body: (repo: Repo) => void): void {
  const repo = makeRepo();
  try {
    body(repo);
  } finally {
    rmSync(repo.dir, { recursive: true, force: true });
  }
}

test("passes when every manifest's commit is an ancestor of the base ref (v1 and v2 manifest shapes)", () => {
  withRepo((repo) => {
    const first = repo.commit("first");
    const second = repo.commit("second");
    repo.writeManifest("run-a/seed-1", { gitCommit: first });
    repo.writeManifest("run-b", { schemaVersion: 2, env: { gitCommit: second } });

    const result = repo.check();
    assert.equal(result.status, 0, result.stderr);
    assert.match(result.stdout, /2 manifests, 2 distinct commits, all reachable/);
  });
});

test("fails, naming the commit and a citing manifest, when a run commit is on no branch and no evidence tag", () => {
  withRepo((repo) => {
    repo.commit("first");
    const orphan = orphanCommit(repo);
    repo.writeManifest("run-orphaned/seed-1", { gitCommit: orphan });

    const result = repo.check();
    assert.equal(result.status, 1);
    assert.match(result.stderr, new RegExp(`UNREACHABLE\\s+${orphan}`));
    assert.match(result.stderr, /artifacts\/run-orphaned\/seed-1\/manifest\.json/);
  });
});

test("an evidence/* tag makes an otherwise-orphaned run commit reachable; a tag outside evidence/ does not", () => {
  withRepo((repo) => {
    repo.commit("first");
    const orphan = orphanCommit(repo);
    repo.writeManifest("run-orphaned", { gitCommit: orphan });

    repo.git("tag", "keep-alive", orphan);
    assert.equal(repo.check().status, 1, "a non-evidence tag must not count");

    repo.git("tag", "-a", "evidence/run-orphaned", orphan, "-m", "anchor");
    const result = repo.check();
    assert.equal(result.status, 0, result.stderr);
    assert.match(result.stdout, /1 via evidence\/\* tags/);
  });
});

test("--base makes the check strict: a commit ahead of the base ref fails even though HEAD contains it", () => {
  withRepo((repo) => {
    const onMain = repo.commit("first");
    repo.git("checkout", "--quiet", "-b", "pr-branch");
    const onBranchOnly = repo.commit("run commit made on the PR branch");
    repo.writeManifest("run-on-branch", { gitCommit: onBranchOnly });
    repo.writeManifest("run-on-main", { gitCommit: onMain });

    assert.equal(repo.check().status, 0, "reachable from HEAD");
    const strict = repo.check("--base", "main");
    assert.equal(strict.status, 1, "not yet on main: a squash-merge would orphan it");
    assert.match(strict.stderr, new RegExp(onBranchOnly));
    assert.doesNotMatch(strict.stderr, new RegExp(onMain));
  });
});

test("fails on a manifest with no gitCommit, on one citing a commit absent from the repository, and on invalid JSON", () => {
  withRepo((repo) => {
    repo.commit("first");
    repo.writeManifest("run-no-commit", { seeds: [1] });
    repo.writeManifest("run-unknown-commit", { gitCommit: "0123456789abcdef0123456789abcdef01234567" });
    mkdirSync(join(repo.dir, "artifacts", "run-bad-json"), { recursive: true });
    writeFileSync(join(repo.dir, "artifacts", "run-bad-json", "manifest.json"), "{ not json");

    const result = repo.check();
    assert.equal(result.status, 1);
    assert.match(result.stderr, /MALFORMED\s+artifacts\/run-no-commit\/manifest\.json: no gitCommit/);
    assert.match(result.stderr, /MALFORMED\s+artifacts\/run-bad-json\/manifest\.json: not valid JSON/);
    assert.match(result.stderr, /UNREACHABLE\s+0123456789abcdef0123456789abcdef01234567: commit object is not in this repository/);
  });
});

test("manifests under artifacts/_scratch are not evidence and are skipped, even when they cite an orphaned commit", () => {
  withRepo((repo) => {
    const onMain = repo.commit("first");
    const orphan = orphanCommit(repo);
    repo.writeManifest("run-a", { gitCommit: onMain });
    repo.writeManifest("_scratch/0001-card/some-run", { schemaVersion: 2, env: { gitCommit: orphan } });

    const result = repo.check();
    assert.equal(result.status, 0, result.stderr);
    assert.match(result.stdout, /1 manifests, 1 distinct commits/);

    // Only the top-level artifacts/_scratch is special: a run directory merely named _scratch is checked.
    repo.writeManifest("run-b/_scratch", { gitCommit: orphan });
    assert.equal(repo.check().status, 1);
  });
});

test("exits 2, not 0, when it cannot check at all (no artifacts directory, or an unresolvable base ref)", () => {
  withRepo((repo) => {
    repo.commit("first");
    assert.equal(repo.check().status, 2, "no artifacts/ directory");

    repo.writeManifest("run-a", { gitCommit: repo.git("rev-parse", "HEAD") });
    assert.equal(repo.check("--base", "origin/does-not-exist").status, 2, "unresolvable base ref");
  });
});
