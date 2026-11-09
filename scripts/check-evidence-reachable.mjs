#!/usr/bin/env node
// Evidence-reachability guard.
//
// Every `artifacts/**/manifest.json` records the git commit its run executed at. A manifest is
// only reproducible while that commit can still be checked out. Two commits were nearly lost in
// 2026-09: PR #88 was squash-merged, which left the Gate G2 run commit (c3f0923) on no branch, and
// PR #52's head (4f080d8) survived only as a stale local ref. This script fails when a manifest
// cites a commit that is neither
//
//   (a) an ancestor of the base ref, nor
//   (b) contained in an `evidence/*` tag.
//
// Base ref: `--base <ref>` if given; else `origin/$GITHUB_BASE_REF` on a pull request (so a run
// commit that exists only on the PR branch fails *before* a squash-merge can orphan it); else
// `HEAD` (a push to main, or a local run).
//
// Fixing a failure: either get the run commit onto main first and re-run there, or anchor it with
// `git tag -a evidence/<run-dir-name> <commit>` and push the tag.
//
// Needs full history and tags: in CI, check out with `fetch-depth: 0` and `fetch-tags: true`.
//
// Exit codes: 0 all reachable · 1 unreachable or malformed evidence · 2 cannot run (bad usage,
// shallow clone, no artifacts directory).

import { execFileSync } from "node:child_process";
import { existsSync, readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";

const ARTIFACTS_DIR = "artifacts";
const EVIDENCE_TAG_GLOB = "evidence/*";

/** Runs git and returns trimmed stdout, or `null` if git exits non-zero. */
function git(args) {
  try {
    return execFileSync("git", args, { encoding: "utf8", stdio: ["ignore", "pipe", "ignore"] }).trim();
  } catch {
    return null;
  }
}

/** True when `git <args>` exits 0. For plumbing that answers by exit code alone. */
function gitOk(args) {
  return git(args) !== null;
}

function fail(code, message) {
  console.error(message);
  process.exit(code);
}

function parseBaseRef(argv) {
  const index = argv.indexOf("--base");
  if (index !== -1) {
    const ref = argv[index + 1];
    if (ref === undefined) {
      fail(2, "check-evidence-reachable: --base needs a ref");
    }
    return ref;
  }
  const prBase = process.env.GITHUB_BASE_REF;
  if (prBase !== undefined && prBase !== "") {
    return `origin/${prBase}`;
  }
  return "HEAD";
}

/** Scratch runs are gitignored and may cite a dirty or local-only commit; they are not evidence. */
const SCRATCH_DIR = join(ARTIFACTS_DIR, "_scratch");

function* findManifests(dir) {
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const path = join(dir, entry.name);
    if (path === SCRATCH_DIR) {
      continue;
    }
    if (entry.isDirectory()) {
      yield* findManifests(path);
    } else if (entry.name === "manifest.json") {
      yield path;
    }
  }
}

/** v1 manifests keep `gitCommit` at the top level; manifest v2 nests it under `env`. */
function recordedCommit(manifest) {
  const commit = manifest?.gitCommit ?? manifest?.env?.gitCommit;
  return typeof commit === "string" && /^[0-9a-f]{7,40}$/.test(commit) ? commit : null;
}

const baseRef = parseBaseRef(process.argv.slice(2));

if (!existsSync(ARTIFACTS_DIR)) {
  fail(2, `check-evidence-reachable: no ${ARTIFACTS_DIR}/ directory in ${process.cwd()}`);
}
if (git(["rev-parse", "--is-shallow-repository"]) === "true") {
  fail(
    2,
    "check-evidence-reachable: this is a shallow clone, so ancestry cannot be checked. " +
      "Fetch full history (CI: actions/checkout with fetch-depth: 0 and fetch-tags: true).",
  );
}
if (!gitOk(["rev-parse", "--verify", "--quiet", `${baseRef}^{commit}`])) {
  fail(2, `check-evidence-reachable: base ref ${baseRef} does not resolve to a commit`);
}

/** commit -> manifest paths citing it */
const citations = new Map();
const malformed = [];
let manifestCount = 0;

for (const path of findManifests(ARTIFACTS_DIR)) {
  manifestCount += 1;
  let manifest;
  try {
    manifest = JSON.parse(readFileSync(path, "utf8"));
  } catch (error) {
    malformed.push(`${path}: not valid JSON (${error.message})`);
    continue;
  }
  const commit = recordedCommit(manifest);
  if (commit === null) {
    malformed.push(`${path}: no gitCommit (expected a 7-40 character hex SHA at .gitCommit or .env.gitCommit)`);
    continue;
  }
  const paths = citations.get(commit) ?? [];
  paths.push(path);
  citations.set(commit, paths);
}

const unreachable = [];
let viaTag = 0;

for (const [commit, paths] of citations) {
  if (!gitOk(["cat-file", "-e", `${commit}^{commit}`])) {
    unreachable.push({ commit, paths, reason: "commit object is not in this repository" });
    continue;
  }
  if (gitOk(["merge-base", "--is-ancestor", commit, baseRef])) {
    continue;
  }
  const tags = git(["tag", "--list", EVIDENCE_TAG_GLOB, "--contains", commit]);
  if (tags !== null && tags !== "") {
    viaTag += 1;
    continue;
  }
  unreachable.push({ commit, paths, reason: `not an ancestor of ${baseRef} and in no ${EVIDENCE_TAG_GLOB} tag` });
}

if (malformed.length > 0 || unreachable.length > 0) {
  for (const line of malformed) {
    console.error(`MALFORMED  ${line}`);
  }
  for (const { commit, paths, reason } of unreachable) {
    console.error(`UNREACHABLE  ${commit}: ${reason}`);
    console.error(`  cited by ${paths.length} manifest(s), e.g. ${paths[0]}`);
  }
  console.error(
    `\ncheck-evidence-reachable: ${unreachable.length} unreachable commit(s), ${malformed.length} malformed ` +
      `manifest(s) out of ${manifestCount}. Anchor a commit with ` +
      "`git tag -a evidence/<run-dir-name> <commit>` and push the tag.",
  );
  process.exit(1);
}

console.log(
  `check-evidence-reachable: ${manifestCount} manifests, ${citations.size} distinct commits, all reachable ` +
    `(${citations.size - viaTag} via ${baseRef}, ${viaTag} via ${EVIDENCE_TAG_GLOB} tags).`,
);
