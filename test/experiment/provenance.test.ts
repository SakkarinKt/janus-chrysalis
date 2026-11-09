import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdirSync, mkdtempSync, renameSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { canonicalJson, gitState, hashDir, hashFileSet, hashFiles, listFiles, sha256 } from "../../src/experiment/provenance.ts";
import { makeFixtureRepo } from "./harnessFixture.ts";

function scratch(files: Record<string, string>): string {
  const dir = mkdtempSync(join(tmpdir(), "janus-provenance-"));
  for (const [path, content] of Object.entries(files)) {
    mkdirSync(join(dir, path, ".."), { recursive: true });
    writeFileSync(join(dir, path), content);
  }
  return dir;
}

test("canonicalJson: key order never changes the output, at any depth", () => {
  const a = { b: 1, a: { d: [1, { z: 1, y: 2 }], c: "x" } };
  const b = { a: { c: "x", d: [1, { y: 2, z: 1 }] }, b: 1 };
  assert.equal(canonicalJson(a), canonicalJson(b));
  assert.equal(canonicalJson(a), '{"a":{"c":"x","d":[1,{"y":2,"z":1}]},"b":1}');
});

test("canonicalJson: array order is significant", () => {
  assert.notEqual(canonicalJson([1, 2]), canonicalJson([2, 1]));
});

test("canonicalJson: throws on values JSON.stringify would silently alter", () => {
  assert.throws(() => canonicalJson({ a: undefined }), /unsupported undefined at \$\.a/);
  assert.throws(() => canonicalJson([NaN]), /non-finite number at \$\[0\]/);
  assert.throws(() => canonicalJson({ a: Infinity }), /non-finite/);
  assert.throws(() => canonicalJson({ when: new Date(0) }), /non-plain object at \$\.when/);
  assert.throws(() => canonicalJson(() => 1), /unsupported function/);
});

test("sha256: matches the published test vector for 'abc'", () => {
  assert.equal(sha256("abc"), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
});

test("listFiles: sorted POSIX-relative paths, nested directories included, dotfiles skipped", () => {
  const dir = scratch({ "b.txt": "b", "a/z.txt": "z", "a/y.txt": "y", ".DS_Store": "junk", ".hidden/x.txt": "x" });
  assert.deepEqual(listFiles(dir), ["a/y.txt", "a/z.txt", "b.txt"]);
  rmSync(dir, { recursive: true, force: true });
});

test("hashDir: equal for equal content, and changes on edit, add, remove and rename", () => {
  const files = { "card.json": "{}", "run.ts": "export {}" };
  const a = scratch(files);
  const b = scratch(files);
  const baseline = hashDir(a);
  assert.equal(hashDir(b), baseline, "same files, different directory");

  writeFileSync(join(b, "run.ts"), "export {};");
  assert.notEqual(hashDir(b), baseline, "edit");
  writeFileSync(join(b, "run.ts"), "export {}");
  assert.equal(hashDir(b), baseline, "edit reverted");

  writeFileSync(join(b, "extra.ts"), "");
  assert.notEqual(hashDir(b), baseline, "add");
  rmSync(join(b, "extra.ts"));

  renameSync(join(b, "run.ts"), join(b, "main.ts"));
  assert.notEqual(hashDir(b), baseline, "rename with identical bytes");
  rmSync(join(b, "main.ts"));
  assert.notEqual(hashDir(b), baseline, "remove");

  rmSync(a, { recursive: true, force: true });
  rmSync(b, { recursive: true, force: true });
});

test("hashDir: a dotfile appearing in the directory does not change the hash", () => {
  const dir = scratch({ "card.json": "{}" });
  const before = hashDir(dir);
  writeFileSync(join(dir, ".DS_Store"), "finder was here");
  assert.equal(hashDir(dir), before);
  rmSync(dir, { recursive: true, force: true });
});

test("hashFileSet: is the SHA-256 of sha256sum-format lines, independent of input order", () => {
  const dir = scratch({ "a.txt": "a", "b.txt": "b" });
  const files = hashFiles(dir);
  const expected = sha256(`${sha256("a")}  a.txt\n${sha256("b")}  b.txt\n`);
  assert.equal(hashFileSet(files), expected);
  assert.equal(hashFileSet([...files].reverse()), expected);
  rmSync(dir, { recursive: true, force: true });
});

test("gitState: clean after a commit; dirty for a modified, a staged and an untracked file; ignored files do not count", () => {
  const repo = makeFixtureRepo();
  try {
    repo.write("tracked.txt", "one");
    const commit = repo.commitAll("first");
    assert.deepEqual(gitState(repo.root), { gitCommit: commit, gitDirty: false });

    repo.write("artifacts/run/telemetry.jsonl", "ignored by .gitignore");
    assert.equal(gitState(repo.root).gitDirty, false, "ignored file");

    repo.write("tracked.txt", "two");
    assert.equal(gitState(repo.root).gitDirty, true, "modified");
    repo.git("add", "tracked.txt");
    assert.equal(gitState(repo.root).gitDirty, true, "staged");
    repo.commitAll("second");

    repo.write("untracked.txt", "new");
    assert.equal(gitState(repo.root).gitDirty, true, "untracked");
  } finally {
    repo.dispose();
  }
});

test("gitState: throws outside a git work tree", () => {
  const dir = mkdtempSync(join(tmpdir(), "janus-not-a-repo-"));
  assert.throws(() => gitState(dir));
  rmSync(dir, { recursive: true, force: true });
});
