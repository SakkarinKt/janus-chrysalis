import { test } from "node:test";
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { existsSync, mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, sep } from "node:path";
import { execSync } from "node:child_process";
import {
  captureRunEnvironment,
  ensureRunDir,
  writeManifest,
  writeTelemetry,
} from "../../src/experiment/telemetry.ts";

/** A fresh, empty scratch dir per test, cleaned up after — never one under `artifacts/`. */
function scratchDir(): string {
  return mkdtempSync(join(tmpdir(), "janus-telemetry-test-")) + sep;
}

test("captureRunEnvironment: gitCommit matches `git rev-parse HEAD`, nodeVersion matches process.version", () => {
  const env = captureRunEnvironment(null);
  assert.equal(env.gitCommit, execSync("git rev-parse HEAD").toString().trim());
  assert.equal(env.nodeVersion, process.version);
  assert.match(env.gitCommit, /^[0-9a-f]{40}$/);
});

test("captureRunEnvironment: records what tfjs it was told about, and null when told nothing", () => {
  // Reading tfjs itself is src/experiment/tfjsInfo.ts's job (tested in tfjsInfo.test.ts), so
  // this module stays usable where the native binding is not installed.
  const without = captureRunEnvironment(null);
  assert.equal(without.tfjsVersion, null);
  assert.equal(without.tfjsBackend, null);

  const withTfjs = captureRunEnvironment({ version: "4.22.0", backend: "tensorflow" });
  assert.equal(withTfjs.tfjsVersion, "4.22.0");
  assert.equal(withTfjs.tfjsBackend, "tensorflow");
});

test("captureRunEnvironment: records platform, arch, a dirty flag, and the hash of this checkout's package-lock.json", () => {
  const env = captureRunEnvironment(null);
  assert.equal(env.platform, process.platform);
  assert.equal(env.arch, process.arch);
  assert.equal(typeof env.gitDirty, "boolean");
  const expected = createHash("sha256").update(readFileSync("package-lock.json")).digest("hex");
  assert.equal(env.lockfileSha256, expected);
});

test("captureRunEnvironment: lockfileSha256 is null in a checkout without a package-lock.json", () => {
  const dir = mkdtempSync(join(tmpdir(), "janus-telemetry-repo-"));
  execSync("git init --quiet && git -c user.name=test -c user.email=test@example.invalid commit --quiet --allow-empty -m first", {
    cwd: dir,
    env: { ...process.env, GIT_CONFIG_GLOBAL: "/dev/null", GIT_CONFIG_SYSTEM: "/dev/null" },
  });
  assert.equal(captureRunEnvironment(null, dir).lockfileSha256, null);
  rmSync(dir, { recursive: true, force: true });
});

test("captureRunEnvironment: createdAt is a valid, round-trippable ISO timestamp", () => {
  const env = captureRunEnvironment(null);
  assert.equal(new Date(env.createdAt).toISOString(), env.createdAt);
});

test("writeTelemetry and writeManifest accept a directory with or without a trailing separator", () => {
  const withSeparator = scratchDir();
  const bare = withSeparator.slice(0, -sep.length);
  writeTelemetry(bare, [{ a: 1 }], "bare.jsonl");
  writeManifest(bare, { a: 1 }, "bare.json");
  writeTelemetry(withSeparator, [{ a: 1 }], "trailing.jsonl");
  for (const name of ["bare.jsonl", "bare.json", "trailing.jsonl"]) {
    assert.equal(existsSync(join(bare, name)), true, name);
  }
  rmSync(bare, { recursive: true, force: true });
});

test("ensureRunDir: creates a nested directory that doesn't exist yet", () => {
  const base = scratchDir();
  const nested = `${base}a/b/c/`;
  assert.equal(existsSync(nested), false);
  ensureRunDir(nested);
  assert.equal(existsSync(nested), true);
  rmSync(base, { recursive: true, force: true });
});

test("ensureRunDir: doesn't throw when the directory already exists", () => {
  const dir = scratchDir();
  assert.doesNotThrow(() => ensureRunDir(dir));
  rmSync(dir, { recursive: true, force: true });
});

test("writeTelemetry: one JSON-encoded record per line, in the given order", () => {
  const dir = scratchDir();
  const records = [{ step: 0, x: 1 }, { step: 1, x: 2 }, { step: 2, x: 3 }];
  const fileName = writeTelemetry(dir, records);
  const lines = readFileSync(`${dir}${fileName}`, "utf8").trimEnd().split("\n");
  assert.deepEqual(
    lines.map((line) => JSON.parse(line)),
    records,
  );
  rmSync(dir, { recursive: true, force: true });
});

test("writeTelemetry: returns the file name it wrote, defaulting to telemetry.jsonl", () => {
  const dir = scratchDir();
  const fileName = writeTelemetry(dir, [{ a: 1 }]);
  assert.equal(fileName, "telemetry.jsonl");
  assert.equal(existsSync(`${dir}telemetry.jsonl`), true);
  rmSync(dir, { recursive: true, force: true });
});

test("writeTelemetry: honors a custom file name", () => {
  const dir = scratchDir();
  const fileName = writeTelemetry(dir, [{ a: 1 }], "custom.jsonl");
  assert.equal(fileName, "custom.jsonl");
  assert.equal(existsSync(`${dir}custom.jsonl`), true);
  assert.equal(existsSync(`${dir}telemetry.jsonl`), false);
  rmSync(dir, { recursive: true, force: true });
});

test("writeManifest: writes pretty-printed JSON that parses back to an equal object", () => {
  const dir = scratchDir();
  const manifest = { runId: "test-run", seed: 42, nested: { ok: true, values: [1, 2, 3] } };
  writeManifest(dir, manifest);
  const written = readFileSync(`${dir}manifest.json`, "utf8");
  assert.deepEqual(JSON.parse(written), manifest);
  assert.ok(written.includes("\n  "), "expected 2-space-indented pretty-printing");
  assert.ok(written.endsWith("\n"), "expected a trailing newline");
  rmSync(dir, { recursive: true, force: true });
});

test("writeManifest: honors a custom file name", () => {
  const dir = scratchDir();
  writeManifest(dir, { a: 1 }, "custom-manifest.json");
  assert.equal(existsSync(`${dir}custom-manifest.json`), true);
  assert.equal(existsSync(`${dir}manifest.json`), false);
  rmSync(dir, { recursive: true, force: true });
});
