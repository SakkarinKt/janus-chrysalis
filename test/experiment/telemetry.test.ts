import { test } from "node:test";
import assert from "node:assert/strict";
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
  const env = captureRunEnvironment();
  assert.equal(env.gitCommit, execSync("git rev-parse HEAD").toString().trim());
  assert.equal(env.nodeVersion, process.version);
  assert.match(env.gitCommit, /^[0-9a-f]{40}$/);
});

test("captureRunEnvironment: tfjsBackend is a non-empty string", () => {
  const env = captureRunEnvironment();
  assert.equal(typeof env.tfjsBackend, "string");
  assert.ok(env.tfjsBackend.length > 0);
});

test("captureRunEnvironment: createdAt is a valid, round-trippable ISO timestamp", () => {
  const env = captureRunEnvironment();
  assert.equal(new Date(env.createdAt).toISOString(), env.createdAt);
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
