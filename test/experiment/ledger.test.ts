import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { appendLedgerEntry, cardStatus, nextCard, readLedger, resultEntry } from "../../src/experiment/ledger.ts";
import type { LedgerEntry } from "../../src/experiment/ledger.ts";
import type { Card, LoadedCard } from "../../src/experiment/card.ts";
import { validCard } from "./harnessFixture.ts";

function entry(overrides: Partial<LedgerEntry> = {}): LedgerEntry {
  return {
    ledgerId: "20261109T000000000Z-abc1234",
    cardId: "0001",
    cardSha256: "c".repeat(64),
    kind: "confirmatory",
    gitCommit: "a".repeat(40),
    configSha256: "d".repeat(64),
    manifestPath: "artifacts/0001-fixture/20261109T000000000Z-abc1234/manifest.json",
    manifestSha256: "e".repeat(64),
    primary: { name: "total", value: 12 },
    verdict: "pass",
    createdAt: "2026-11-09T00:00:00.000Z",
    reproOf: null,
    match: null,
    ...overrides,
  };
}

/** A LoadedCard without touching the filesystem: status logic only reads `card` and `dirName`. */
function loaded(id: string, overrides: Partial<Card> = {}): LoadedCard {
  return { card: validCard({ id, ...overrides }), dirName: `${id}-card`, dir: `/nowhere/${id}-card`, sha256: "c".repeat(64) };
}

function ledgerFile(content = ""): string {
  const dir = mkdtempSync(join(tmpdir(), "janus-ledger-"));
  const path = join(dir, "ledger.jsonl");
  writeFileSync(path, content);
  return path;
}

test("appendLedgerEntry + readLedger: entries round-trip, in order, one canonical line each", () => {
  const path = ledgerFile();
  const first = entry({ ledgerId: "run-1" });
  const second = entry({ ledgerId: "run-2", cardId: "0002", verdict: "fail", primary: { name: "total", value: null } });
  appendLedgerEntry(path, first);
  appendLedgerEntry(path, second);

  assert.deepEqual(readLedger(path), [first, second]);
  const lines = readFileSync(path, "utf8").split("\n");
  assert.equal(lines.length, 3, "two lines and a trailing newline");
  assert.ok(lines[0]?.startsWith('{"cardId":"0001","cardSha256"'), "keys are sorted, so lines diff cleanly");
  rmSync(join(path, ".."), { recursive: true, force: true });
});

test("appendLedgerEntry: never rewrites what is there, even when the last line lacks a newline", () => {
  const original = JSON.stringify(entry({ ledgerId: "run-1" }));
  const path = ledgerFile(original);
  appendLedgerEntry(path, entry({ ledgerId: "run-2" }));
  const text = readFileSync(path, "utf8");
  assert.ok(text.startsWith(`${original}\n`), "the existing line is byte-for-byte intact");
  assert.equal(readLedger(path).length, 2);
  rmSync(join(path, ".."), { recursive: true, force: true });
});

test("appendLedgerEntry: refuses a malformed entry and leaves the file untouched", () => {
  const path = ledgerFile();
  assert.throws(() => appendLedgerEntry(path, { ...entry(), verdict: "maybe" } as unknown as LedgerEntry), /verdict must be/);
  assert.equal(readFileSync(path, "utf8"), "");
  rmSync(join(path, ".."), { recursive: true, force: true });
});

test("readLedger: a missing file is empty; blank lines are skipped", () => {
  assert.deepEqual(readLedger("/nonexistent/ledger.jsonl"), []);
  const path = ledgerFile(`\n${JSON.stringify(entry())}\n\n`);
  assert.equal(readLedger(path).length, 1);
  rmSync(join(path, ".."), { recursive: true, force: true });
});

test("readLedger: a malformed line throws with its line number instead of being skipped", () => {
  const good = JSON.stringify(entry({ ledgerId: "run-1" }));
  const cases: Array<[string, RegExp]> = [
    [`${good}\n{ not json\n`, /ledger\.jsonl:2: not valid JSON/],
    [`${good}\n${JSON.stringify({ ...entry({ ledgerId: "run-2" }), kind: "vibes" })}\n`, /:2: kind must be one of/],
    [`${good}\n${good}\n`, /:2: duplicate ledgerId run-1/],
    [`${JSON.stringify({ ...entry(), manifestSha256: "" })}\n`, /:1: manifestSha256 must be a non-empty string/],
  ];
  for (const [content, expected] of cases) {
    const path = ledgerFile(content);
    assert.throws(() => readLedger(path), expected);
    rmSync(join(path, ".."), { recursive: true, force: true });
  }
});

test("resultEntry: a card's result is its first non-repro entry; reproductions never count", () => {
  const ledger = [
    entry({ ledgerId: "repro-first", kind: "repro", reproOf: "x", match: true }),
    entry({ ledgerId: "the-result", verdict: "fail" }),
    entry({ ledgerId: "other-card", cardId: "0002" }),
  ];
  assert.equal(resultEntry("0001", ledger)?.ledgerId, "the-result");
  assert.equal(resultEntry("0003", ledger), undefined);
  assert.equal(resultEntry("0001", [ledger[0] as LedgerEntry]), undefined);
});

test("cardStatus: ready with no dependencies and no result; done once it has one", () => {
  const card = loaded("0001");
  assert.deepEqual(cardStatus(card, [card], []), { state: "ready" });
  assert.deepEqual(cardStatus(card, [card], [entry({ verdict: "fail" })]), {
    state: "done",
    verdict: "fail",
    ledgerId: "20261109T000000000Z-abc1234",
  });
});

test("cardStatus: waits on a dependency with no verdict, runs after the right one, and dies after the wrong one", () => {
  const first = loaded("0001");
  const second = loaded("0002", { dependsOn: [{ card: "0001", verdict: "pass" }] });
  const cards = [first, second];

  assert.deepEqual(cardStatus(second, cards, []), { state: "waiting", on: ["0001"] });
  assert.deepEqual(cardStatus(second, cards, [entry({ verdict: "pass" })]), { state: "ready" });

  const dead = cardStatus(second, cards, [entry({ verdict: "fail" })]);
  assert.equal(dead.state, "dead");
  assert.match(dead.state === "dead" ? dead.reason : "", /needs card 0001 to pass, but it reached "fail"/);
});

test("cardStatus: a reproduction of a dependency does not satisfy it", () => {
  const first = loaded("0001");
  const second = loaded("0002", { dependsOn: [{ card: "0001", verdict: "pass" }] });
  const reproOnly = [entry({ kind: "repro", reproOf: "x", match: true })];
  assert.deepEqual(cardStatus(second, [first, second], reproOnly), { state: "waiting", on: ["0001"] });
});

test("cardStatus: dead when a dependency does not exist, or when a later card supersedes this one before it ran", () => {
  const orphan = loaded("0002", { dependsOn: [{ card: "0007", verdict: "pass" }] });
  const status = cardStatus(orphan, [orphan], []);
  assert.match(status.state === "dead" ? status.reason : "", /depends on card 0007, which does not exist/);

  const old = loaded("0003");
  const replacement = loaded("0004", { supersedes: "0003" });
  const superseded = cardStatus(old, [old, replacement], []);
  assert.match(superseded.state === "dead" ? superseded.reason : "", /superseded by 0004-card/);
  assert.equal(cardStatus(old, [old, replacement], [entry({ cardId: "0003" })]).state, "done", "a result already recorded stands");
});

test("cardStatus: a repro card is always on demand, whatever the ledger says", () => {
  const canary = loaded("0000", { kind: "repro", config: { reproOf: "legacy:x" } });
  assert.deepEqual(cardStatus(canary, [canary], []), { state: "on-demand" });
  assert.deepEqual(cardStatus(canary, [canary], [entry({ cardId: "0000", kind: "repro" })]), { state: "on-demand" });
});

test("nextCard: the lowest-numbered ready card, skipping done, waiting, dead and repro cards; null when nothing is ready", () => {
  const canary = loaded("0000", { kind: "repro", config: { reproOf: "legacy:x" } });
  const first = loaded("0001");
  const second = loaded("0002", { dependsOn: [{ card: "0001", verdict: "pass" }] });
  const third = loaded("0003", { dependsOn: [{ card: "0002", verdict: "pass" }] });
  const cards = [canary, first, second, third];

  assert.equal(nextCard(cards, [])?.card.id, "0001");
  const afterFirst = [entry({ cardId: "0001", ledgerId: "r1", verdict: "pass" })];
  assert.equal(nextCard(cards, afterFirst)?.card.id, "0002");

  // The kill rule in miniature: 0002 fails, so 0003 can never run and the queue is empty.
  const afterSecondFails = [...afterFirst, entry({ cardId: "0002", ledgerId: "r2", verdict: "fail" })];
  assert.equal(nextCard(cards, afterSecondFails), null);
  assert.equal(cardStatus(third, cards, afterSecondFails).state, "dead");
});
