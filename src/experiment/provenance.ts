import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";

/**
 * Provenance primitives for the experiment harness (`scripts/exp.ts`, `scripts/repro.ts`):
 * deterministic serialisation, content hashes, and git state. Nothing here imports tfjs, so the
 * harness's bookkeeping can be tested on a machine that cannot run a model.
 */

/**
 * JSON with object keys sorted (by UTF-16 code unit) at every depth and no whitespace, so two
 * structurally equal values always serialise to the same bytes and therefore the same hash.
 *
 * Throws on anything `JSON.stringify` would silently alter: `undefined` (dropped from objects,
 * `null` in arrays), non-finite numbers (`null`), and non-plain objects such as `Date` or `Map`.
 * A hash that quietly ignores a field is worse than no hash.
 */
export function canonicalJson(value: unknown): string {
  return serialise(value, "$");
}

function serialise(value: unknown, path: string): string {
  if (value === null) {
    return "null";
  }
  if (typeof value === "string" || typeof value === "boolean") {
    return JSON.stringify(value);
  }
  if (typeof value === "number") {
    if (!Number.isFinite(value)) {
      throw new TypeError(`canonicalJson: non-finite number at ${path}`);
    }
    return JSON.stringify(value);
  }
  if (Array.isArray(value)) {
    return `[${value.map((item, index) => serialise(item, `${path}[${index}]`)).join(",")}]`;
  }
  if (typeof value === "object") {
    const prototype: unknown = Object.getPrototypeOf(value);
    if (prototype !== Object.prototype && prototype !== null) {
      throw new TypeError(`canonicalJson: non-plain object at ${path}`);
    }
    const record = value as Record<string, unknown>;
    const members = Object.keys(record)
      .sort()
      .map((key) => `${JSON.stringify(key)}:${serialise(record[key], `${path}.${key}`)}`);
    return `{${members.join(",")}}`;
  }
  throw new TypeError(`canonicalJson: unsupported ${typeof value} at ${path}`);
}

/** Lower-case hex SHA-256 of `data`. */
export function sha256(data: string | Uint8Array): string {
  return createHash("sha256").update(data).digest("hex");
}

/** SHA-256 of a file's bytes. */
export function hashFile(path: string): string {
  return sha256(readFileSync(path));
}

/**
 * Every file under `dir`, as POSIX-style paths relative to `dir`, sorted by code unit.
 *
 * Dotfiles and dot-directories are skipped. They are never part of a card or a run's outputs,
 * and one of them (`.DS_Store`) appears on its own on macOS: counting it would make the same
 * directory hash differently on two machines.
 */
export function listFiles(dir: string): string[] {
  const found: string[] = [];
  const walk = (absolute: string, relative: string): void => {
    for (const entry of readdirSync(absolute, { withFileTypes: true })) {
      if (entry.name.startsWith(".")) {
        continue;
      }
      const childRelative = relative === "" ? entry.name : `${relative}/${entry.name}`;
      if (entry.isDirectory()) {
        walk(join(absolute, entry.name), childRelative);
      } else if (entry.isFile()) {
        found.push(childRelative);
      }
    }
  };
  walk(dir, "");
  return found.sort();
}

export interface FileHash {
  /** POSIX-style path relative to the hashed directory. */
  path: string;
  sha256: string;
}

/** `listFiles(dir)` with each file's SHA-256. */
export function hashFiles(dir: string): FileHash[] {
  return listFiles(dir).map((path) => ({ path, sha256: hashFile(join(dir, path)) }));
}

/**
 * One hash for a set of files: SHA-256 over one `"<sha256>  <path>\n"` line per file, sorted by
 * path. That is `sha256sum`'s output format, so the value can be recomputed by hand, and it
 * depends only on paths and bytes, so it can equally be computed from git objects at a commit
 * (the pre-registration check does exactly that).
 */
export function hashFileSet(files: readonly FileHash[]): string {
  const lines = [...files]
    .sort((a, b) => (a.path < b.path ? -1 : a.path > b.path ? 1 : 0))
    .map((file) => `${file.sha256}  ${file.path}\n`);
  return sha256(lines.join(""));
}

/** `hashFileSet` over every non-dotfile under `dir`. Changes if any file is edited, added, removed or renamed. */
export function hashDir(dir: string): string {
  return hashFileSet(hashFiles(dir));
}

/**
 * `hashDir` computed from git objects: the hash `dir` (a repo-relative POSIX path) had at
 * `commit`, without checking anything out. Same file set (dotfiles skipped) and same line format
 * as `hashDir`, so for a tracked, unmodified directory the two agree. `cwd` must be the
 * repository root. Returns `null` if the directory has no files at that commit.
 */
export function hashDirAtCommit(commit: string, dir: string, cwd: string = process.cwd()): string | null {
  const prefix = `${dir.replace(/\/+$/, "")}/`;
  const listing = execFileSync("git", ["ls-tree", "-r", "-z", "--name-only", commit, "--", prefix], {
    cwd,
    encoding: "utf8",
    stdio: ["ignore", "pipe", "pipe"],
  });
  const files: FileHash[] = listing
    .split("\0")
    .filter((path) => path.startsWith(prefix))
    .map((path) => path.slice(prefix.length))
    .filter((relative) => !relative.split("/").some((segment) => segment.startsWith(".")))
    .map((relative) => ({
      path: relative,
      sha256: sha256(execFileSync("git", ["cat-file", "blob", `${commit}:${prefix}${relative}`], { cwd, stdio: ["ignore", "pipe", "pipe"] })),
    }));
  return files.length === 0 ? null : hashFileSet(files);
}

export interface GitState {
  /** `git rev-parse HEAD`. */
  gitCommit: string;
  /**
   * True if `git status --porcelain` reports a modified, staged or untracked file, other than
   * the exemptions in `GitStateOptions`. Ignored files never count.
   */
  gitDirty: boolean;
}

export interface GitStateOptions {
  /**
   * Directories (repo-relative, POSIX, no trailing slash) under which a *new* file (untracked,
   * or staged but never committed) does not make the tree dirty. A modified or deleted
   * committed file there still does.
   */
  allowUntrackedUnder?: readonly string[];
  /** Files (repo-relative, POSIX) whose modification, or first appearance, does not make the tree dirty. */
  allowChangesTo?: readonly string[];
}

function git(args: string[], cwd: string): string {
  return execFileSync("git", args, { cwd, encoding: "utf8", stdio: ["ignore", "pipe", "pipe"] }).trim();
}

/**
 * The current commit and whether the work tree is dirty. Throws if `cwd` is not inside a git
 * work tree.
 *
 * The options exist for one caller: the harness. A recorded run writes new files under
 * `artifacts/` and appends a line to the ledger, and those must not make the *next* run look
 * dirty, or two runs could never be recorded from the same commit. What they describe is
 * results, not the code that produced them.
 */
export function gitState(cwd: string = process.cwd(), options: GitStateOptions = {}): GitState {
  const allowUntrackedUnder = options.allowUntrackedUnder ?? [];
  const allowChangesTo = options.allowChangesTo ?? [];
  // -z: NUL-separated "XY path" records with no quoting; a rename adds one extra record holding
  // the original path, which starts with neither a status nor a space-padded prefix we rely on.
  const records = execFileSync("git", ["status", "--porcelain=v1", "-z"], { cwd, encoding: "utf8", stdio: ["ignore", "pipe", "pipe"] })
    .split("\0")
    .filter((record) => record !== "");
  let dirty = false;
  for (let index = 0; index < records.length; index++) {
    const record = records[index] ?? "";
    const status = record.slice(0, 2);
    const path = record.slice(3);
    if (status.startsWith("R") || status.startsWith("C")) {
      index += 1; // skip the rename/copy source record
      dirty = true;
      continue;
    }
    // "New" covers untracked and staged-but-never-committed; "changed" adds modifications, staged or not.
    const isNew = status === "??" || status === "A " || status === "AM";
    const isChanged = isNew || status === " M" || status === "M " || status === "MM";
    if (isNew && allowUntrackedUnder.some((dir) => path === `${dir}/` || path.startsWith(`${dir}/`))) {
      continue;
    }
    if (isChanged && allowChangesTo.includes(path)) {
      continue;
    }
    dirty = true;
  }
  return { gitCommit: git(["rev-parse", "HEAD"], cwd), gitDirty: dirty };
}
