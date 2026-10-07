/**
 * Reading a Skill to import, from a folder or a zip, into memory: every file
 * as a path inside the Skill and its bytes. Nothing that leads outside the
 * Skill gets in: zip entries whose paths climb out ("../", absolute paths)
 * and symbolic links whose targets are outside are refused, and links inside
 * become copies of what they point to, so a stored Skill holds no links.
 * Sizes are checked as files are read (`SKILL_LIMITS`). The Skill is the
 * folder (or zip) with SKILL.md at its top, or its one top-level folder.
 */
import { lstat, readdir, readFile, realpath, stat } from "node:fs/promises";
import { isAbsolute, join, relative, sep } from "node:path";
import { SKILL_LIMITS, type SkillImportError, type SkillImportErrorKind } from "../api";
import { readZip, type ZipEntry, ZipError } from "./zip";

export interface SkillSourceFile {
  /** Inside the Skill, "/" between folders, e.g. "references/guide.md". */
  path: string;
  data: Buffer;
}

export type SkillSource =
  | { ok: true; files: SkillSourceFile[] }
  | { ok: false; error: SkillImportError };

export type SkillLimits = { readonly [K in keyof typeof SKILL_LIMITS]: number };

export const INSTRUCTIONS_FILE = "SKILL.md";

/** Left out of a Skill: what operating systems and version control leave behind. */
const JUNK_FOLDERS = new Set(["__MACOSX", ".git"]);
const JUNK_FILES = new Set([".DS_Store", "Thumbs.db", "desktop.ini"]);

/** How many links in a row a zip's link may go through before it counts as broken. */
const MAX_LINK_HOPS = 8;

class SourceError extends Error {
  constructor(readonly detail: SkillImportError) {
    super(detail.message);
  }
}

const fail = (kind: SkillImportErrorKind, path: string | null, message: string) =>
  new SourceError({ kind, path, field: null, message });

/** Whether a path inside a Skill is junk an operating system or version control left behind. */
export const isJunk = (segments: readonly string[]) =>
  segments.some((segment) => JUNK_FOLDERS.has(segment)) || JUNK_FILES.has(segments.at(-1) ?? "");

/**
 * A path inside the Skill as its segments, or null if it leads outside:
 * absolute, on a drive, or climbing out with "..". Backslashes count as "/".
 */
export function insidePath(name: string): string[] | null {
  const normalized = name.replace(/\\/g, "/");
  if (normalized.startsWith("/") || /^[A-Za-z]:/.test(normalized) || normalized.includes("\0")) {
    return null;
  }
  const segments = normalized.split("/").filter((segment) => segment !== "" && segment !== ".");
  return segments.includes("..") ? null : segments;
}

/** Whether `target` (a real path) is `root` or inside it. */
export function isWithin(root: string, target: string): boolean {
  const path = relative(root, target);
  return path === "" || (!isAbsolute(path) && path.split(sep)[0] !== "..");
}

/** Counts files and bytes as they are read, failing as soon as a limit is passed. */
function budget(limits: SkillLimits) {
  let files = 0;
  let bytes = 0;
  return (path: string, size: number) => {
    files++;
    bytes += size;
    if (files > limits.maxFiles) {
      throw fail("too-many-files", null, `The Skill has more than ${limits.maxFiles} files.`);
    }
    if (bytes > limits.maxBytes) {
      throw fail(
        "too-large",
        path,
        `The Skill's files add up to more than ${limits.maxBytes} bytes.`,
      );
    }
  };
}

async function readFolder(root: string, limits: SkillLimits): Promise<SkillSourceFile[]> {
  const realRoot = await realpath(root);
  const files: SkillSourceFile[] = [];
  const count = budget(limits);

  /** `ancestors`: the real paths of the folders being read, to stop a link that loops back. */
  async function walk(segments: string[], ancestors: ReadonlySet<string>): Promise<void> {
    const folder = join(root, ...segments);
    const real = await realpath(folder);
    if (ancestors.has(real)) return;
    const inside = new Set(ancestors).add(real);
    const names = (await readdir(folder)).sort();
    for (const name of names) {
      const path = [...segments, name];
      if (isJunk(path)) continue;
      const shown = path.join("/");
      const full = join(folder, name);
      let stats = await lstat(full);
      if (stats.isSymbolicLink()) {
        let target: string;
        try {
          target = await realpath(full);
        } catch {
          throw fail("unreadable", shown, `${shown} is a link to something that isn't there.`);
        }
        if (!isWithin(realRoot, target)) {
          throw fail("link-outside", shown, `${shown} is a link to something outside the Skill.`);
        }
        stats = await stat(full);
      }
      if (stats.isDirectory()) {
        await walk(path, inside);
      } else if (stats.isFile()) {
        count(shown, stats.size);
        try {
          files.push({ path: shown, data: await readFile(full) });
        } catch (error) {
          throw fail("unreadable", shown, `${shown} can't be read: ${(error as Error).message}`);
        }
      }
      // Sockets, devices and the like aren't part of a Skill.
    }
  }

  await walk([], new Set());
  return files;
}

function readZipFile(bytes: Buffer, limits: SkillLimits): SkillSourceFile[] {
  let entries: ZipEntry[];
  try {
    entries = readZip(bytes, { maxBytes: limits.maxBytes, maxFiles: limits.maxFiles });
  } catch (error) {
    if (!(error instanceof ZipError)) throw error;
    throw fail(error.kind === "invalid" ? "invalid-zip" : error.kind, null, error.message);
  }

  const byPath = new Map<string, ZipEntry & { segments: string[] }>();
  for (const entry of entries) {
    const segments = insidePath(entry.name);
    if (!segments) {
      throw fail("path-traversal", entry.name, `${entry.name} would be written outside the Skill.`);
    }
    if (segments.length === 0 || isJunk(segments)) continue;
    byPath.set(segments.join("/"), { ...entry, segments });
  }

  /** Where a link leads, as a path inside the Skill; throws if it leads outside. */
  const resolveLink = (link: { segments: string[]; data: Buffer }, shown: string): string => {
    const target = link.data.toString("utf8").replace(/\\/g, "/");
    if (target.startsWith("/") || /^[A-Za-z]:/.test(target)) {
      throw fail("link-outside", shown, `${shown} is a link to something outside the Skill.`);
    }
    const resolved = link.segments.slice(0, -1);
    for (const segment of target.split("/")) {
      if (segment === "" || segment === ".") continue;
      if (segment !== "..") resolved.push(segment);
      else if (resolved.pop() === undefined) {
        throw fail("link-outside", shown, `${shown} is a link to something outside the Skill.`);
      }
    }
    return resolved.join("/");
  };

  const files: SkillSourceFile[] = [];
  const count = budget(limits);
  const add = (path: string, data: Buffer) => {
    count(path, data.length);
    files.push({ path, data });
  };
  for (const [path, entry] of byPath) {
    if (entry.kind === "file") {
      add(path, entry.data);
      continue;
    }
    if (entry.kind !== "symlink") continue;
    // A link inside the Skill becomes a copy of what it points to.
    let target = resolveLink(entry, path);
    for (let hops = 0; byPath.get(target)?.kind === "symlink"; hops++) {
      const next = byPath.get(target) as ZipEntry & { segments: string[] };
      if (hops >= MAX_LINK_HOPS) throw fail("unreadable", path, `${path} is a link that loops.`);
      target = resolveLink(next, path);
    }
    const pointed = byPath.get(target);
    if (pointed?.kind === "file") {
      add(path, pointed.data);
      continue;
    }
    // A link to a folder: a copy of the files in it.
    const prefix = `${target}/`;
    const inFolder = [...byPath].filter(
      ([other, each]) => other.startsWith(prefix) && each.kind === "file",
    );
    if (inFolder.length === 0 && pointed?.kind !== "directory") {
      throw fail("unreadable", path, `${path} is a link to something that isn't there.`);
    }
    for (const [other, each] of inFolder) add(`${path}/${other.slice(prefix.length)}`, each.data);
  }
  return files;
}

/** The files with the Skill's own folder as their root: where SKILL.md is. Null if it's nowhere. */
function rooted(files: SkillSourceFile[]): SkillSourceFile[] | null {
  if (files.some((file) => file.path === INSTRUCTIONS_FILE)) return files;
  const tops = new Set(files.map((file) => file.path.split("/")[0]));
  const [top] = tops;
  if (tops.size !== 1 || top === undefined) return null;
  const prefix = `${top}/`;
  if (!files.some((file) => file.path === `${prefix}${INSTRUCTIONS_FILE}`)) return null;
  return files.map((file) => ({ ...file, path: file.path.slice(prefix.length) }));
}

/** SKILL.md first, then the others in path order. */
export function inSkillOrder<T extends { path: string }>(files: readonly T[]): T[] {
  return [...files].sort((a, b) =>
    a.path === INSTRUCTIONS_FILE
      ? -1
      : b.path === INSTRUCTIONS_FILE
        ? 1
        : a.path < b.path
          ? -1
          : a.path > b.path
            ? 1
            : 0,
  );
}

/** Reads the Skill at `path`: a folder, or a zip file. */
export async function readSkillSource(
  path: string,
  limits: SkillLimits = SKILL_LIMITS,
): Promise<SkillSource> {
  try {
    let stats: Awaited<ReturnType<typeof stat>>;
    try {
      stats = await stat(path);
    } catch (error) {
      throw fail("unreadable", null, `${path} can't be opened: ${(error as Error).message}`);
    }
    let files: SkillSourceFile[];
    if (stats.isDirectory()) {
      files = await readFolder(path, limits);
    } else if (stats.isFile()) {
      if (stats.size > limits.maxBytes) {
        throw fail("too-large", null, `The zip is larger than ${limits.maxBytes} bytes.`);
      }
      files = readZipFile(await readFile(path), limits);
    } else {
      throw fail("unreadable", null, `${path} is neither a folder nor a file.`);
    }

    const skill = rooted(files);
    if (!skill) {
      throw fail("not-a-skill", null, "There is no SKILL.md at the top of the folder or zip.");
    }
    const instructions = skill.find((file) => file.path === INSTRUCTIONS_FILE);
    if (instructions && instructions.data.length > limits.maxInstructionsBytes) {
      throw fail(
        "too-large",
        INSTRUCTIONS_FILE,
        `SKILL.md is larger than ${limits.maxInstructionsBytes} bytes.`,
      );
    }
    return { ok: true, files: inSkillOrder(skill) };
  } catch (error) {
    if (error instanceof SourceError) return { ok: false, error: error.detail };
    if (error instanceof Error && "code" in error) {
      return {
        ok: false,
        error: { kind: "unreadable", path: null, field: null, message: error.message },
      };
    }
    throw error;
  }
}
