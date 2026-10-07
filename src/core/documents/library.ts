/**
 * The User's library on disk (ADR-0010): Linked folders, and the files of
 * every Document, indexed where they are. This module keeps the index in step
 * with the disk, and only ever reads the User's files: it never writes,
 * renames or deletes anything in a Linked folder or next to a single file.
 *
 * It works in passes, one at a time. A pass looks at some paths (a whole
 * Linked folder at start-up, after a watcher error, or when linked; the paths
 * a watcher reported; a single file being opened) and compares what is on
 * disk with the Documents indexed there:
 * - a known file is read again only when its size or modified time changed,
 *   and then hashed with SHA-256: the same hash only updates what was seen; a
 *   new one is a new version, processed again;
 * - a new file is hashed: if a Document with that hash lost its file (missing
 *   since, gone in this pass, or a copy left in the data folder by the old
 *   layout), it is that Document, moved, keeping its id, Tags and Citations;
 *   otherwise it is a new Document. New files are indexed newest first;
 * - a known file that is gone makes its Document missing, or unavailable if
 *   what it was in can't be reached (an unplugged drive).
 *
 * Cloud placeholders (online-only files) are counted, never read, unless the
 * User asks to download them: reading one would download it.
 */
import { randomUUID } from "node:crypto";
import { open, realpath, rm } from "node:fs/promises";
import { basename, dirname, isAbsolute, join, sep } from "node:path";
import type {
  DocumentKind,
  LinkedFolder,
  LinkedFolderLayout,
  LinkedFolderPreview,
  LinkedFolderStatus,
  SkippedFile,
} from "../api";
import { InvalidInputError, NotFoundError } from "../errors";
import { type Folders, folderIdFor } from "../folders";
import type { Database } from "../storage";
import {
  createIgnore,
  type FoundFile,
  hashFile,
  iCloudStubTarget,
  isDataless,
  isFolder,
  isGone,
  isInside,
  kindOf,
  nameFromPath,
  parentOf,
  relativePath,
  statFile,
  walk,
} from "./files";
import type { FolderWatcher, WatchFolder } from "./watcher";

/** A Document as this module sees it. */
export interface FileRow {
  id: string;
  path: string;
  name: string;
  kind: string;
  content_hash: string;
  size: number;
  file_mtime_ms: number | null;
  file_status: string;
  linked_folder_id: string | null;
  folder_id: string | null;
  status: string;
}

const FILE_COLUMNS = `id, path, name, kind, content_hash, size, file_mtime_ms, file_status,
  linked_folder_id, folder_id, status`;

interface LinkedFolderRow {
  id: string;
  path: string;
  layout: string | null;
  paused: number;
  ignore_patterns: string;
  created_at: string;
  updated_at: string;
}

const FOLDER_COLUMNS = "id, path, layout, paused, ignore_patterns, created_at, updated_at";

/** Statuses of a Document once processing has finished with it. */
const PROCESSED = "('ready', 'no-text', 'failed')";

/**
 * A rough guess at indexing speed with the built-in model on a laptop CPU
 * (ADR-0009: about 25 Passages a second): a PDF holds about 3 seconds of work
 * per megabyte, plain text, which is much denser, about 30, and every file
 * costs a little on top. Office files are zipped, with images: a guess from
 * the samples' text per megabyte (unmeasured).
 */
const SECONDS_PER_FILE = 0.2;
const SECONDS_PER_MEGABYTE: Readonly<Record<DocumentKind, number>> = {
  pdf: 3,
  text: 30,
  markdown: 30,
  docx: 10,
  pptx: 2,
  xlsx: 20,
  csv: 30,
};

/** A folder is laid out flat when this share of its folders with files hold exactly one. */
const FLAT_SHARE = 0.9;
/**
 * …and there are at least this many such folders: a big, Zotero-like
 * library. A small one keeps its folders, which the User made on purpose.
 */
const FLAT_MIN_FOLDERS = 20;

/** How often Linked folders' progress reaches the UI, at most. */
const PROGRESS_INTERVAL_MS = 250;

/**
 * The layout a Linked folder gets: flat when nearly every folder in it that
 * holds supported files holds exactly one, as in Zotero's storage folder
 * (storage/<KEY>/paper.pdf). Takes the files' paths relative to it.
 */
export function suggestLayout(relatives: Iterable<string>): LinkedFolderLayout {
  const perFolder = new Map<string, number>();
  for (const relative of relatives) {
    const parent = parentOf(relative);
    if (parent !== "") perFolder.set(parent, (perFolder.get(parent) ?? 0) + 1);
  }
  if (perFolder.size < FLAT_MIN_FOLDERS) return "tree";
  const single = [...perFolder.values()].filter((count) => count === 1).length;
  return single / perFolder.size >= FLAT_SHARE ? "flat" : "tree";
}

/** What the library needs from the rest of the Documents module. */
export interface LibraryHooks {
  /** Processing is due (a new file, or a new version): queue it, unless its Linked folder is paused. */
  process(documentId: string): void;
  /** Push a Document's current state. */
  announce(documentId: string): void;
  /** These Documents moved to another Folder, or into or out of a Linked folder. */
  moved(documentIds: string[]): void;
  /** A Document's name followed its file's: its keyword index needs the new name. */
  renamed(documentId: string): void;
  /** Removes these Documents from the index, e.g. with their Linked folder. */
  remove(documentIds: string[], at: string): void;
  /** A Linked folder was paused (hold its queued work) or resumed (queue it again). */
  paused(linkedFolderId: string, paused: boolean): void;
  foldersChanged(): void;
  linkedFoldersChanged(list: LinkedFolder[]): void;
  /** Asks iCloud Drive to download the file an ".icloud" stub stands for. Optional. */
  downloadStub?: (path: string) => Promise<void>;
  reportError(error: unknown): void;
}

export interface LibraryOptions {
  db: Database;
  now: () => string;
  /** The data folder: Documents copied into its `documents/` by the old layout live there until relinked. */
  dataDir: string;
  folders: Folders;
  hooks: LibraryHooks;
  watch: WatchFolder;
  retryMs: number;
  detectDataless: boolean;
}

/** What one pass looks at. */
interface PassRequest {
  /** Every Linked folder, whole, and every single file. */
  all: boolean;
  /** These Linked folders, whole. */
  folders: Set<string>;
  /** These paths, and everything below them. */
  paths: Set<string>;
}

const emptyRequest = (): PassRequest => ({ all: false, folders: new Set(), paths: new Set() });

/** A file found on disk, with the Linked folder it is in. */
interface Found extends FoundFile {
  owner: LinkedFolderRow | null;
}

/** Whether a file can be opened for reading, without reading any of it. */
async function canOpen(path: string): Promise<boolean> {
  try {
    const handle = await open(path, "r");
    await handle.close();
    return true;
  } catch {
    return false;
  }
}

/** Strings from `from` (inclusive) to `to` (exclusive): every path inside the folder `prefix`. */
const insideRange = (prefix: string) => {
  const folder = prefix.endsWith(sep) ? prefix : `${prefix}${sep}`;
  const next = String.fromCharCode(sep.charCodeAt(0) + 1);
  return [folder, `${folder.slice(0, -1)}${next}`] as const;
};

export type Library = ReturnType<typeof createLibrary>;

export function createLibrary(options: LibraryOptions) {
  const { db, now, folders, hooks } = options;
  const legacyFolder = join(options.dataDir, "documents");
  const isLegacyCopy = (path: string) => path !== legacyFolder && isInside(legacyFolder, path);

  const watchers = new Map<string, FolderWatcher>();
  const statuses = new Map<string, Exclude<LinkedFolderStatus, "paused">>();
  /** Each Linked folder's cloud placeholders: path to size. */
  const onlineOnly = new Map<string, Map<string, number>>();
  /** Linked folders whose placeholders the User asked to download: read in their next whole pass. */
  const downloading = new Set<string>();
  /** Files found but not yet added in the pass under way, per Linked folder: counted in its progress. */
  const pendingNew = new Map<string, number>();
  /** New files that couldn't be read: tried again later. */
  const unreadable = new Set<string>();
  let closed = false;

  // --- Linked folder rows -------------------------------------------------

  const liveFolders = () =>
    db.all<LinkedFolderRow>(
      `SELECT ${FOLDER_COLUMNS} FROM linked_folders WHERE deleted_at IS NULL ORDER BY path`,
    );

  const folderRow = (id: string) =>
    db.get<LinkedFolderRow>(
      `SELECT ${FOLDER_COLUMNS} FROM linked_folders WHERE id = ? AND deleted_at IS NULL`,
      [id],
    );

  const ignoreOf = (row: LinkedFolderRow) => {
    let patterns: unknown;
    try {
      patterns = JSON.parse(row.ignore_patterns);
    } catch {
      patterns = [];
    }
    return createIgnore(
      Array.isArray(patterns) ? patterns.filter((each) => typeof each === "string") : [],
    );
  };

  /** The live Linked folder a path is in (the innermost, though they never nest), or undefined. */
  const ownerOf = (path: string, rows = liveFolders()) =>
    rows.filter((row) => isInside(row.path, path)).sort((a, b) => b.path.length - a.path.length)[0];

  const statusOf = (row: LinkedFolderRow): LinkedFolderStatus =>
    row.paused ? "paused" : (statuses.get(row.id) ?? "scanning");

  function toLinkedFolder(row: LinkedFolderRow): LinkedFolder {
    const counts = db.get<{ files: number; indexed: number | null }>(
      `SELECT count(*) AS files, sum(status IN ${PROCESSED}) AS indexed FROM documents
       WHERE linked_folder_id = ? AND deleted_at IS NULL AND file_status <> 'missing'`,
      [row.id],
    );
    const placeholders = onlineOnly.get(row.id) ?? new Map<string, number>();
    let bytes = 0;
    for (const size of placeholders.values()) bytes += size;
    return {
      id: row.id,
      path: row.path,
      folderId: folderIdFor(row.id, ""),
      status: statusOf(row),
      layout: row.layout === "flat" ? "flat" : "tree",
      progress: {
        files: (counts?.files ?? 0) + (pendingNew.get(row.id) ?? 0),
        indexed: counts?.indexed ?? 0,
      },
      onlineOnly: { files: placeholders.size, bytes, downloading: downloading.has(row.id) },
      createdAt: row.created_at,
      updatedAt: row.updated_at,
    };
  }

  const list = (): LinkedFolder[] => liveFolders().map(toLinkedFolder);

  const announceFolders = () => {
    if (!closed) hooks.linkedFoldersChanged(list());
  };

  /** Progress reaches the UI at most every PROGRESS_INTERVAL_MS. */
  let progressTimer: ReturnType<typeof setTimeout> | undefined;
  const announceProgress = () => {
    if (progressTimer || closed) return;
    progressTimer = setTimeout(() => {
      progressTimer = undefined;
      announceFolders();
    }, PROGRESS_INTERVAL_MS);
    progressTimer.unref?.();
  };

  const setStatus = (id: string, status: Exclude<LinkedFolderStatus, "paused">) => {
    if (statuses.get(id) === status) return;
    statuses.set(id, status);
    announceFolders();
  };

  // --- Passes, one at a time ------------------------------------------------

  let chain: Promise<unknown> = Promise.resolve();
  /** Runs `task` after every task queued before it. */
  function exclusive<T>(task: () => Promise<T>): Promise<T> {
    const run = chain.then(task, task);
    chain = run.catch(() => {});
    return run;
  }

  /** Work queued or under way; `idle()` waits for none. */
  let outstanding = 0;
  let idleWaiters: (() => void)[] = [];
  const begin = () => {
    outstanding++;
  };
  const end = () => {
    outstanding--;
    if (outstanding > 0) return;
    const waiters = idleWaiters;
    idleWaiters = [];
    for (const resolve of waiters) resolve();
  };
  const idle = () =>
    outstanding === 0
      ? Promise.resolve()
      : new Promise<void>((resolve) => idleWaiters.push(resolve));

  /** Changes reported but not yet looked at: one pass takes them all. */
  let pending = emptyRequest();
  let pendingTimer: ReturnType<typeof setTimeout> | undefined;

  /** Queues a pass over what `change` names, merged with any other queued meanwhile. */
  function request(change: {
    all?: boolean;
    folders?: Iterable<string>;
    paths?: Iterable<string>;
  }): void {
    if (closed) return;
    if (change.all) pending.all = true;
    for (const id of change.folders ?? []) pending.folders.add(id);
    for (const path of change.paths ?? []) pending.paths.add(path);
    if (pendingTimer) return;
    begin();
    // A moment's wait, so paths reported together (a folder renamed) go in one pass.
    pendingTimer = setTimeout(() => {
      pendingTimer = undefined;
      const taken = pending;
      pending = emptyRequest();
      exclusive(() => runTracked(taken))
        .catch((error: unknown) => hooks.reportError(error))
        .finally(end);
    }, 20);
    pendingTimer.unref?.();
  }

  /** Whether a pass is under way: a long one (a big folder linked) shouldn't hold up opening a file. */
  let passRunning = false;
  async function runTracked(taken: PassRequest): Promise<void> {
    passRunning = true;
    try {
      await runPass(taken);
    } finally {
      passRunning = false;
    }
  }

  // --- Watching -------------------------------------------------------------

  function startWatcher(row: LinkedFolderRow): void {
    if (closed || row.paused || watchers.has(row.id)) return;
    const ignore = ignoreOf(row);
    const watcher = options.watch(row.path, {
      onChange(paths) {
        const current = folderRow(row.id);
        if (!current || current.paused) return;
        const relevant = paths.filter((path) => {
          if (!isInside(row.path, path)) return false;
          if (iCloudStubTarget(basename(path))) return true;
          return !ignore(relativePath(row.path, path));
        });
        if (relevant.length > 0) request({ paths: relevant });
      },
      onError() {
        // Events may have been missed: look at the whole folder, then watch it again.
        watchers.delete(row.id);
        request({ folders: [row.id] });
      },
    });
    watchers.set(row.id, watcher);
  }

  const stopWatcher = (id: string) => {
    watchers.get(id)?.close();
    watchers.delete(id);
  };

  // --- Documents --------------------------------------------------------------

  const documentAt = (path: string) =>
    db.get<FileRow>(`SELECT ${FILE_COLUMNS} FROM documents WHERE path = ? AND deleted_at IS NULL`, [
      path,
    ]);

  const documentById = (id: string) =>
    db.get<FileRow>(`SELECT ${FILE_COLUMNS} FROM documents WHERE id = ? AND deleted_at IS NULL`, [
      id,
    ]);

  /** The live Documents at `path` or inside it. */
  const documentsUnder = (path: string) => {
    const [from, to] = insideRange(path);
    return db.all<FileRow>(
      `SELECT ${FILE_COLUMNS} FROM documents
       WHERE deleted_at IS NULL AND (path = ? OR (path >= ? AND path < ?))`,
      [path, from, to],
    );
  };

  /** Where a file belongs: its Linked folder and Folder, or neither for a single file. */
  const placeOf = (path: string, owner: LinkedFolderRow | null) =>
    owner
      ? {
          linkedFolderId: owner.id as string | null,
          folderId: folderIdFor(owner.id, parentOf(relativePath(owner.path, path))) as
            | string
            | null,
        }
      : { linkedFolderId: null, folderId: null };

  /** The User removed a Document at this path from this Linked folder: its file stays out. */
  const removedByUser = (path: string, linkedFolderId: string) =>
    db.get(
      "SELECT 1 FROM documents WHERE path = ? AND linked_folder_id = ? AND deleted_at IS NOT NULL",
      [path, linkedFolderId],
    ) !== undefined;

  /** The owner may have been removed or paused since the pass began: then its files are left alone. */
  const ownerStillActive = (owner: LinkedFolderRow | null) => {
    if (!owner) return true;
    const current = folderRow(owner.id);
    return current !== undefined && !current.paused;
  };

  /** Records what changed about a Document; the pass pushes it once at the end. */
  interface Changes {
    announced: Set<string>;
    moved: Set<string>;
    folders: Set<string>;
  }

  function setFileStatus(row: FileRow, status: string, changes: Changes): void {
    if (row.file_status === status) return;
    db.run("UPDATE documents SET file_status = ?, updated_at = ? WHERE id = ?", [
      status,
      now(),
      row.id,
    ]);
    changes.announced.add(row.id);
  }

  /**
   * A known Document whose file is still at its path: reads it again only if
   * its size or modified time changed. Returns false if it turned out to be
   * gone after all.
   */
  async function checkKnown(row: FileRow, file: Found, changes: Changes): Promise<boolean> {
    const place = placeOf(row.path, file.owner);
    const placeChanged =
      row.linked_folder_id !== place.linkedFolderId || row.folder_id !== place.folderId;
    const unchanged = row.size === file.size && row.file_mtime_ms === file.mtimeMs;
    // A placeholder isn't read: what was indexed of it stands.
    if (file.onlineOnly || unchanged) {
      // Back after being out of reach, or unreadable: it can be opened again, or still waits.
      if (row.file_status !== "available" && !file.onlineOnly && !(await canOpen(row.path))) {
        setFileStatus(row, "unavailable", changes);
        return true;
      }
      if (placeChanged || row.file_status !== "available") {
        db.run(
          `UPDATE documents SET linked_folder_id = ?, folder_id = ?, file_status = 'available',
             updated_at = ? WHERE id = ?`,
          [place.linkedFolderId, place.folderId, now(), row.id],
        );
        changes.announced.add(row.id);
      }
      if (placeChanged) noteMoved(row, place, changes);
      if (row.status === "queued") hooks.process(row.id);
      return true;
    }
    let hashed: Awaited<ReturnType<typeof hashFile>>;
    try {
      hashed = await hashFile(row.path);
    } catch (error) {
      if (isGone(error)) return false;
      setFileStatus(row, "unavailable", changes);
      return true;
    }
    const current = documentById(row.id);
    if (!current || current.path !== row.path) return true; // removed or moved meanwhile
    const newVersion = hashed.contentHash !== current.content_hash;
    db.run(
      `UPDATE documents SET size = ?, file_mtime_ms = ?, file_status = 'available',
         linked_folder_id = ?, folder_id = ?, updated_at = ?
       WHERE id = ?`,
      [hashed.size, file.mtimeMs, place.linkedFolderId, place.folderId, now(), row.id],
    );
    // Only what the UI shows is announced: a modified time alone isn't. (A new
    // version is announced as it is queued.)
    if (placeChanged || current.file_status !== "available") changes.announced.add(row.id);
    if (placeChanged) noteMoved(row, place, changes);
    // A new version is processed; its text replaces the old one's in search once indexed.
    if (newVersion || current.status === "queued") hooks.process(row.id);
    return true;
  }

  function noteMoved(
    row: Pick<FileRow, "id" | "linked_folder_id" | "folder_id">,
    place: { linkedFolderId: string | null; folderId: string | null },
    changes: Changes,
  ): void {
    if (row.folder_id !== place.folderId || row.linked_folder_id !== place.linkedFolderId) {
      changes.moved.add(row.id);
    }
    if (row.linked_folder_id) changes.folders.add(row.linked_folder_id);
    if (place.linkedFolderId) changes.folders.add(place.linkedFolderId);
  }

  /**
   * The Document a new file is, moved: one with the same content whose file
   * went, in this pass (`removed`) or before (missing), or whose copy in the
   * data folder the old layout made. Taken out of `removed` if found there.
   */
  async function movedDocument(
    contentHash: string,
    path: string,
    removed: FileRow[],
  ): Promise<FileRow | undefined> {
    const name = basename(path);
    const inPass = removed
      .filter((row) => row.content_hash === contentHash)
      .sort((a, b) => Number(basename(b.path) === name) - Number(basename(a.path) === name))[0];
    if (inPass) {
      removed.splice(removed.indexOf(inPass), 1);
      return inPass;
    }
    const candidates = db.all<FileRow>(
      `SELECT ${FILE_COLUMNS} FROM documents
       WHERE content_hash = ? AND path <> ? AND deleted_at IS NULL
       ORDER BY file_status = 'missing' DESC, created_at`,
      [contentHash, path],
    );
    for (const row of candidates) {
      if (row.file_status === "missing" || isLegacyCopy(row.path)) return row;
      // Gone, though no event has said so yet: moved before its old path was looked at.
      if (row.file_status === "available" && !(await statFile(row.path).catch(() => true))) {
        return row;
      }
    }
    return undefined;
  }

  /**
   * Points a Document at its file's new path, keeping its id, Tags and
   * Citations. Returns false if that can't be done any more: the Document
   * changed meanwhile, or another one is at the path.
   */
  async function relocate(
    row: FileRow,
    file: Found,
    size: number,
    changes: Changes,
  ): Promise<boolean> {
    const current = documentById(row.id);
    const there = documentAt(file.path);
    if (!current || current.path !== row.path || (there && there.id !== row.id)) return false;
    const place = placeOf(file.path, file.owner);
    // A name that was the file's name follows the file; one the User chose stays.
    const name = row.name === nameFromPath(row.path) ? nameFromPath(file.path) : row.name;
    db.run(
      `UPDATE documents SET path = ?, linked_folder_id = ?, folder_id = ?, file_status = 'available',
         size = ?, file_mtime_ms = ?, name = ?, updated_at = ?
       WHERE id = ? AND deleted_at IS NULL`,
      [file.path, place.linkedFolderId, place.folderId, size, file.mtimeMs, name, now(), row.id],
    );
    changes.announced.add(row.id);
    noteMoved(row, place, changes);
    if (name !== row.name) hooks.renamed(row.id);
    // The old layout's copy in the data folder isn't needed once the original is linked.
    if (current.status === "queued") hooks.process(row.id);
    if (isLegacyCopy(row.path)) {
      await rm(row.path, { force: true }).catch((error: unknown) => hooks.reportError(error));
    }
    return true;
  }

  /**
   * A file not indexed at its path: the Document it is, moved, or a new one.
   * Returns its Document's id, or undefined if it wasn't added.
   */
  async function addFound(
    file: Found,
    removed: FileRow[],
    changes: Changes,
    explicit: boolean,
  ): Promise<string | undefined> {
    if (!explicit && file.owner && removedByUser(file.path, file.owner.id)) return undefined;
    let hashed: Awaited<ReturnType<typeof hashFile>>;
    try {
      hashed = await hashFile(file.path);
    } catch (error) {
      if (!isGone(error)) unreadable.add(file.path);
      return undefined;
    }
    unreadable.delete(file.path);
    // A paused (or removed) Linked folder's files are left alone, unless the User adds one.
    if (closed || (!explicit && !ownerStillActive(file.owner))) return undefined;
    const moved = await movedDocument(hashed.contentHash, file.path, removed);
    if (closed) return undefined;
    if (moved && (await relocate(moved, file, hashed.size, changes))) return moved.id;
    // Checked just before writing, with nothing awaited in between: another task may have added it.
    const there = documentAt(file.path);
    if (there) return there.id;
    const place = placeOf(file.path, file.owner);
    const id = randomUUID();
    const at = now();
    db.run(
      `INSERT INTO documents (id, content_hash, name, kind, size, status, path, linked_folder_id,
         folder_id, file_status, file_mtime_ms, created_at, updated_at)
       VALUES (?, ?, ?, ?, ?, 'queued', ?, ?, ?, 'available', ?, ?, ?)`,
      [
        id,
        hashed.contentHash,
        nameFromPath(file.path),
        file.kind,
        hashed.size,
        file.path,
        place.linkedFolderId,
        place.folderId,
        file.mtimeMs,
        at,
        at,
      ],
    );
    hooks.announce(id);
    hooks.process(id);
    if (place.linkedFolderId) changes.folders.add(place.linkedFolderId);
    return id;
  }

  /**
   * A known Document whose file is gone: missing, or unavailable when what it
   * was in can't be reached (for a single file, its folder is gone too).
   */
  async function gone(row: FileRow, owner: LinkedFolderRow | null, changes: Changes) {
    const current = documentById(row.id);
    if (!current || current.path !== row.path) return;
    const status = owner || (await isFolder(dirname(row.path))) ? "missing" : "unavailable";
    setFileStatus(current, status, changes);
  }

  /** A Linked folder's root can't be reached: its Documents stay searchable, but can't be opened. */
  function unreachable(row: LinkedFolderRow, changes: Changes): void {
    stopWatcher(row.id);
    const affected = db.all<{ id: string }>(
      `SELECT id FROM documents
       WHERE linked_folder_id = ? AND deleted_at IS NULL AND file_status = 'available'`,
      [row.id],
    );
    if (affected.length > 0) {
      db.run(
        `UPDATE documents SET file_status = 'unavailable', updated_at = ?
         WHERE linked_folder_id = ? AND deleted_at IS NULL AND file_status = 'available'`,
        [now(), row.id],
      );
      for (const { id } of affected) changes.announced.add(id);
    }
    setStatus(row.id, "unavailable");
  }

  /** Makes a Linked folder's Folders those its Documents are in. Returns whether any changed. */
  function syncFolders(id: string): boolean {
    const row = folderRow(id);
    if (!row) return false;
    const relatives = db
      .all<{ path: string }>(
        "SELECT path FROM documents WHERE linked_folder_id = ? AND deleted_at IS NULL",
        [id],
      )
      .map((each) => parentOf(relativePath(row.path, each.path)));
    return folders.sync(id, row.path, relatives);
  }

  /** The Folders, events and statuses at the end of a pass. */
  function finish(changes: Changes, scanned: Set<string>): void {
    let foldersChanged = false;
    for (const id of changes.folders) if (syncFolders(id)) foldersChanged = true;
    for (const id of scanned) {
      const row = folderRow(id);
      if (!row) continue;
      if (row.layout === null) {
        const relatives = db
          .all<{ path: string }>(
            "SELECT path FROM documents WHERE linked_folder_id = ? AND deleted_at IS NULL",
            [id],
          )
          .map((each) => relativePath(row.path, each.path));
        db.run("UPDATE linked_folders SET layout = ?, updated_at = ? WHERE id = ?", [
          suggestLayout(relatives),
          now(),
          id,
        ]);
      }
      downloading.delete(id);
      if (!row.paused) {
        statuses.set(id, "watching");
        startWatcher(row);
      }
    }
    if (foldersChanged) hooks.foldersChanged();
    for (const id of changes.announced) hooks.announce(id);
    if (changes.moved.size > 0) hooks.moved([...changes.moved]);
    announceFolders();
  }

  async function runPass(request: PassRequest): Promise<void> {
    if (closed) return;
    const changes: Changes = { announced: new Set(), moved: new Set(), folders: new Set() };
    const linked = liveFolders();
    /** Documents looked at, by path, and files found there, by path. */
    const known = new Map<string, FileRow>();
    const found = new Map<string, Found>();
    /** Linked folders scanned whole and reachable. */
    const scanned = new Set<string>();

    for (const row of linked) {
      if (row.paused) continue;
      const whole = request.all || request.folders.has(row.id);
      const parts = whole
        ? [row.path]
        : topmost(
            [...request.paths].filter(
              (path) => isInside(row.path, path) && ownerOf(path, linked)?.id === row.id,
            ),
          );
      if (parts.length === 0) continue;
      if (!(await isFolder(row.path))) {
        unreachable(row, changes);
        continue;
      }
      if (whole) {
        statuses.set(row.id, "scanning");
        announceFolders();
      } else if (statuses.get(row.id) === "unavailable") {
        // Back after being out of reach: look at all of it.
        request.folders.add(row.id);
        parts.splice(0, parts.length, row.path);
        statuses.set(row.id, "scanning");
      }
      startWatcher(row); // before reading, so nothing that changes meanwhile is missed
      changes.folders.add(row.id);
      if (parts[0] === row.path) scanned.add(row.id);
      const ignore = ignoreOf(row);
      const placeholders = onlineOnly.get(row.id) ?? new Map<string, number>();
      onlineOnly.set(row.id, placeholders);
      for (const part of parts) {
        for (const document of documentsUnder(part)) known.set(document.path, document);
        let files: FoundFile[] = [];
        try {
          ({ files } = await walk(part, {
            root: row.path,
            ignore,
            detectDataless: options.detectDataless,
          }));
        } catch {
          // Gone: whatever was indexed there is gone too.
        }
        for (const path of placeholders.keys()) if (isInside(part, path)) placeholders.delete(path);
        for (const file of files) {
          if (!file.onlineOnly || known.has(file.path)) {
            // A known Document's placeholder isn't read either (see `checkKnown`).
            found.set(file.path, { ...file, owner: row });
          } else if (!downloading.has(row.id)) {
            placeholders.set(file.path, file.size);
          } else if (await statFile(file.path).catch(() => undefined)) {
            // Asked to download: reading a dataless file downloads it. (An
            // iCloud stub's file is only there once iCloud has downloaded it.)
            found.set(file.path, { ...file, onlineOnly: false, owner: row });
          } else {
            placeholders.set(file.path, file.size);
          }
        }
      }
    }

    // Files added on their own: each looked at by itself.
    const singles = request.all
      ? db.all<FileRow>(
          `SELECT ${FILE_COLUMNS} FROM documents WHERE deleted_at IS NULL AND linked_folder_id IS NULL`,
        )
      : [...request.paths].flatMap((path) =>
          documentsUnder(path).filter((row) => row.linked_folder_id === null),
        );
    for (const row of singles) {
      if (known.has(row.path) || ownerOf(row.path, linked)) continue;
      known.set(row.path, row);
      let info: Awaited<ReturnType<typeof statFile>>;
      try {
        info = await statFile(row.path);
      } catch {
        known.delete(row.path);
        setFileStatus(row, "unavailable", changes);
        continue;
      }
      if (info) {
        found.set(row.path, {
          path: row.path,
          kind: row.kind as DocumentKind,
          size: info.size,
          mtimeMs: info.mtimeMs,
          onlineOnly: options.detectDataless && isDataless(info),
          owner: null,
        });
      }
    }

    // Known files first: what is gone is known before new files are matched against it.
    const removed: FileRow[] = [];
    for (const row of known.values()) {
      const file = found.get(row.path);
      found.delete(row.path);
      if (closed) return;
      if (!file || !(await checkKnown(row, file, changes))) removed.push(row);
    }

    // New files, newest first.
    const fresh = [...found.values()].sort((a, b) => b.mtimeMs - a.mtimeMs);
    for (const file of fresh) {
      if (file.owner) pendingNew.set(file.owner.id, (pendingNew.get(file.owner.id) ?? 0) + 1);
    }
    announceFolders();
    try {
      for (const file of fresh) {
        if (closed) return;
        await addFound(file, removed, changes, false);
        if (file.owner) {
          pendingNew.set(file.owner.id, (pendingNew.get(file.owner.id) ?? 1) - 1);
          announceProgress();
        }
      }
    } finally {
      for (const file of fresh) if (file.owner) pendingNew.delete(file.owner.id);
    }

    // What is still unmatched has gone.
    for (const row of removed) {
      const owner = row.linked_folder_id ? (ownerOf(row.path, linked) ?? null) : null;
      await gone(row, owner, changes);
    }
    finish(changes, scanned);
  }

  /** The paths not inside another of them. */
  function topmost(paths: string[]): string[] {
    const sorted = [...new Set(paths)].sort((a, b) => a.length - b.length);
    const kept: string[] = [];
    for (const path of sorted) if (!kept.some((each) => isInside(each, path))) kept.push(path);
    return kept;
  }

  // --- Trying again -----------------------------------------------------------

  const retry = () => {
    if (closed) return;
    const linked = liveFolders();
    const folderIds = linked
      .filter((row) => !row.paused && statuses.get(row.id) === "unavailable")
      .map((row) => row.id);
    const unreachableIds = new Set(folderIds);
    const paths = [
      ...unreadable,
      ...db
        .all<{ path: string; linked_folder_id: string | null }>(
          `SELECT path, linked_folder_id FROM documents
           WHERE deleted_at IS NULL AND file_status = 'unavailable'`,
        )
        .filter((row) => row.linked_folder_id === null || !unreachableIds.has(row.linked_folder_id))
        .map((row) => row.path),
    ];
    if (folderIds.length > 0 || paths.length > 0) request({ folders: folderIds, paths });
    // A watched folder can vanish without an event (a drive unplugged): look for it.
    for (const row of linked) {
      if (row.paused || statuses.get(row.id) !== "watching") continue;
      void isFolder(row.path).then((there) => {
        if (!there) request({ folders: [row.id] });
      });
    }
  };
  const retryTimer = setInterval(retry, options.retryMs);
  retryTimer.unref?.();

  // --- The interface ----------------------------------------------------------

  async function resolveFolder(input: unknown): Promise<string> {
    if (typeof input !== "string" || !isAbsolute(input)) {
      throw new InvalidInputError("A Linked folder must be given as an absolute folder path.");
    }
    let real: string;
    try {
      real = await realpath(input);
    } catch {
      throw new NotFoundError("There is no folder at that path.");
    }
    if (!(await isFolder(real))) throw new InvalidInputError("That path isn't a folder.");
    return real;
  }

  const getRow = (idInput: unknown): LinkedFolderRow => {
    if (typeof idInput !== "string" || idInput === "") {
      throw new InvalidInputError("A Linked folder id is required.");
    }
    const row = folderRow(idInput);
    if (!row) throw new NotFoundError("That Linked folder doesn't exist or has been removed.");
    return row;
  };

  return {
    list,

    /** Starts watching every Linked folder, and reconciles everything with the disk. */
    start(): void {
      for (const row of liveFolders()) statuses.set(row.id, "scanning");
      request({ all: true });
    },

    /** Resolves when no pass is queued or under way. */
    idle,

    /** Reconciles everything with the disk now. Resolves when done. */
    async sync(): Promise<void> {
      request({ all: true });
      await idle();
    },

    /**
     * Reconciles these paths (e.g. a file about to be opened) now, and
     * resolves when done. While a pass is under way, which may take long, they
     * are queued for the next one instead, and it resolves at once.
     */
    async check(paths: readonly string[]): Promise<void> {
      if (passRunning) {
        request({ paths });
        return;
      }
      begin();
      try {
        await exclusive(() =>
          runTracked({ all: false, folders: new Set(), paths: new Set(paths) }),
        );
      } finally {
        end();
      }
    },

    async preview(input: unknown): Promise<LinkedFolderPreview> {
      const path = await resolveFolder(input);
      const linked = liveFolders();
      const outer = ownerOf(path, linked);
      const { files } = await walk(path, {
        root: path,
        ignore: createIgnore(),
        detectDataless: options.detectDataless,
      });
      const local = files.filter((file) => !file.onlineOnly);
      const placeholders = files.filter((file) => file.onlineOnly);
      let bytes = 0;
      let seconds = 0;
      for (const file of local) {
        bytes += file.size;
        seconds += SECONDS_PER_FILE + (file.size / 1_000_000) * SECONDS_PER_MEGABYTE[file.kind];
      }
      return {
        path,
        files: local.length,
        bytes,
        onlineOnly: {
          files: placeholders.length,
          bytes: placeholders.reduce((sum, file) => sum + file.size, 0),
        },
        estimatedSeconds: Math.ceil(seconds),
        layout: suggestLayout(local.map((file) => relativePath(path, file.path))),
        insideLinkedFolderId: outer?.id ?? null,
        containsLinkedFolderIds: linked
          .filter((row) => row.path !== path && isInside(path, row.path))
          .map((row) => row.id),
      };
    },

    /**
     * Links a folder (see `CoreApi.addLinkedFolder`). Nested Linked folders
     * are merged into the outer one: a folder inside one adds nothing, and
     * one around others takes their place, their Documents keeping their ids.
     */
    async add(input: unknown): Promise<LinkedFolder> {
      const path = await resolveFolder(input);
      const linked = liveFolders();
      const outer = ownerOf(path, linked);
      if (outer) return toLinkedFolder(outer);
      const at = now();
      const id = randomUUID();
      db.transaction(() => {
        db.run(
          `INSERT INTO linked_folders (id, path, created_at, updated_at) VALUES (?, ?, ?, ?)`,
          [id, path, at, at],
        );
        for (const inner of linked.filter((row) => isInside(path, row.path))) {
          // Its Documents become the new one's at its first scan: same paths.
          stopWatcher(inner.id);
          statuses.delete(inner.id);
          onlineOnly.delete(inner.id);
          db.run("UPDATE linked_folders SET deleted_at = ?, updated_at = ? WHERE id = ?", [
            at,
            at,
            inner.id,
          ]);
          folders.removeAll(inner.id, at);
        }
      });
      statuses.set(id, "scanning");
      request({ folders: [id] });
      announceFolders();
      const row = folderRow(id);
      if (!row) throw new Error("The Linked folder wasn't saved.");
      return toLinkedFolder(row);
    },

    /** Unlinks a folder: its Documents and Folders leave the index. Nothing on disk changes. */
    remove(idInput: unknown): void {
      const row = getRow(idInput);
      stopWatcher(row.id);
      const at = now();
      const ids = db
        .all<{ id: string }>(
          "SELECT id FROM documents WHERE linked_folder_id = ? AND deleted_at IS NULL",
          [row.id],
        )
        .map((each) => each.id);
      let foldersChanged = false;
      db.transaction(() => {
        hooks.remove(ids, at);
        db.run("UPDATE linked_folders SET deleted_at = ?, updated_at = ? WHERE id = ?", [
          at,
          at,
          row.id,
        ]);
        foldersChanged = folders.removeAll(row.id, at);
      });
      statuses.delete(row.id);
      onlineOnly.delete(row.id);
      downloading.delete(row.id);
      for (const path of unreadable) if (isInside(row.path, path)) unreadable.delete(path);
      if (foldersChanged) hooks.foldersChanged();
      announceFolders();
    },

    setLayout(idInput: unknown, layout: unknown): LinkedFolder {
      const row = getRow(idInput);
      if (layout !== "tree" && layout !== "flat") {
        throw new InvalidInputError('A Linked folder\'s layout is "tree" or "flat".');
      }
      db.run("UPDATE linked_folders SET layout = ?, updated_at = ? WHERE id = ?", [
        layout,
        now(),
        row.id,
      ]);
      announceFolders();
      return toLinkedFolder(getRow(row.id));
    },

    setPaused(idInput: unknown, paused: unknown): LinkedFolder {
      const row = getRow(idInput);
      if (typeof paused !== "boolean") throw new InvalidInputError("paused must be true or false.");
      if (Boolean(row.paused) !== paused) {
        db.run("UPDATE linked_folders SET paused = ?, updated_at = ? WHERE id = ?", [
          paused ? 1 : 0,
          now(),
          row.id,
        ]);
        if (paused) stopWatcher(row.id);
        hooks.paused(row.id, paused);
        if (!paused) {
          statuses.set(row.id, "scanning");
          // Nothing was read while it was paused: look at all of it.
          request({ folders: [row.id] });
        }
        announceFolders();
      }
      return toLinkedFolder(getRow(row.id));
    },

    /** Downloads and indexes the folder's online-only files: reading one downloads it. */
    downloadOnlineOnly(idInput: unknown): LinkedFolder {
      const row = getRow(idInput);
      downloading.add(row.id);
      for (const path of onlineOnly.get(row.id)?.keys() ?? []) {
        const stub = join(dirname(path), `.${basename(path)}.icloud`);
        if (hooks.downloadStub) {
          void statFile(stub)
            .then((info) => (info ? hooks.downloadStub?.(path) : undefined))
            .catch((error: unknown) => hooks.reportError(error));
        }
      }
      request({ folders: [row.id] });
      announceFolders();
      return toLinkedFolder(row);
    },

    /** Whether a Linked folder is paused: its Documents' work waits. */
    isPaused(linkedFolderId: string | null): boolean {
      if (!linkedFolderId) return false;
      return Boolean(folderRow(linkedFolderId)?.paused);
    },

    /** A Document's status changed: its Linked folder's progress may have. */
    progressed(linkedFolderId: string | null): void {
      if (linkedFolderId) announceProgress();
    },

    /** The User removed a Document from the index: its Folder may hold nothing now. */
    documentRemoved(linkedFolderId: string | null): void {
      if (!linkedFolderId) return;
      if (syncFolders(linkedFolderId)) hooks.foldersChanged();
      announceFolders();
    },

    /**
     * Adds files on their own (see `CoreApi.addDocuments`), in order: the
     * Document of each, or why it was skipped. Not held up by a pass under
     * way: every write checks what is there just before it.
     */
    addFiles(paths: readonly string[]): Promise<(string | SkippedFile["reason"])[]> {
      begin();
      return (async () => {
        const changes: Changes = { announced: new Set(), moved: new Set(), folders: new Set() };
        const results: (string | SkippedFile["reason"])[] = [];
        const linked = liveFolders();
        for (const path of paths) {
          const kind = kindOf(path);
          if (!kind) {
            results.push("unsupported-type");
            continue;
          }
          let real: string;
          let info: Awaited<ReturnType<typeof statFile>>;
          try {
            real = await realpath(path);
            info = await statFile(real);
          } catch {
            info = undefined;
            real = path;
          }
          if (!info) {
            results.push("unreadable");
            continue;
          }
          const inFolder = ownerOf(real, linked);
          const owner =
            inFolder && !ignoreOf(inFolder)(relativePath(inFolder.path, real)) ? inFolder : null;
          const file: Found = {
            path: real,
            kind,
            size: info.size,
            mtimeMs: info.mtimeMs,
            onlineOnly: false,
            owner,
          };
          const existing = documentAt(real);
          if (existing) {
            if (!(await checkKnown(existing, file, changes))) {
              await gone(existing, owner, changes);
              results.push("unreadable");
            } else {
              results.push(existing.id);
            }
            continue;
          }
          results.push((await addFound(file, [], changes, true)) ?? "unreadable");
        }
        finish(changes, new Set());
        return results;
      })().finally(end);
    },

    close(): void {
      closed = true;
      clearInterval(retryTimer);
      if (pendingTimer) {
        clearTimeout(pendingTimer);
        pendingTimer = undefined;
        end();
      }
      if (progressTimer) clearTimeout(progressTimer);
      for (const id of [...watchers.keys()]) stopWatcher(id);
    },
  };
}
