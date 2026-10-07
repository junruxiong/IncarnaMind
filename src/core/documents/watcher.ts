/**
 * Watching a Linked folder while the app runs, with `fs.watch` in recursive
 * mode: Node 24 supports it on macOS (FSEvents), Windows
 * (ReadDirectoryChangesW) and Linux (inotify on each folder, set up by Node),
 * so no dependency is needed.
 *
 * Events come in bursts (an editor saving, a download being written), and
 * name a path without saying what happened to it. Each path is reported once
 * it has settled: no event for `settleMs`, and its size and modified time the
 * same at two checks `settleMs` apart, so a file still being written is read
 * once it is complete. What changed is for the caller to work out.
 */
import { type FSWatcher, watch } from "node:fs";
import { lstat } from "node:fs/promises";
import { join } from "node:path";

export interface WatchListener {
  /** Paths (absolute) that changed and have settled: files or folders, there or gone. */
  onChange(paths: string[]): void;
  /**
   * The watcher failed or overflowed, so events may have been missed: the
   * caller reconciles the whole folder. The watcher has stopped.
   */
  onError(error: unknown): void;
}

export interface FolderWatcher {
  close(): void;
}

/** Starts watching a folder, at any depth. */
export type WatchFolder = (root: string, listener: WatchListener) => FolderWatcher;

/** A path's size, modified time and type, or "gone": what must stay the same for it to have settled. */
async function signatureOf(path: string): Promise<string> {
  try {
    const info = await lstat(path);
    return `${info.isDirectory() ? "d" : "f"}:${info.size}:${info.mtimeMs}`;
  } catch {
    return "gone";
  }
}

/** Watches with `fs.watch`, reporting each path once it has settled for `settleMs`. */
export function fsWatchFolder(settleMs: number): WatchFolder {
  return (root, listener) => {
    interface Pending {
      timer: ReturnType<typeof setTimeout>;
      signature?: string;
    }
    const pending = new Map<string, Pending>();
    let closed = false;
    let watcher: FSWatcher | undefined;

    const stop = () => {
      closed = true;
      watcher?.close();
      for (const entry of pending.values()) clearTimeout(entry.timer);
      pending.clear();
    };

    const fail = (error: unknown) => {
      if (closed) return;
      stop();
      listener.onError(error);
    };

    const schedule = (path: string, entry: Pending) => {
      entry.timer = setTimeout(() => void settle(path, entry), settleMs);
      entry.timer.unref?.();
    };

    const touched = (path: string) => {
      const entry = pending.get(path);
      if (entry) {
        clearTimeout(entry.timer);
        entry.signature = undefined;
        schedule(path, entry);
      } else {
        const created = { timer: undefined as unknown as ReturnType<typeof setTimeout> };
        pending.set(path, created);
        schedule(path, created);
      }
    };

    const settle = async (path: string, entry: Pending) => {
      const signature = await signatureOf(path);
      if (closed || pending.get(path) !== entry) return;
      if (entry.signature !== signature) {
        // Still changing, or not checked yet: look again once more time has passed.
        entry.signature = signature;
        schedule(path, entry);
        return;
      }
      pending.delete(path);
      listener.onChange([path]);
    };

    try {
      watcher = watch(root, { recursive: true, persistent: false }, (_event, filename) => {
        if (closed) return;
        // No name: something changed, but not said where. Look at the whole folder.
        touched(filename ? join(root, filename.toString()) : root);
      });
      watcher.on("error", fail);
    } catch (error) {
      queueMicrotask(() => fail(error));
    }
    return { close: stop };
  };
}
