import { mkdirSync } from "node:fs";
import { join } from "node:path";
import type { CoreAdapters } from "./adapters";
import type { CoreApi, CoreEventSource, Unsubscribe } from "./api";
import { createDocuments, parseListOptions } from "./documents";
import { type AnyEventListener, createEventHub } from "./events";
import { createFolders, parseFolderId } from "./folders";
import { createMindContent } from "./mindContent";
import { createMinds, parseMindId } from "./minds";
import { createSettings } from "./settings";
import { migrate, openDatabase } from "./storage";

export const DATABASE_FILE = "incarnamind.db";

/** The core as its host sees it: the public interface (methods and events) plus host-only hooks. */
export interface Core extends CoreApi, CoreEventSource {
  /** Every event the core emits, for the host to forward to the UI. */
  onAnyEvent(listener: AnyEventListener): Unsubscribe;
  /** Closes the database and drops all listeners. The core can't be used afterwards. Safe to call twice. */
  close(): void;
}

/**
 * Opens (or creates) the data folder and its database, brings the schema up to
 * date, and returns the core. The host passes in every platform capability.
 */
export function createCore(adapters: CoreAdapters): Core {
  const { dataDir } = adapters.paths;
  mkdirSync(dataDir, { recursive: true });

  const db = openDatabase(join(dataDir, DATABASE_FILE));
  try {
    migrate(db);
  } catch (error) {
    db.close();
    throw error;
  }

  const now = () => (adapters.now?.() ?? new Date()).toISOString();
  const events = createEventHub();
  const minds = createMinds(db, now);
  const mindsChanged = () => events.emit("minds.changed", minds.list());
  const content = createMindContent(db, now, (mindId, update) => {
    events.emit("mind.update", { mindId, update });
    if (minds.markEdited(mindId)) mindsChanged();
  });
  const settings = createSettings(db, now, adapters.systemLanguages);
  const folders = createFolders(db, now);
  const foldersChanged = () => events.emit("folders.changed", folders.list());
  let documents: ReturnType<typeof createDocuments>;
  try {
    documents = createDocuments({
      db,
      dataDir,
      now,
      emitStatus: (document) => events.emit("document.status", document),
    });
  } catch (error) {
    db.close();
    throw error;
  }

  // Async on purpose: the renderer reaches these over IPC, and a future hosted core may be remote.
  return {
    createMind: async (input) => {
      const mind = minds.create(input);
      mindsChanged();
      return mind;
    },
    listMinds: async () => minds.list(),
    renameMind: async (mindId, title) => {
      const mind = minds.rename(mindId, title);
      mindsChanged();
      return mind;
    },
    deleteMind: async (mindId) => {
      const at = now();
      db.transaction(() => {
        const mind = minds.delete(mindId, at);
        content.remove(mind.id, at);
      });
      mindsChanged();
    },
    openMind: async (mindId) => {
      const mind = minds.get(mindId);
      return { mind, state: content.state(mind.id) };
    },
    applyMindUpdate: async (mindId, update) => {
      content.apply(minds.get(mindId).id, update);
    },
    closeMind: async (mindId) => {
      content.close(parseMindId(mindId));
    },
    getSettings: async () => settings.get(),
    updateSettings: async (patch) => {
      const updated = settings.update(patch);
      events.emit("settings.changed", updated);
      return updated;
    },
    addDocuments: (paths) => documents.add(paths),
    listDocuments: async (options) => {
      const { folderId, includeSubfolders } = parseListOptions(options);
      if (folderId === undefined) return documents.list();
      const folder = folders.get(folderId);
      return documents.list(includeSubfolders ? folders.subtree(folder.id) : [folder.id]);
    },
    renameDocument: async (id, name) => documents.rename(id, name),
    deleteDocument: (id) => documents.delete(id),
    searchPassages: async (query, limit) => documents.search(query, limit),
    moveDocument: async (documentId, folderInput) => {
      const folderId = folderInput === null ? null : parseFolderId(folderInput);
      const { document, moved } = db.transaction(() => {
        if (folderId !== null) folders.get(folderId);
        return documents.move(documentId, folderId);
      });
      if (moved) events.emit("documents.moved", [document]);
      return document;
    },
    createFolder: async (input) => {
      const folder = folders.create(input);
      foldersChanged();
      return folder;
    },
    listFolders: async () => folders.list(),
    renameFolder: async (folderId, name) => {
      const folder = folders.rename(folderId, name);
      foldersChanged();
      return folder;
    },
    moveFolder: async (folderId, parentId) => {
      const folder = folders.move(folderId, parentId);
      foldersChanged();
      return folder;
    },
    deleteFolder: async (folderId) => {
      const at = now();
      const unfiled = db.transaction(() => documents.unfile(folders.delete(folderId, at), at));
      if (unfiled.length > 0) events.emit("documents.moved", unfiled);
      foldersChanged();
    },
    on: (event, listener) => events.on(event, listener),
    onAnyEvent: (listener) => events.onAny(listener),
    close: () => {
      documents.close();
      events.clear();
      content.closeAll();
      db.close();
    },
  };
}
