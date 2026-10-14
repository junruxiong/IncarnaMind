/**
 * Importing a Word file as a Mind (#104), from "Import a Word file…" (the +
 * menu, ⌘K) or a .docx dropped on an open Mind: the import screen reads the
 * file first, says why it can't be imported, or asks whose document it is
 * (someone else's when it carries tracked changes or comments), then imports
 * it and opens the new Mind in a tab.
 */
import { create } from "zustand";
import type { WordFileCheck, WordOrigin, WordRefusalReason } from "../../core/api";
import type { MessageKey } from "../../shared/i18n";
import { core, files } from "./core";
import { useAppStore } from "./store";

interface WordImportState {
  /** The file the import screen is about; null while it is closed. */
  path: string | null;
  /** It was dropped on a Mind, so the screen also offers to add it as a Document, as other drops do. */
  dropped: boolean;
  /** What reading the file found; null while it is being read. */
  check: WordFileCheck | null;
  origin: WordOrigin;
  importing: boolean;
  /** Why importing refused it, when the file changed after it was read. */
  refusal: WordRefusalReason | null;
  /** "Import a Word file…": the system's open dialog, then the import screen. */
  pick(): Promise<void>;
  /** Opens the import screen on a file. */
  open(path: string, dropped: boolean): Promise<void>;
  setOrigin(origin: WordOrigin): void;
  confirm(): Promise<void>;
  /** For a dropped file: adds it as a Document, as a drop elsewhere would, instead. */
  addAsDocument(): Promise<void>;
  close(): void;
}

export const refusalKey = (reason: WordRefusalReason): MessageKey => `wordImport.refusal.${reason}`;

/** A file the import screen takes: a Word file, which it imports or says why it can't. */
export const isWordFile = (name: string) => /\.(docx|docm|doc|dotx|dotm|dot)$/i.test(name);

const reportError = (error: unknown) => useAppStore.getState().reportError(error);

export const useWordImport = create<WordImportState>()((set, get) => ({
  path: null,
  dropped: false,
  check: null,
  origin: "own",
  importing: false,
  refusal: null,

  async pick() {
    const path = await files.pickWordFile();
    if (path) await get().open(path, false);
  },

  async open(path, dropped) {
    set({ path, dropped, check: null, origin: "own", importing: false, refusal: null });
    try {
      const check = await core.checkWordFile(path);
      if (get().path !== path) return;
      set({ check, origin: check.origin });
    } catch (error) {
      set({ path: null });
      reportError(error);
    }
  },

  setOrigin: (origin) => set({ origin }),

  async confirm() {
    const { path, origin, importing } = get();
    if (!path || importing) return;
    set({ importing: true });
    try {
      const result = await core.importWordFile({ path, origin });
      if (!result.ok) {
        set({ importing: false, refusal: result.refusal });
        return;
      }
      get().close();
      await useAppStore.getState().showNewMind(result.mind);
    } catch (error) {
      set({ importing: false });
      reportError(error);
    }
  },

  async addAsDocument() {
    const { path } = get();
    get().close();
    if (path) await useAppStore.getState().addPaths([path]);
  },

  close: () => set({ path: null, check: null, importing: false, refusal: null }),
}));
