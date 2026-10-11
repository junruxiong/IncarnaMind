import { create } from "zustand";
import type {
  AddFoundSourceResult,
  FindSourcesResult,
  FoundSource,
  SourceProviderInfo,
} from "../../core/api";
import { core } from "./core";
import { errorMessage } from "./errors";
import { useAppStore } from "./store";

/** A work being added, or how adding it went, with the Folder it went to (null: Not in a Folder). */
export type AddState =
  | { status: "adding" }
  | { status: "done"; result: AddFoundSourceResult; folderId: string | null }
  | { status: "failed"; message: string };

/** What a search is for, as the panel's fields hold it. */
export interface SourceSearch {
  query: string;
  fromYear: number | null;
  toYear: number | null;
  openAccessOnly: boolean;
}

interface FindSourcesState extends SourceSearch {
  /** The panel shows in the Library. */
  open: boolean;
  /** The providers that search, with the User's consent decisions; empty until first opened. */
  providers: SourceProviderInfo[];
  /** Providers the User turned off for searches here, by id. */
  off: string[];
  /** Where added works go: a Folder, or null for Not in a Folder. */
  folderId: string | null;
  searching: boolean;
  /** The last search's results; null before any. */
  result: FindSourcesResult | null;
  /** What went wrong with the last search, in words. */
  error: string | null;
  /** Each work added this time, by key. */
  adds: Record<string, AddState>;
  /**
   * Opens Find sources in the Library, adding to the Folder shown there; with
   * a search (an Answer's), fills it in and runs it.
   */
  show(search?: Partial<SourceSearch>): void;
  close(): void;
  edit(change: Partial<SourceSearch & { folderId: string | null }>): void;
  toggleProvider(id: string): void;
  search(): Promise<void>;
  add(work: FoundSource): Promise<void>;
}

/** Counts searches, so only the latest one's answer is shown. */
let searchTicket = 0;

/** The Folder the Library shows, if it shows one: where Find sources adds by default. */
function shownFolder(): string | null {
  const { libraryFilter, library } = useAppStore.getState();
  return library?.groups.some((group) => group.id === libraryFilter) ? libraryFilter : null;
}

export const useFindSources = create<FindSourcesState>()((set, get) => ({
  open: false,
  query: "",
  fromYear: null,
  toYear: null,
  openAccessOnly: false,
  providers: [],
  off: [],
  folderId: null,
  searching: false,
  result: null,
  error: null,
  adds: {},

  show(search) {
    const app = useAppStore.getState();
    const folderId = get().open ? get().folderId : shownFolder();
    set({ open: true, folderId, ...search });
    // The Library, as a tab, as it was: the User's Mind stays where it is.
    app.openLibrary(app.libraryFilter === "new" ? "all" : app.libraryFilter);
    core.listSourceProviders().then(
      (providers) => set({ providers }),
      (error: unknown) => app.reportError(error),
    );
    if (search?.query) void get().search();
  },

  close() {
    set({ open: false });
  },

  edit(change) {
    set(change);
  },

  toggleProvider(id) {
    const { off } = get();
    set({ off: off.includes(id) ? off.filter((each) => each !== id) : [...off, id] });
  },

  async search() {
    const { query, fromYear, toYear, openAccessOnly, providers, off } = get();
    if (query.trim() === "") return;
    const ticket = ++searchTicket;
    set({ searching: true, error: null });
    try {
      const searching = providers.filter((each) => each.searches).map((each) => each.id);
      const result = await core.findSources({
        query,
        fromYear,
        toYear,
        openAccessOnly,
        ...(providers.length > 0 && { providers: searching.filter((id) => !off.includes(id)) }),
      });
      // A search started since has the last word.
      if (ticket === searchTicket) set({ result, adds: {} });
    } catch (error) {
      if (ticket === searchTicket) set({ error: errorMessage(error) });
    } finally {
      if (ticket === searchTicket) set({ searching: false });
      // A consent decision may have changed what each provider shows.
      core.listSourceProviders().then(
        (providers) => set({ providers }),
        () => undefined,
      );
    }
  },

  async add(work) {
    const { folderId } = get();
    set((state) => ({ adds: { ...state.adds, [work.key]: { status: "adding" } } }));
    let next: AddState;
    try {
      const result = await core.addFoundSource({ key: work.key, folderId });
      next = { status: "done", result, folderId };
    } catch (error) {
      next = { status: "failed", message: errorMessage(error) };
    }
    set((state) => ({ adds: { ...state.adds, [work.key]: next } }));
  },
}));
