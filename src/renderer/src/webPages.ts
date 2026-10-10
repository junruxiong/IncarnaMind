/**
 * Saving web pages as Documents (#261), from "Add a web page…", a paste on
 * the sidebar or a drop on the window, and "Fetch again": what is being
 * fetched, and the pages that couldn't be, until dismissed or tried again.
 */
import { create } from "zustand";
import type { Document, WebPageFailureReason, WebPageResult } from "../../core/api";
import type { Language } from "../../core/language";
import type { MessageKey } from "../../shared/i18n";
import { pastedAddresses } from "../../shared/webAddresses";
import { core } from "./core";
import { errorMessage } from "./errors";
import { useAppStore } from "./store";

/** A page that couldn't be saved, or fetched again. */
export interface WebPageProblem {
  /** The address, or for "Fetch again", the Document's id. */
  key: string;
  address: string;
  reason: WebPageFailureReason;
  message: string;
  /** The Folder it was to be saved in. */
  folderId: string | null;
  /** The Document fetched again. */
  document: { id: string; name: string } | null;
}

interface WebPagesState {
  dialogOpen: boolean;
  /**
   * The addresses "Add a web page…" is saving: it shows how each goes, and
   * why one couldn't be added, where the sidebar's status would otherwise.
   */
  dialogAddresses: readonly string[];
  /** The addresses being saved and the Documents being fetched again, by key. */
  fetching: readonly string[];
  problems: readonly WebPageProblem[];
  openDialog(): void;
  /** Saves the addresses typed in the dialog, which shows how each goes. */
  submitDialog(addresses: readonly string[]): Promise<void>;
  closeDialog(): void;
  add(addresses: readonly string[], folderId?: string | null): Promise<void>;
  fetchAgain(document: Pick<Document, "id" | "name">): Promise<void>;
  tryAgain(problem: WebPageProblem): Promise<void>;
  dismiss(problem: WebPageProblem): void;
}

/** What the sidebar's status shows: the pages being saved or that failed outside the dialog. */
export const outsideDialog = (state: Pick<WebPagesState, "dialogAddresses">, key: string) =>
  !state.dialogAddresses.includes(key);

/** Fetched at most this many at once: a paste of many links doesn't open them all together. */
const AT_ONCE = 4;

/** Reasons a page wasn't saved because of what it is, rather than because it couldn't be fetched. */
const REFUSED: ReadonlySet<WebPageFailureReason> = new Set([
  "invalid-address",
  "not-a-page",
  "video",
  "not-text",
]);

/** Whether trying again may help: the page couldn't be fetched, rather than being refused. */
export const canTryAgain = (problem: Pick<WebPageProblem, "reason">) =>
  !REFUSED.has(problem.reason);

export const reasonKey = (reason: WebPageFailureReason): MessageKey => `webPages.reason.${reason}`;

/** The web addresses a paste or a drop carries, if it carries only those (see `pastedAddresses`). */
export function linksIn(data: DataTransfer): string[] {
  const list = data
    .getData("text/uri-list")
    .split(/\r?\n/)
    .filter((line) => line && !line.startsWith("#"))
    .join("\n");
  return pastedAddresses(list || data.getData("text/plain"));
}

/** A date as the page or the fetch gives it, as "10 Oct 2026" or "2026年10月10日"; a bare year stays. */
export function pageDate(value: string, language: Language): string {
  if (!/^\d{4}-\d{2}/.test(value)) return value;
  const time = Date.parse(value);
  if (Number.isNaN(time)) return value;
  return new Intl.DateTimeFormat(language === "zh-CN" ? "zh-CN" : "en-GB", {
    day: "numeric",
    month: "short",
    year: "numeric",
  }).format(time);
}

export const useWebPages = create<WebPagesState>()((set, get) => {
  const start = (key: string) => {
    set((state) => ({
      fetching: [...state.fetching, key],
      problems: state.problems.filter((each) => each.key !== key),
    }));
  };
  const finish = (key: string, problem: WebPageProblem | null) => {
    set((state) => ({
      fetching: state.fetching.filter((each) => each !== key),
      problems: problem ? [...state.problems, problem] : state.problems,
    }));
  };
  const problemOf = (
    key: string,
    result: Extract<WebPageResult, { outcome: "failed" }>,
    folderId: string | null,
    document: WebPageProblem["document"],
  ): WebPageProblem => ({
    key,
    address: result.address,
    reason: result.failure.reason,
    message: result.failure.message,
    folderId,
    document,
  });
  const failedAction = (error: unknown) =>
    useAppStore.setState({ actionError: errorMessage(error) });

  const save = async (address: string, folderId: string | null) => {
    if (get().fetching.includes(address)) return;
    start(address);
    try {
      const result = await core.saveWebPage({ address, folderId });
      if (result.outcome === "failed") {
        finish(address, problemOf(address, result, folderId, null));
        return;
      }
      finish(address, null);
      if (result.existing) {
        useAppStore.setState((state) => ({
          alreadyAdded: [...new Set([...state.alreadyAdded, result.document.name])],
        }));
      }
    } catch (error) {
      finish(address, null);
      failedAction(error);
    }
  };

  return {
    dialogOpen: false,
    dialogAddresses: [],
    fetching: [],
    problems: [],

    openDialog() {
      set({ dialogOpen: true, dialogAddresses: [] });
    },
    async submitDialog(addresses) {
      set({ dialogAddresses: [...new Set(addresses)] });
      await get().add(addresses);
    },
    closeDialog() {
      set((state) => ({
        dialogOpen: false,
        dialogAddresses: [],
        // What it said has been read; a page still being fetched says how it went in the sidebar.
        problems: state.problems.filter((each) => !state.dialogAddresses.includes(each.key)),
      }));
    },

    async add(addresses, folderId = null) {
      const queue = [...new Set(addresses)];
      const worker = async () => {
        for (let next = queue.shift(); next !== undefined; next = queue.shift()) {
          await save(next, folderId);
        }
      };
      await Promise.all(Array.from({ length: Math.min(AT_ONCE, queue.length) }, worker));
    },

    async fetchAgain(document) {
      const key = `document:${document.id}`;
      if (get().fetching.includes(key)) return;
      start(key);
      try {
        const result = await core.fetchWebPageAgain(document.id);
        finish(
          key,
          result.outcome === "failed"
            ? problemOf(key, result, null, { id: document.id, name: document.name })
            : null,
        );
      } catch (error) {
        finish(key, null);
        failedAction(error);
      }
    },

    async tryAgain(problem) {
      if (problem.document) return get().fetchAgain(problem.document);
      await save(problem.address, problem.folderId);
    },

    dismiss(problem) {
      set((state) => ({ problems: state.problems.filter((each) => each.key !== problem.key) }));
    },
  };
});
