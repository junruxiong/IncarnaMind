import { create } from "zustand";
import type {
  CheckRecordFormat,
  ExportFormat,
  ExportMindOptions,
  ExportPreferences,
  SavedExport,
  SavedMindExport,
  SentenceChecksExported,
  WordExportReport,
} from "../../core/api";
import type { MessageKey } from "../../shared/i18n";
import { core } from "./core";
import { errorMessage } from "./errors";
import { translateNow } from "./i18n";
import { folderName } from "./linkedFolders";
import { useAppStore } from "./store";

/** Each format's name, as the options sheet, the menus and ⌘K say it. */
export const FORMAT_LABELS: Readonly<Record<ExportFormat, MessageKey>> = {
  docx: "export.format.docx",
  pdf: "export.format.pdf",
  "source-pack": "export.format.source-pack",
  html: "export.format.html",
  "share-page": "export.format.share-page",
  markdown: "export.format.markdown",
};

export const RECORD_FORMAT_LABELS: Readonly<Record<CheckRecordFormat, MessageKey>> = {
  pdf: "export.format.pdf",
  csv: "checkRecord.format.csv",
};

/** Until settings load, as the core's defaults have it. */
const DEFAULT_PREFERENCES: ExportPreferences = {
  format: "docx",
  docx: {
    includeQuestions: false,
    trackChanges: true,
    markAiText: true,
    commentFailedChecks: true,
  },
  pdf: { includeQuestions: false },
  html: { includeQuestions: false },
  "share-page": { includeQuestions: false, citationCards: true, checkedLine: false },
};

export const exportPreferences = (): ExportPreferences =>
  useAppStore.getState().settings?.user.exportOptions ?? DEFAULT_PREFERENCES;

/** Whether a format has the Include Questions option: Markdown keeps them, and a source pack has none. */
export const hasQuestionsOption = (
  format: ExportFormat,
): format is Exclude<ExportFormat, "markdown" | "source-pack"> =>
  format !== "markdown" && format !== "source-pack";

/** What an export of `format` asks the core for. */
export const optionsFor = (
  format: ExportFormat,
  preferences: ExportPreferences,
): ExportMindOptions => {
  if (format === "docx") return { format, ...preferences.docx };
  if (format === "source-pack") return { format };
  if (format === "share-page") {
    return {
      format,
      ...preferences["share-page"],
      theirChanges: preferences.theirChanges ?? "shown",
    };
  }
  const theirChanges = preferences.theirChanges ?? "shown";
  return hasQuestionsOption(format)
    ? { format, includeQuestions: preferences[format].includeQuestions, theirChanges }
    : { format, theirChanges };
};

export type ExportKind = ExportFormat | `check-record-${CheckRecordFormat}`;

export const isCheckRecord = (kind: ExportKind): kind is `check-record-${CheckRecordFormat}` =>
  kind.startsWith("check-record-");

export const checkRecordKind = (format: CheckRecordFormat): `check-record-${CheckRecordFormat}` =>
  `check-record-${format}`;

/** How the last export went, said in the Mind it exported (`ExportResultLine`). */
export type ExportResult =
  | { mindId: string; state: "saving"; format: ExportKind }
  | {
      mindId: string;
      state: "saved";
      format: ExportKind;
      path: string;
      name: string;
      /** The folder it is in, which the line names. */
      folder: string;
      inDefaultFolder: boolean;
      /** When it was saved, in ms, so a line the User left behind doesn't come back long after. */
      at: number;
      unverified: number;
      /** The sentences whose check isn't good: how many, and whether they went in as comments. */
      checks: SentenceChecksExported;
      /** The changes waiting that went into a Word file as tracked changes. */
      tracked: number;
      /** The header and footer kept in a Word file though deleted from the Mind (#105). */
      kept: ("header" | "footer")[];
      /** The first save, which took Documents › IncarnaMind: the line says where (D209). */
      defaulted: boolean;
    }
  /** Not saved yet: what a Word file can't keep, for the User to see first (#105). */
  | { mindId: string; state: "review"; format: ExportFormat; word: WordExportReport }
  | { mindId: string; state: "failed"; format: ExportKind; message: string }
  /** A source pack of a Mind with no Citations: nothing to make, and nothing went wrong. */
  | { mindId: string; state: "empty"; format: ExportFormat }
  | {
      mindId: string;
      state: "not-allowed";
      format: ExportKind;
      linkedFolderId: string;
      folder: string;
    };

export const useExports = create<{
  result: ExportResult | null;
  dismiss(): void;
}>()((set) => ({
  result: null,
  dismiss: () => set({ result: null }),
}));

// Null while the same Mind is still exporting, so a held ⇧⌘E makes one file; a newer export or a
// dismissal wins over a late answer.
function begin(mindId: string, kind: ExportKind): ((next: ExportResult | null) => void) | null {
  const { result } = useExports.getState();
  if (result?.mindId === mindId && result.state === "saving") return null;
  useExports.setState({ result: { mindId, state: "saving", format: kind } });
  return (next) => {
    const current = useExports.getState().result;
    if (current?.mindId === mindId && current.state === "saving") {
      useExports.setState({ result: next });
    }
  };
}

const NO_CHECKS: SentenceChecksExported = {
  contradicted: 0,
  unsupported: 0,
  needsRecheck: 0,
  commented: false,
};

function savedResult(
  mindId: string,
  format: ExportKind,
  saved: Exclude<SavedMindExport, { status: "review" }> | SavedExport,
): ExportResult | null {
  if (saved.status === "saved") {
    return {
      mindId,
      state: "saved",
      format,
      path: saved.file.path,
      name: saved.file.name,
      folder: saved.file.folder,
      inDefaultFolder: saved.file.inDefaultFolder,
      at: Date.now(),
      unverified: "unverifiedCitations" in saved ? saved.unverifiedCitations : 0,
      checks: "sentenceChecks" in saved ? saved.sentenceChecks : NO_CHECKS,
      tracked: "trackedChanges" in saved ? saved.trackedChanges : 0,
      kept: "word" in saved ? (saved.word?.keptFurniture ?? []) : [],
      defaulted: saved.file.defaulted,
    };
  }
  if (saved.status === "not-allowed") {
    return {
      mindId,
      state: "not-allowed",
      format,
      linkedFolderId: saved.linkedFolderId,
      folder: folderName(saved.linkedFolderPath),
    };
  }
  // The User didn't choose where new files go: nothing was saved, and nothing went wrong.
  return null;
}

/**
 * Exports a Mind into the Output location (the first save asks where that is), in `format` or
 * the last one, which it becomes; the result line says how it went. `first` fixes what stopped
 * the last try (another Output location, allowing a Linked folder), unless it returns false.
 * `accept` is what a Word file can't keep that the User has seen and exports anyway (#105).
 */
export async function exportMind(
  mindId: string,
  format?: ExportFormat,
  first?: () => Promise<boolean>,
  accept?: readonly string[],
): Promise<void> {
  const preferences = exportPreferences();
  const chosen = format ?? preferences.format;
  const settle = begin(mindId, chosen);
  if (!settle) return;
  if (chosen !== preferences.format) {
    void useAppStore
      .getState()
      .updateSettings({ user: { exportOptions: { ...preferences, format: chosen } } });
  }
  try {
    if (chosen === "source-pack") {
      const { citations } = await core.previewMindExport(mindId, { format: chosen });
      if (citations === 0) {
        settle({ mindId, state: "empty", format: chosen });
        return;
      }
    }
    if (first && !(await first())) {
      settle(null);
      return;
    }
    const saved = await core.saveMindExport(mindId, {
      ...optionsFor(chosen, preferences),
      ...(accept && { accept: [...accept] }),
    });
    settle(
      saved.status === "review"
        ? { mindId, state: "review", format: chosen, word: saved.word }
        : savedResult(mindId, chosen, saved),
    );
  } catch (failure) {
    settle({
      mindId,
      state: "failed",
      format: chosen,
      message: errorMessage(failure, translateNow),
    });
  }
}

export async function exportCheckRecord(
  mindId: string,
  format: CheckRecordFormat,
  first?: () => Promise<boolean>,
): Promise<void> {
  const kind = checkRecordKind(format);
  const settle = begin(mindId, kind);
  if (!settle) return;
  try {
    if (first && !(await first())) {
      settle(null);
      return;
    }
    settle(savedResult(mindId, kind, await core.saveCheckRecord(mindId, format)));
  } catch (failure) {
    settle({ mindId, state: "failed", format: kind, message: errorMessage(failure, translateNow) });
  }
}

export function exportAgain(
  mindId: string,
  kind: ExportKind,
  first: () => Promise<boolean>,
): Promise<void> {
  if (!isCheckRecord(kind)) return exportMind(mindId, kind, first);
  return exportCheckRecord(mindId, kind === "check-record-pdf" ? "pdf" : "csv", first);
}

/**
 * Shows the first Citation of the Mind whose quote wasn't found, as the result line's Show does:
 * scrolled to the middle and focused, so its card opens and the focus ring marks it.
 */
export function revealUnverified(mindId: string): boolean {
  const citation = document.querySelector<HTMLElement>(
    `article[data-mind-id="${CSS.escape(mindId)}"] [data-testid="citation"]:not([data-check="found"])`,
  );
  if (!citation) return false;
  citation.scrollIntoView({ block: "center" });
  const chip = citation.querySelector<HTMLElement>('[data-testid="citation-chip"]') ?? citation;
  chip.focus();
  return true;
}
