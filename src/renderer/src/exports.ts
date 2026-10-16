import { create } from "zustand";
import type {
  CheckRecordFormat,
  ExportFormat,
  ExportMindOptions,
  ExportPreferences,
  SavedExport,
  SavedMindExport,
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
  markdown: "export.format.markdown",
};

/** Each check record format's name, as the menus and ⌘K say it. */
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
  const theirChanges = preferences.theirChanges ?? "shown";
  return hasQuestionsOption(format)
    ? { format, includeQuestions: preferences[format].includeQuestions, theirChanges }
    : { format, theirChanges };
};

/** What an export makes: a Mind in a format, or the record of what was checked in it (#738). */
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
      unverified: number;
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

/**
 * Starts an export of `kind`: the result line says "saving". Returns how to settle it, which
 * only a result still saving takes, as a newer export or a dismissal wins over a late answer;
 * null while the same Mind is still exporting, so a held ⇧⌘E makes one file.
 */
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

/** What a save says in the result line; null when the User didn't choose where new files go: nothing went wrong. */
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
      unverified: "unverifiedCitations" in saved ? saved.unverifiedCitations : 0,
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

/**
 * Saves the record of what was checked in a Mind (#738) into the Output location, as
 * `exportMind` does, in `format`; the result line says how it went.
 */
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

/** Exports as the result line's last try did: what its fixes (another folder, allowing a Linked folder) go on to do. */
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
