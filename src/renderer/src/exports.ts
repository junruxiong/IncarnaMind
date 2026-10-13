import { create } from "zustand";
import type {
  ExportFormat,
  ExportMindOptions,
  ExportPreferences,
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

/** Until settings load, as the core's defaults have it. */
const DEFAULT_PREFERENCES: ExportPreferences = {
  format: "docx",
  docx: { includeQuestions: false },
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
): ExportMindOptions =>
  hasQuestionsOption(format)
    ? { format, includeQuestions: preferences[format].includeQuestions }
    : { format };

/** How the last export went, said in the Mind it exported (`ExportResultLine`). */
export type ExportResult =
  | { mindId: string; state: "saving"; format: ExportFormat }
  | {
      mindId: string;
      state: "saved";
      format: ExportFormat;
      path: string;
      name: string;
      unverified: number;
      /** The header and footer kept in a Word file though deleted from the Mind (#105). */
      kept: ("header" | "footer")[];
    }
  /** Not saved yet: what a Word file can't keep, for the User to see first (#105). */
  | { mindId: string; state: "review"; format: ExportFormat; word: WordExportReport }
  | { mindId: string; state: "failed"; format: ExportFormat; message: string }
  /** A source pack of a Mind with no Citations: nothing to make, and nothing went wrong. */
  | { mindId: string; state: "empty"; format: ExportFormat }
  | {
      mindId: string;
      state: "not-allowed";
      format: ExportFormat;
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
 * Exports a Mind into the Output location (the first save asks where that is), in `format` or
 * the last one, which it becomes; the result line says how it went. `first` fixes what stopped
 * the last try (another Output location, allowing a Linked folder), unless it returns false.
 * `accept` is what a Word file can't keep that the User has seen and exports anyway (#105).
 * Nothing starts while the same Mind is still exporting: a held ⇧⌘E makes one file.
 */
export async function exportMind(
  mindId: string,
  format?: ExportFormat,
  first?: () => Promise<boolean>,
  accept?: readonly string[],
): Promise<void> {
  const { result } = useExports.getState();
  if (result?.mindId === mindId && result.state === "saving") return;
  const preferences = exportPreferences();
  const chosen = format ?? preferences.format;
  useExports.setState({ result: { mindId, state: "saving", format: chosen } });
  if (chosen !== preferences.format) {
    void useAppStore
      .getState()
      .updateSettings({ user: { exportOptions: { ...preferences, format: chosen } } });
  }
  const settle = (next: ExportResult | null) => {
    // A newer export, or a dismissal, wins over this one's late answer.
    const current = useExports.getState().result;
    if (current?.mindId === mindId && current.state === "saving") {
      useExports.setState({ result: next });
    }
  };
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
    if (saved.status === "saved") {
      settle({
        mindId,
        state: "saved",
        format: chosen,
        path: saved.file.path,
        name: saved.file.name,
        unverified: saved.unverifiedCitations,
        kept: saved.word?.keptFurniture ?? [],
      });
    } else if (saved.status === "review") {
      settle({ mindId, state: "review", format: chosen, word: saved.word });
    } else if (saved.status === "not-allowed") {
      settle({
        mindId,
        state: "not-allowed",
        format: chosen,
        linkedFolderId: saved.linkedFolderId,
        folder: folderName(saved.linkedFolderPath),
      });
    } else {
      // The User didn't choose where new files go: nothing was saved, and nothing went wrong.
      settle(null);
    }
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
