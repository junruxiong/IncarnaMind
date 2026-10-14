import type { Editor } from "@tiptap/core";
import { type ReactNode, useEffect, useState } from "react";
import type * as Y from "yjs";
import {
  BLOCK_ID_ATTRIBUTE,
  type Document,
  type KeptChangeKind,
  type WordAtomKind,
  type WordImportReport,
} from "../../../core/api";
import {
  layoutNotShown,
  observeWordFile,
  wordFileOf,
  wordImportProgressOf,
  wordReportOf,
  wordSourceOf,
} from "../../../shared/wordFiles";
import { DocumentLineIcon } from "../components/lineIcons";
import { useObjectMenu } from "../components/ObjectMenu";
import { ProblemLine } from "../components/ProblemLine";
import { menuClass, menuItemClass, menuItemUnavailableClass } from "../components/ui";
import { usePopoverMenu } from "../components/usePopoverMenu";
import { files } from "../core";
import { showFileLabelKey } from "../fileInfo";
import { useLanguage, useT } from "../i18n";
import { fileStatusLabel } from "../linkedFolders";
import { useAppStore } from "../store";
import { openDocumentBeside } from "../verify";
import { useWordImport } from "../wordImport";
import { ChevronDownSmallIcon } from "./icons";
import { wordAtomName } from "./WordAtomView";

/** What the Mind keeps about the Word file it was imported from, kept up to date as the import goes on. */
export function useWordFile(doc: Y.Doc) {
  const read = () => ({
    file: wordFileOf(doc),
    progress: wordImportProgressOf(doc),
    report: wordReportOf(doc),
  });
  const [state, setState] = useState(read);
  // biome-ignore lint/correctness/useExhaustiveDependencies: `read` only reads `doc`.
  useEffect(() => {
    setState(read());
    return observeWordFile(doc, () => setState(read()));
  }, [doc]);
  return state;
}

/** What the import made editable and what it kept as is, someone's changes too (#108), by count, for the line's tooltip. */
function reportLines(
  report: WordImportReport,
  t: ReturnType<typeof useT>,
  language: string,
): string[] {
  const list = (items: string[]) =>
    new Intl.ListFormat(language, { type: "conjunction" }).format(items);
  const editable = [
    report.paragraphs > 0 && t("wordImport.count.paragraphs", { count: report.paragraphs }),
    report.tables > 0 && t("wordImport.count.tables", { count: report.tables }),
    report.images > 0 && t("wordImport.count.images", { count: report.images }),
    report.links > 0 && t("wordImport.count.links", { count: report.links }),
  ].filter((item): item is string => typeof item === "string");
  const kept = Object.entries(report.kept)
    .filter((entry): entry is [WordAtomKind, number] => (entry[1] ?? 0) > 0)
    .map(([kind, count]) =>
      t("wordImport.count.kept", { kind: wordAtomName({ kind, info: null }), count }),
    );
  const keptChanges = Object.entries(report.keptChanges ?? {})
    .filter((entry): entry is [KeptChangeKind, number] => (entry[1] ?? 0) > 0)
    .map(([kind, count]) => t(`wordImport.keptChange.${kind}`, { count }));
  return [
    editable.length > 0 ? t("wordImport.report.editable", { list: list(editable) }) : "",
    kept.length > 0 ? t("wordImport.report.kept", { list: list(kept) }) : "",
    keptChanges.length > 0 ? t("wordImport.report.keptChanges", { list: list(keptChanges) }) : "",
  ].filter(Boolean);
}

/**
 * The line under the title of a Mind imported from a Word file (#104; the
 * board Final-30-Word-Import): "From Lease.docx · Export keeps its styles",
 * what the import did on hover. "From Lease.docx" opens the original beside
 * the Mind while it is a Document (#720), as a verified PDF's line does, at
 * the paragraph the caret is in (#736); its menu also offers Show in Finder.
 * While the import goes on, how far it has got, with a 2px bar, as a Linked
 * folder's indexing shows it.
 */
export function WordOrigin({ doc, editor }: { doc: Y.Doc; editor: Editor | null }) {
  const t = useT();
  const language = useLanguage();
  const { file, progress, report } = useWordFile(doc);
  const original = useAppStore((state) =>
    file?.documentId ? state.documents.find((each) => each.id === file.documentId) : undefined,
  );
  if (!file) return null;
  const icon = <DocumentLineIcon kind="docx" className="size-3.5 shrink-0" />;
  if (progress) {
    const percent = Math.floor((progress.done / Math.max(1, progress.total)) * 100);
    return (
      <p className="mind-word" data-testid="mind-word-progress" role="status">
        {icon}
        <span className="min-w-0 truncate">
          {t("wordImport.progress", { name: file.name, percent })}
        </span>
        <span
          className="mind-count-progress"
          role="progressbar"
          aria-label={t("wordImport.progress.label")}
          aria-valuenow={percent}
          aria-valuemin={0}
          aria-valuemax={100}
        >
          <span className="mind-count-fill" style={{ width: `${percent}%` }} />
        </span>
      </p>
    );
  }
  const tooltip = [
    t("wordImport.from.tooltip"),
    layoutNotShown(report?.layout) ? t("wordImport.layoutNotShown") : "",
    ...(report ? reportLines(report, t, language) : []),
  ]
    .filter(Boolean)
    .join("\n");
  const from = t("wordImport.from", { name: file.name });
  return (
    <div className="mind-word" data-testid="mind-word-origin" title={tooltip}>
      {original ? (
        <OriginalLink
          original={original}
          name={file.name}
          open={() => {
            const wordPlace = caretWordPlace(editor, doc);
            openDocumentBeside(original.id, wordPlace ? { wordPlace } : {});
          }}
        >
          {icon}
          <span className="min-w-0 truncate">{from}</span>
        </OriginalLink>
      ) : (
        <>
          {icon}
          <span className="min-w-0 truncate">{from}</span>
        </>
      )}
      <span className="shrink-0">{` · ${t("wordImport.keepsStyles")}`}</span>
    </div>
  );
}

/** The paragraph's or table's place in the Word file, of the Block the caret is in, as the import recorded it. */
function caretWordPlace(editor: Editor | null, doc: Y.Doc): string | null {
  if (!editor) return null;
  const { $from } = editor.state.selection;
  for (let depth = $from.depth; depth > 0; depth--) {
    const id: unknown = $from.node(depth).attrs[BLOCK_ID_ATTRIBUTE];
    const place = typeof id === "string" ? wordSourceOf(doc, id)?.n : undefined;
    if (place && /^w:(p|tbl)#\d+$/.test(place)) return place;
  }
  return null;
}

/**
 * "From Lease.docx", opening the original, and its menu, by the chevron beside it or a
 * right-click: Open the original, and Show in Finder.
 */
function OriginalLink({
  original,
  name,
  open,
  children,
}: {
  original: Document;
  name: string;
  open(): void;
  children: ReactNode;
}) {
  const t = useT();
  const menu = usePopoverMenu();
  // A right-click, the context-menu key or Shift+F10 opens the chevron's menu, as on any row.
  const row = useObjectMenu({ kind: "document", id: original.id });
  const choose = (action: () => void) => {
    menu.close();
    action();
  };
  const away = fileStatusLabel(original.fileStatus);
  return (
    <span className="mind-word-original" {...row.rowProps}>
      <button
        type="button"
        data-testid="mind-word-original"
        className="mind-word-from"
        title={t("wordImport.from.open", { name })}
        onClick={open}
      >
        {children}
      </button>
      <button
        type="button"
        {...menu.buttonProps}
        {...row.buttonProps}
        data-testid="mind-word-original-menu"
        aria-label={t("wordImport.original.more", { name })}
        className="mind-word-more"
      >
        <ChevronDownSmallIcon />
      </button>
      <div
        {...menu.menuProps}
        role="menu"
        aria-label={t("wordImport.original.more", { name })}
        data-testid="mind-word-original-items"
        className={menuClass}
      >
        {menu.open && (
          <>
            <button
              type="button"
              role="menuitem"
              data-testid="mind-word-open-original"
              onClick={() => choose(open)}
              className={`${menuItemClass} pl-2`}
            >
              {t("wordImport.openOriginal")}
            </button>
            <button
              type="button"
              role="menuitem"
              data-testid="mind-word-show-original"
              aria-disabled={away ? true : undefined}
              title={away ? t(away.reason) : undefined}
              onClick={() => {
                if (away) return;
                choose(() => {
                  files
                    .showDocumentInFolder(original.id)
                    .catch((error) => useAppStore.getState().reportError(error));
                });
              }}
              className={`${menuItemClass} ${menuItemUnavailableClass} pl-2`}
            >
              {t(showFileLabelKey(navigator.platform))}
            </button>
          </>
        )}
      </div>
    </span>
  );
}

/** The original the Library couldn't take at import (#720), said under the Mind's line, with a way to try again. */
export function OriginalNotAdded({ mindId }: { mindId: string }) {
  const t = useT();
  const path = useWordImport((state) => state.notAdded[mindId]);
  const retrying = useWordImport((state) => state.retrying === mindId);
  if (path === undefined) return null;
  return (
    <ProblemLine
      testId="mind-word-not-added"
      className="mind-word-problem"
      action={{
        label: t("wordImport.addToLibrary"),
        testId: "mind-word-add-original",
        disabled: retrying,
        onClick: () => void useWordImport.getState().retryOriginal(mindId),
      }}
    >
      {t("wordImport.notAdded", { name: path.split(/[\\/]/).pop() ?? path })}
    </ProblemLine>
  );
}
