import { type KeyboardEvent, memo, type Ref, useId, useRef, useState } from "react";
import type { Document, ReferenceField, ReferenceRecord } from "../../../../core/api";
import { displayName, sourceOf } from "../../../../shared/documentNames";
import { missingDetails } from "../../../../shared/references";
import { commands } from "../../commands/registry";
import { useLanguage, useT } from "../../i18n";
import { FORMAT_LABELS } from "../../libraryFilters";
import { addedLabel, foundLine, type SearchMatch } from "../../libraryPage";
import { atPassage, passagePlace } from "../../palette";
import { missingList } from "../../referenceEditing";
import { useSelectable } from "../../selection";
import { useAppStore } from "../../store";
import { isDeleteKey, isMoveToKey, isTagsKey, TREE_ROW } from "../../treeKeys";
import { typesText } from "../../undo";
import { useRename } from "../../useRename";
import { CiteAsPanel } from "../CiteAsPanel";
import { DocumentStatusLabel } from "../DocumentsSection";
import { InlineName } from "../InlineName";
import { LibrarySelectBox } from "../LibrarySelection";
import { DocumentLineIcon, FolderLineIcon } from "../lineIcons";
import { Marked } from "../Marked";
import { ObjectMenu, useObjectMenu } from "../ObjectMenu";
import { rowActionButtonClass } from "../sidebarRows";
import { ReviewRing } from "../TagColour";
import { useTabPlace } from "../tabs/tabItems";
import { documentsToMove, useDocumentMove } from "./DocumentMoveTo";
import { RowTags } from "./RowTags";

/**
 * The columns past the name, and the widths of the page at which they give
 * way, Added first, then Format, then Tags, so the name keeps its room.
 */
export const COLUMN_CLASSES = {
  // Narrow, only the chips go: the "+" stays, so Tags… from the menu still has its place.
  tags: "w-[200px] shrink-0 @max-[512px]/page:w-auto @max-[512px]/page:[&_[data-testid^=tag-chip]]:hidden",
  format: "w-[76px] shrink-0 @max-[704px]/page:hidden",
  added: "w-24 shrink-0 @max-[768px]/page:hidden",
  more: "flex w-7 shrink-0 justify-end",
} as const;

/**
 * A row of the page: 40px folded, washed when pointed at, its focus ring
 * around the whole row, as while its Tags or its menu are open over it.
 */
export const pageRowClass = (selected: boolean) =>
  `group/row relative border-b border-rule ${
    selected ? "bg-accent-wash" : "hover:bg-frame"
  } has-[[data-row-main]:focus-visible]:shadow-[inset_0_0_0_2px_var(--color-accent)] has-[:popover-open]:shadow-[inset_0_0_0_2px_var(--color-accent)]`;

/** Keeps the preview showing a Document, if one does. */
function keepPreviewOf(documentId: string): void {
  const { layout, changeLayout } = useAppStore.getState();
  const preview = layout?.groups
    .flatMap((group) => group.tabs)
    .find(
      (tab) => tab.preview && tab.item.kind === "document" && tab.item.documentId === documentId,
    );
  if (preview) void changeLayout({ type: "keep", tabId: preview.id });
}

/**
 * Opens the Tags of a row: its own cell's editor, or, when it is one of several
 * selected, Tags… for all of them in the selection's bar.
 */
function openTags(row: HTMLElement | null, documentId: string): void {
  const several = documentsToMove(documentId).length > 1;
  const opener = several
    ? row?.closest('[data-surface="library"]')?.querySelector<HTMLElement>("[data-selection-tags]")
    : row?.querySelector<HTMLElement>('[data-testid="document-tags-menu"]');
  opener?.click();
}

/**
 * Deletes a row's Document, or every one selected with it, to Recently
 * deleted, and moves the focus to the row after them (before them at the end),
 * as Finder does.
 */
function deleteFrom(row: HTMLElement | null, documentId: string): void {
  const ids = documentsToMove(documentId);
  const gone = new Set(ids);
  const grid = row?.closest('[role="grid"]');
  const rows = [...(grid?.querySelectorAll<HTMLElement>("[data-document-id]") ?? [])];
  const at = rows.findIndex((each) => each.dataset.documentId === documentId);
  const stays = (each: HTMLElement) => !gone.has(each.dataset.documentId ?? "");
  const next = rows.slice(at + 1).find(stays) ?? rows.slice(0, at).reverse().find(stays);
  const nextId = next?.dataset.documentId;
  commands
    .run("document.delete", {
      selection: ids.map((id) => ({ kind: "document", id })),
      focus: { kind: "document", id: documentId },
      surface: "library",
    })
    .then(() => {
      if (!nextId) return;
      grid
        ?.querySelector<HTMLElement>(`[data-document-id="${CSS.escape(nextId)}"] [data-row-main]`)
        ?.focus();
    })
    .catch(useAppStore.getState().reportError);
}

/** Clicks on these, inside a row, are theirs, not the row's. */
const OWN_CLICKS = "button:not([data-row-main]), input, a, [role=menu], [role=dialog]";

/**
 * A Document's row on the Library page: its name (with the kind's icon, a
 * problem's word, and "Suggested" while Organize's Folder waits), its Tags, its
 * format and when it was added, then ⋯; the checkbox hangs in the margin. A
 * click anywhere but its Tags, checkbox and ⋯ opens it beside the list; ⌘-click
 * and Shift-click select. Memoised (#156): while Organize goes through
 * thousands of Documents, only the rows whose Document changed are drawn again.
 */
export const LibraryRow = memo(function LibraryRow({
  doc,
  suggestedIn,
  notSorted,
  match,
  path,
}: {
  doc: Document;
  /** The Suggested Folder it waits in, if Organize suggested one. */
  suggestedIn: string | null;
  /** Why Organize couldn't sort it, if it couldn't. */
  notSorted: string | null;
  /** How it matches the search, while there is one. */
  match: SearchMatch | null;
  /** Its Folder, shown under it while searching. */
  path: string | null;
}) {
  const t = useT();
  const language = useLanguage();
  const place = useTabPlace();
  const name = displayName(doc);
  const source = sourceOf(doc);
  const [citing, setCiting] = useState<{ focus: ReferenceField | null } | null>(null);
  const citeToggle = useRef<HTMLButtonElement>(null);
  const citePanelId = useId();
  const objectMenu = useObjectMenu({ kind: "document", id: doc.id });
  const inside = match?.text ?? null;
  const foundAt = inside ? passagePlace(doc.kind, inside, t) : "";
  const line = inside ? foundLine(inside) : null;
  const select = useSelectable({ kind: "document", id: doc.id }, "library", () =>
    useAppStore
      .getState()
      .openDocument(
        { documentId: doc.id, ...(inside && atPassage(inside)) },
        { as: "library", ...(place && { from: place.groupId }) },
      ),
  );
  // A double-click keeps the preview it opened, then renames.
  const rename = useRename<HTMLButtonElement>(`library-document:${doc.id}`, () =>
    keepPreviewOf(doc.id),
  );
  const renameDocument = useAppStore((state) => state.renameDocument);
  const row = useRef<HTMLDivElement>(null);
  const moveTo = (anchor: HTMLElement | null) => {
    if (anchor) useDocumentMove.getState().show({ anchor, documentIds: documentsToMove(doc.id) });
  };
  const menuButton = () => row.current?.querySelector<HTMLElement>("[data-object-menu]") ?? null;

  const onKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    objectMenu.rowProps.onKeyDown(event);
    if (event.defaultPrevented || typesText(event.target)) return;
    if (isMoveToKey(event)) {
      event.preventDefault();
      moveTo(menuButton());
    } else if (isTagsKey(event)) {
      event.preventDefault();
      openTags(row.current, doc.id);
    } else if (isDeleteKey(event)) {
      event.preventDefault();
      deleteFrom(row.current, doc.id);
    }
  };

  return (
    // biome-ignore lint/a11y/useSemanticElements: a row of columns in a grid of sections, not a table row.
    // biome-ignore lint/a11y/useFocusableInteractive: the name's button takes the focus, and the arrow keys move between them.
    <div
      ref={row}
      data-testid="library-document"
      data-document-id={doc.id}
      data-suggested={suggestedIn !== null ? "" : undefined}
      role="row"
      {...select.rowProps}
      {...objectMenu.rowProps}
      onKeyDown={onKeyDown}
      onClick={(event) => {
        const target = event.target instanceof Element ? event.target : null;
        if (target?.closest(OWN_CLICKS) || target?.closest("[data-row-cell=tags]")) return;
        select.onClick(event);
      }}
      className={`${pageRowClass(select.selected)} cursor-default`}
    >
      <LibrarySelectBox document={doc} />
      <div className="flex min-h-[39px] items-center gap-4 pr-1.5 pl-3">
        <div className="flex min-w-0 flex-1 items-center">
          <span aria-hidden="true" className="w-5 shrink-0" />
          <div className={`min-w-0 flex-1 ${match ? "py-1.5" : ""}`}>
            <div className="flex min-w-0 items-center gap-2">
              {rename.renaming ? (
                <span className="flex min-w-0 flex-1 items-center gap-2">
                  <DocumentLineIcon kind={doc.kind} className="size-4 shrink-0 text-ink-meta" />
                  <InlineName
                    value={doc.name}
                    label={t("documents.renameLabel", { name: doc.name })}
                    {...rename.field((next) => renameDocument(doc.id, next))}
                  />
                </span>
              ) : (
                <button
                  {...rename.nameProps}
                  type="button"
                  {...TREE_ROW}
                  data-row-main=""
                  title={name === source ? name : `${name}\n${source}`}
                  className="flex min-w-0 items-center gap-2 text-left focus-visible:outline-0"
                >
                  <DocumentLineIcon kind={doc.kind} className="size-4 shrink-0 text-ink-meta" />
                  <span
                    data-testid="library-document-name"
                    className="min-w-0 truncate text-ui text-ink"
                  >
                    {match ? <Marked text={name} ranges={match.ranges} /> : name}
                  </span>
                </button>
              )}
              {foundAt && (
                <span
                  data-testid="library-found-place"
                  className="shrink-0 text-[12px] text-ink-meta"
                >
                  {foundAt}
                </span>
              )}
              <span className="contents">
                <DocumentStatusLabel item={doc} />
                {notSorted !== null && (
                  <span
                    data-testid="library-not-sorted"
                    title={t("library.notSorted.title", { reason: notSorted })}
                    className="shrink-0 text-label font-semibold text-ink-meta"
                  >
                    {t("library.notSorted")}
                  </span>
                )}
                {suggestedIn !== null && (
                  <span
                    data-testid="library-suggested"
                    title={t("library.suggested.title", { folder: suggestedIn })}
                    className="inline-flex shrink-0 items-center gap-1 text-label font-semibold text-review"
                  >
                    <ReviewRing />
                    {t("library.suggested")}
                  </span>
                )}
              </span>
              {doc.reference && !rename.renaming && (
                <CiteAsToggle
                  ref={citeToggle}
                  document={doc}
                  record={doc.reference}
                  panelId={citePanelId}
                  open={citing !== null}
                  onToggle={(focus) => setCiting(citing === null ? { focus } : null)}
                />
              )}
            </div>
            {line && (
              <p
                data-testid="library-found-text"
                className="truncate pl-6 text-[13px] leading-5 text-ink-secondary"
              >
                <Marked text={line.text} ranges={line.ranges} />
              </p>
            )}
            {path !== null && (
              <p
                data-testid="library-path"
                className="flex min-w-0 items-center gap-1 pl-6 text-[12px] leading-4 text-ink-meta"
              >
                <FolderLineIcon className="size-3 shrink-0" />
                <span className="truncate">{path}</span>
              </p>
            )}
          </div>
        </div>
        <div data-row-cell="tags" className={`${COLUMN_CLASSES.tags} flex min-w-0 items-center`}>
          <RowTags document={doc} />
        </div>
        <span
          data-testid="library-document-format"
          className={`${COLUMN_CLASSES.format} truncate text-[13px] text-ink-secondary`}
        >
          {t(FORMAT_LABELS[doc.kind])}
        </span>
        <span
          data-testid="library-document-added"
          className={`${COLUMN_CLASSES.added} truncate text-[13px] whitespace-nowrap text-ink-secondary`}
        >
          {addedLabel(doc.createdAt, new Date(), language)}
        </span>
        <span className={COLUMN_CLASSES.more}>
          <ObjectMenu
            target={{ kind: "document", id: doc.id }}
            label={t("documents.more", { name: doc.name })}
            here={{
              "document.rename": rename.start,
              // Once the menu has closed and given the focus back, so the picker opens beside ⋯.
              "document.moveTo": () => requestAnimationFrame(() => moveTo(menuButton())),
            }}
            className={`${rowActionButtonClass} opacity-0 group-hover/row:opacity-100 group-has-[:focus-visible]/row:opacity-100 aria-expanded:opacity-100`}
          />
        </span>
      </div>
      {citing && doc.reference && (
        <div id={citePanelId} className="min-w-0 pr-1.5 pb-3 pl-14">
          <CiteAsPanel
            document={doc}
            record={doc.reference}
            focusField={citing.focus}
            onClose={() => {
              setCiting(null);
              citeToggle.current?.focus();
            }}
          />
        </div>
      )}
    </div>
  );
});

/**
 * A row's way into "Cite as" (#286), after its name: "Needs details", with the
 * waiting dot, while the record is too thin to cite, which opens it at the
 * first missing field; otherwise "Cite as", shown while the row is pointed at
 * or focused.
 */
function CiteAsToggle({
  ref,
  document,
  record,
  panelId,
  open,
  onToggle,
}: {
  ref: Ref<HTMLButtonElement>;
  document: Document;
  record: ReferenceRecord;
  panelId: string;
  open: boolean;
  onToggle(focus: ReferenceField | null): void;
}) {
  const t = useT();
  const language = useLanguage();
  const missing = missingDetails(record);
  const list = missingList(language, record);
  const label = t("reference.citeAs.label", { name: displayName(document) });
  return (
    <button
      ref={ref}
      type="button"
      data-testid="cite-as-toggle"
      data-needs-details={missing.length > 0 ? "" : undefined}
      aria-expanded={open}
      aria-controls={open ? panelId : undefined}
      aria-label={
        list ? `${label}. ${t("reference.needsDetails.title", { missing: list })}` : label
      }
      title={list ? t("reference.needsDetails.title", { missing: list }) : undefined}
      onClick={() => onToggle(missing[0] ?? null)}
      className={`inline-flex shrink-0 items-center gap-1.5 rounded-sm text-[12px] leading-5 whitespace-nowrap text-ink-meta hover:text-ink hover:underline ${
        missing.length > 0 || open
          ? ""
          : "opacity-0 group-focus-within/row:opacity-100 group-hover/row:opacity-100"
      }`}
    >
      {missing.length > 0 && (
        <span aria-hidden="true" className="size-1.5 shrink-0 rounded-full bg-attention" />
      )}
      {t(missing.length > 0 ? "reference.needsDetails" : "reference.citeAs")}
    </button>
  );
}
