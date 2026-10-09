import { type ChangeEvent, useEffect, useRef, useState } from "react";
import { useShallow } from "zustand/react/shallow";
import type { Document, DocumentFailureReason, DocumentStatus } from "../../../core/api";
import type { MessageKey } from "../../../shared/i18n";
import { useT } from "../i18n";
import { fileStatusLabel } from "../linkedFolders";
import { selectVisibleDocuments, useAppStore } from "../store";
import { DocumentFileMenu } from "./DocumentFileMenu";
import { ActiveTagFilter, DocumentTagMenu, TagFilterMenu } from "./DocumentTags";
import { FolderTree } from "./FolderTree";
import { LibraryFolders } from "./LibraryFolders";
import { LinkFolderDialog } from "./LinkFolderDialog";
import { DocumentLineIcon, FolderPlusLineIcon, PlusLineIcon } from "./lineIcons";
import {
  openRowMenu,
  rowActionButtonClass,
  rowActionsClass,
  rowButtonClass,
  rowClass,
  rowIconClass,
  rowInputClass,
  rowPadding,
} from "./sidebarRows";
import {
  buttonClass,
  dangerButtonClass,
  dialogActionsClass,
  dialogBodyClass,
  dialogClass,
  dialogTextClass,
  dialogTitleClass,
} from "./ui";
import { useModal } from "./useModal";

/** What the file picker offers. The core decides what it takes. */
const ACCEPTED_FILES = ".pdf,.docx,.pptx,.xlsx,.csv,.txt,.md,.markdown";

/** A Document's status in full: for its tooltip and for screen readers. */
const statusMessages: Record<Exclude<DocumentStatus, "failed">, MessageKey> = {
  queued: "documents.status.queued",
  extracting: "documents.status.extracting",
  "waiting-for-model": "embedding.status.waiting",
  embedding: "embedding.status.embedding",
  ready: "documents.status.ready",
  "no-text": "documents.status.noText",
};

/** A Document's status as its row's end shows it: nothing once it's ready. */
const shortStatusMessages: Record<DocumentStatus, MessageKey | null> = {
  queued: "documents.statusShort.queued",
  extracting: "documents.statusShort.extracting",
  "waiting-for-model": "documents.statusShort.waiting",
  embedding: "documents.statusShort.embedding",
  ready: null,
  failed: "documents.statusShort.failed",
  "no-text": "documents.statusShort.noText",
};

const failureMessages: Record<DocumentFailureReason, MessageKey> = {
  unreadable: "documents.failure.unreadable",
  "password-protected": "documents.failure.passwordProtected",
  "file-missing": "documents.failure.fileMissing",
  "processing-error": "documents.failure.processingError",
};

/** Still being processed: the row is muted until it's done. */
const isProcessing = (status: DocumentStatus) =>
  status === "queued" ||
  status === "extracting" ||
  status === "waiting-for-model" ||
  status === "embedding";

/**
 * The sidebar's Documents: under the "Documents" label (with the Tag filter,
 * "Add folder…" to link a folder and adding files), each Linked folder with
 * its Folders as on disk, then "Other Documents", the files added on their
 * own (see `FolderTree`). Each Document is one row: its name and, while it's
 * processed, if it failed, or if its file is missing or can't be reached,
 * its status at the end. Its Tags are in its Tags menu, and the Tags dialog;
 * its "More" menu renames it, opens its file or shows it in its folder, and
 * deletes it. Dropping files anywhere on the window adds them too (see
 * `FileDrop`). Clicking a Document opens it in the viewer. With nothing
 * linked or added yet, the section says how to start: "Add folder…" (which
 * asks first, see `LinkFolderDialog`) or "Add Documents".
 */
export function DocumentsSection() {
  const t = useT();
  const hasLibraryFolders = useAppStore((state) => (state.library?.groups.length ?? 0) > 0);
  const [showSources, setShowSources] = useState(false);
  const libraryOpen = useAppStore(
    (state) => state.libraryOpen && ["all", "new"].includes(state.libraryFilter),
  );
  // Filtering makes a new array each time: compare it item by item, or React re-renders forever.
  const documents = useAppStore(useShallow(selectVisibleDocuments));
  const filtering = useAppStore((state) => state.tagFilter.length > 0);
  const hasFolders = useAppStore(
    (state) => state.folders.length > 0 || state.linkedFolders.length > 0,
  );
  const addDocuments = useAppStore((state) => state.addDocuments);
  const addLinkedFolder = useAppStore((state) => state.addLinkedFolder);
  const pickDocuments = useAppStore((state) => state.pickDocuments);
  const picking = useAppStore((state) => state.pickingDocuments);
  const picker = useRef<HTMLInputElement>(null);
  const [deleting, setDeleting] = useState<Document | null>(null);

  const addPicked = (event: ChangeEvent<HTMLInputElement>) => {
    const picked = Array.from(event.target.files ?? []);
    event.target.value = ""; // so picking the same file again still counts as a change
    if (picked.length > 0) void addDocuments(picked);
  };

  // A Linked folder shows even before its first Document, unless a filter hides it.
  const empty = documents.length === 0 && (filtering || !hasFolders);

  return (
    <section aria-labelledby="documents-heading">
      <div
        data-testid="documents-heading-row"
        className="mt-3 flex h-7 shrink-0 items-end justify-between rounded-md pr-1 pb-0.5 pl-2"
      >
        <h2 id="documents-heading" className="pb-0.5 text-label font-semibold text-ink-meta">
          {t("documents.title")}
        </h2>
        <div className="flex items-center">
          <TagFilterMenu />
          <button
            type="button"
            data-testid="add-linked-folder"
            aria-label={t("linkedFolders.add")}
            title={t("linkedFolders.add")}
            onClick={() => void addLinkedFolder()}
            className={rowActionButtonClass}
          >
            <FolderPlusLineIcon className="size-4" />
          </button>
          <button
            type="button"
            data-testid="add-documents"
            aria-label={t("documents.add")}
            title={t("documents.add")}
            disabled={picking}
            onClick={() => void pickDocuments()}
            className={rowActionButtonClass}
          >
            <PlusLineIcon className="size-4" />
          </button>
        </div>
        {/* Files given straight to the page arrive here, as from the dialog: the smoke tests add them so. */}
        <input
          ref={picker}
          type="file"
          multiple
          accept={ACCEPTED_FILES}
          data-testid="add-documents-input"
          className="hidden"
          onChange={addPicked}
        />
      </div>
      <div className={rowClass(libraryOpen)}>
        <button
          type="button"
          data-testid="open-library"
          onClick={() => useAppStore.getState().openLibrary()}
          className={rowButtonClass}
        >
          <DocumentLineIcon kind="text" className="size-4 shrink-0" />
          <span className="truncate">{t("library.all")}</span>
        </button>
      </div>
      <ActiveTagFilter />
      {hasLibraryFolders && (
        <fieldset
          className="mx-2 my-2 flex gap-3 border-b border-rule pb-1 text-[12px] text-ink-meta"
          aria-label={t("library.browse")}
        >
          <button
            type="button"
            aria-pressed={!showSources}
            className={!showSources ? "font-semibold text-ink" : "hover:text-ink"}
            onClick={() => setShowSources(false)}
          >
            {t("library.groups")}
          </button>
          <button
            type="button"
            aria-pressed={showSources}
            className={showSources ? "font-semibold text-ink" : "hover:text-ink"}
            onClick={() => setShowSources(true)}
          >
            {t("library.sources")}
          </button>
          <button
            type="button"
            aria-label={t("library.newGroup")}
            title={t("library.newGroup")}
            className={`${rowActionButtonClass} ml-auto`}
            onClick={() => useAppStore.getState().openLibrary("new")}
          >
            <PlusLineIcon className="size-4" />
          </button>
        </fieldset>
      )}
      {hasLibraryFolders && !showSources ? (
        <LibraryFolders
          documents={documents}
          renderDocument={(item, depth) => (
            <DocumentRow
              key={item.id}
              item={item}
              depth={depth}
              onDelete={() => setDeleting(item)}
            />
          )}
        />
      ) : (
        <FolderTree
          documents={documents}
          filtering={filtering}
          renderDocument={(item, depth) => (
            <DocumentRow
              key={item.id}
              item={item}
              depth={depth}
              onDelete={() => setDeleting(item)}
            />
          )}
        />
      )}
      {empty &&
        (filtering ? (
          <p className="px-2 py-1 text-[13px] leading-5 text-ink-meta">{t("tags.filter.empty")}</p>
        ) : (
          <div data-testid="documents-empty">
            <p className="px-2 pt-1 pb-1.5 text-[13px] leading-5 text-ink-meta">
              {t("documents.none")}
            </p>
            {/* Two actions, as rows like "New Mind": not a list of items. */}
            <div className={rowClass(false)}>
              <button
                type="button"
                data-testid="empty-add-linked-folder"
                onClick={() => void addLinkedFolder()}
                className={rowButtonClass}
              >
                <FolderPlusLineIcon className={rowIconClass(false)} />
                <span data-testid="row-text" className="truncate">
                  {t("linkedFolders.add")}
                </span>
              </button>
            </div>
            <div className={rowClass(false)}>
              <button
                type="button"
                data-testid="empty-add-documents"
                disabled={picking}
                onClick={() => void pickDocuments()}
                className={rowButtonClass}
              >
                <PlusLineIcon className={rowIconClass(false)} />
                <span data-testid="row-text" className="truncate">
                  {t("documents.add")}
                </span>
              </button>
            </div>
          </div>
        ))}
      <DeleteDocumentDialog target={deleting} onClose={() => setDeleting(null)} />
      <LinkFolderDialog />
    </section>
  );
}

/**
 * A Document's row: its icon and name, and its status at the end while it's
 * processed, if it failed, or if its file is missing or can't be reached
 * (then muted too; clicking still opens what IncarnaMind kept of it).
 * Pointed at, it offers its Tags and a menu for the rest. It is in the
 * Folder its file is in, so there is no moving it here.
 */
function DocumentRow({
  item,
  depth,
  onDelete,
}: {
  item: Document;
  depth: number;
  onDelete(): void;
}) {
  const [renaming, setRenaming] = useState(false);
  const openDocument = useAppStore((state) => state.openDocument);
  const isOpen = useAppStore(
    (state) => state.viewerOpen && state.viewerTarget?.documentId === item.id,
  );
  const muted = !isOpen && (isProcessing(item.status) || item.fileStatus !== "available");

  return (
    <li
      data-testid="document-list-item"
      data-document-id={item.id}
      data-folder-id={item.folderId ?? ""}
      data-depth={depth}
      data-status={item.status}
      data-file-status={item.fileStatus}
      data-tagging={item.tagging}
      onContextMenu={renaming ? undefined : (event) => openRowMenu(event, "document-file-menu")}
      className={rowClass(isOpen, muted ? "muted" : "item")}
    >
      {renaming ? (
        <div
          className="flex h-full w-full min-w-0 items-center gap-2 pr-1"
          style={rowPadding(depth)}
        >
          <DocumentLineIcon kind={item.kind} className={rowIconClass(false)} />
          <RenameInput item={item} onDone={() => setRenaming(false)} />
        </div>
      ) : (
        // Opens the Document in the viewer, replacing whatever it showed.
        <button
          type="button"
          data-testid="open-document"
          aria-current={isOpen ? "true" : undefined}
          title={item.name}
          onClick={() => openDocument({ documentId: item.id })}
          className={rowButtonClass}
          style={rowPadding(depth)}
        >
          <DocumentLineIcon kind={item.kind} className={rowIconClass(isOpen)} />
          <span data-testid="row-text" className="min-w-0 flex-1 truncate">
            {item.name}
          </span>
          <DocumentStatusLabel item={item} />
        </button>
      )}
      {!renaming && (
        <div className={rowActionsClass}>
          <DocumentTagMenu item={item} buttonClassName={rowActionButtonClass} />
          <DocumentFileMenu
            item={item}
            buttonClassName={rowActionButtonClass}
            onRename={() => setRenaming(true)}
            onDelete={onDelete}
          />
        </div>
      )}
    </li>
  );
}

/**
 * The status at a row's end: "Missing" or "Unavailable" when its file isn't
 * where it was (that comes first: it's what the User can act on), "38%"
 * while it's embedded, "Failed" if it failed, nothing once it's ready. Its
 * full words (and a failure's message) are its tooltip, and what a screen
 * reader reads.
 */
function DocumentStatusLabel({ item }: { item: Document }) {
  const t = useT();
  const fileState = fileStatusLabel(item.fileStatus);
  if (fileState) {
    return (
      <span
        data-testid="document-status"
        data-file-status={item.fileStatus}
        title={t(fileState.full)}
        className="min-w-0 shrink truncate text-label font-semibold text-ink-meta"
      >
        <span aria-hidden="true">{t(fileState.short)}</span>
        <span className="sr-only">{t(fileState.full)}</span>
      </span>
    );
  }
  const percent = Math.floor((item.progress ?? 0) * 100);
  const full =
    item.status === "failed"
      ? t("documents.status.failed", {
          reason: t(failureMessages[item.failure?.reason ?? "processing-error"]),
        })
      : t(statusMessages[item.status], { percent });
  const short = shortStatusMessages[item.status];
  const title = item.failure?.message ? `${full}\n${item.failure.message}` : full;
  if (!short) {
    return (
      <span data-testid="document-status" className="sr-only">
        {full}
      </span>
    );
  }
  return (
    <span
      data-testid="document-status"
      title={title}
      className={`shrink-0 text-label font-semibold ${
        item.status === "failed" ? "text-danger" : "text-ink-meta"
      }`}
    >
      <span aria-hidden="true">{t(short, { percent })}</span>
      <span className="sr-only">{full}</span>
    </span>
  );
}

/** Enter or leaving the field saves; Esc cancels. An empty name changes nothing. */
function RenameInput({ item, onDone }: { item: Document; onDone(): void }) {
  const t = useT();
  const renameDocument = useAppStore((state) => state.renameDocument);
  const [value, setValue] = useState(item.name);
  const field = useRef<HTMLInputElement>(null);
  const finished = useRef(false);

  useEffect(() => {
    field.current?.focus();
    field.current?.select();
  }, []);

  const finish = (save: boolean) => {
    if (finished.current) return;
    finished.current = true;
    const name = value.trim();
    if (save && name && name !== item.name) void renameDocument(item.id, name);
    onDone();
  };

  return (
    <input
      ref={field}
      value={value}
      aria-label={t("documents.renameLabel", { name: item.name })}
      onChange={(event) => setValue(event.target.value)}
      onKeyDown={(event) => {
        if (event.key === "Enter" || event.key === "Escape") {
          event.preventDefault();
          finish(event.key === "Enter");
        }
      }}
      onBlur={() => finish(true)}
      className={rowInputClass}
    />
  );
}

/** Asks before deleting a Document, or removing a missing one. */
function DeleteDocumentDialog({ target, onClose }: { target: Document | null; onClose(): void }) {
  const t = useT();
  const dialog = useModal(target !== null);
  const deleteDocument = useAppStore((state) => state.deleteDocument);

  // A missing Document's file is gone already: removing it drops what IncarnaMind kept of it.
  const missing = target?.fileStatus === "missing";

  const confirm = () => {
    if (target) void deleteDocument(target.id);
    onClose();
  };

  return (
    <dialog
      ref={dialog}
      onClose={onClose}
      data-testid="delete-document-dialog"
      aria-labelledby="delete-document-title"
      className={`${dialogClass} w-[26rem]`}
    >
      <div className={dialogBodyClass}>
        <h2 id="delete-document-title" className={dialogTitleClass}>
          {t(missing ? "documents.remove.title" : "documents.delete.title")}
        </h2>
        <p className={dialogTextClass}>
          {t(missing ? "documents.remove.body" : "documents.delete.body", {
            name: target?.name ?? "",
          })}
        </p>
        <div className={dialogActionsClass}>
          <button type="button" onClick={onClose} className={buttonClass}>
            {t("documents.delete.cancel")}
          </button>
          <button
            type="button"
            data-testid="confirm-delete-document"
            onClick={confirm}
            className={dangerButtonClass}
          >
            {t(missing ? "documents.remove.confirm" : "documents.delete.confirm")}
          </button>
        </div>
      </div>
    </dialog>
  );
}
