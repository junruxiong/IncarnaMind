import { type ChangeEvent, useEffect, useRef, useState } from "react";
import { useShallow } from "zustand/react/shallow";
import type { Document, DocumentFailureReason, DocumentStatus } from "../../../core/api";
import type { MessageKey } from "../../../shared/i18n";
import { endSidebarDrag, startSidebarDrag } from "../folders";
import { useT } from "../i18n";
import { selectVisibleDocuments, useAppStore } from "../store";
import { DocumentFileMenu } from "./DocumentFileMenu";
import { ActiveTagFilter, DocumentTagMenu, TagFilterMenu } from "./DocumentTags";
import { FolderTree, type NewFolderPlace, rowInputClass, useDropTarget } from "./FolderTree";
import { DocumentLineIcon, FolderPlusLineIcon, PlusLineIcon } from "./lineIcons";
import { MoveToMenu } from "./MoveToMenu";
import {
  dropTargetClass,
  rowActionButtonClass,
  rowActionsClass,
  rowButtonClass,
  rowClass,
  rowIconClass,
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
const ACCEPTED_FILES = ".pdf,.txt,.md,.markdown";

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
 * a new Folder and adding files), the tree of Folders with their Documents,
 * then the Documents in no Folder. Each Document is one row: its name and,
 * while it's processed or if it failed, its status at the end. Its Tags are
 * in its Tags menu, and the Tags dialog. Dropping files anywhere on the window
 * adds them too (see `FileDrop`). Documents are filed by dragging them onto a
 * Folder or with "Move to…". Clicking a Document opens it in the viewer.
 */
export function DocumentsSection() {
  const t = useT();
  // Filtering makes a new array each time: compare it item by item, or React re-renders forever.
  const documents = useAppStore(useShallow(selectVisibleDocuments));
  const filtering = useAppStore((state) => state.tagFilter !== null);
  const hasFolders = useAppStore((state) => state.folders.length > 0);
  const addDocuments = useAppStore((state) => state.addDocuments);
  const picker = useRef<HTMLInputElement>(null);
  const [deleting, setDeleting] = useState<Document | null>(null);
  const [newFolderIn, setNewFolderIn] = useState<NewFolderPlace>(undefined);
  const topLevel = useDropTarget(null);

  const addPicked = (event: ChangeEvent<HTMLInputElement>) => {
    const picked = Array.from(event.target.files ?? []);
    event.target.value = ""; // so picking the same file again still counts as a change
    if (picked.length > 0) void addDocuments(picked);
  };

  const empty = documents.length === 0 && (filtering || !hasFolders) && newFolderIn === undefined;

  return (
    <section aria-labelledby="documents-heading">
      {/* The label is also where a Document or Folder is dropped to go to the top level. */}
      <div
        data-testid="documents-heading-row"
        {...topLevel.handlers}
        className={`mt-3 flex h-7 shrink-0 items-end justify-between rounded-md pr-1 pb-0.5 pl-2 ${
          topLevel.over ? dropTargetClass : ""
        }`}
      >
        <h2 id="documents-heading" className="pb-0.5 text-label font-semibold text-ink-meta">
          {t("documents.title")}
        </h2>
        <div className="flex items-center">
          <TagFilterMenu />
          <button
            type="button"
            data-testid="new-folder"
            aria-label={t("folders.new")}
            title={t("folders.new")}
            onClick={() => setNewFolderIn(null)}
            className={rowActionButtonClass}
          >
            <FolderPlusLineIcon className="size-4" />
          </button>
          <button
            type="button"
            data-testid="add-documents"
            aria-label={t("documents.add")}
            title={t("documents.add")}
            onClick={() => picker.current?.click()}
            className={rowActionButtonClass}
          >
            <PlusLineIcon className="size-4" />
          </button>
        </div>
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
      <ActiveTagFilter />
      <FolderTree
        documents={documents}
        filtering={filtering}
        newFolderIn={newFolderIn}
        onNewFolder={setNewFolderIn}
        onNewFolderDone={() => setNewFolderIn(undefined)}
        renderDocument={(item, depth) => (
          <DocumentRow key={item.id} item={item} depth={depth} onDelete={() => setDeleting(item)} />
        )}
      />
      {empty && (
        <p className="px-2 py-1 text-[13px] leading-5 text-ink-meta">
          {t(filtering ? "tags.filter.empty" : "documents.none")}
        </p>
      )}
      <DeleteDocumentDialog target={deleting} onClose={() => setDeleting(null)} />
    </section>
  );
}

/**
 * A Document's row: its icon and name, and its status at the end while it's
 * processed or if it failed. Pointed at, it offers its Tags, "Move to…" and
 * a menu for the rest. Dragged, it can be dropped on a Folder.
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
  const muted = !isOpen && isProcessing(item.status);

  return (
    <li
      data-testid="document-list-item"
      data-document-id={item.id}
      data-folder-id={item.folderId ?? ""}
      data-depth={depth}
      data-status={item.status}
      data-tagging={item.tagging}
      draggable={!renaming}
      onDragStart={(event) => startSidebarDrag(event, { kind: "document", id: item.id })}
      onDragEnd={endSidebarDrag}
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
          <MoveToMenu item={item} buttonClassName={rowActionButtonClass} />
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
 * The status at a row's end: "38%" while it's embedded, "Failed" if it
 * failed, nothing once it's ready. Its full words (and a failure's message)
 * are its tooltip, and what a screen reader reads.
 */
function DocumentStatusLabel({ item }: { item: Document }) {
  const t = useT();
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

/** Asks before deleting a Document. */
function DeleteDocumentDialog({ target, onClose }: { target: Document | null; onClose(): void }) {
  const t = useT();
  const dialog = useModal(target !== null);
  const deleteDocument = useAppStore((state) => state.deleteDocument);

  const confirm = () => {
    if (target) void deleteDocument(target.id);
    onClose();
  };

  return (
    <dialog
      ref={dialog}
      onClose={onClose}
      aria-labelledby="delete-document-title"
      className={`${dialogClass} w-[26rem]`}
    >
      <div className={dialogBodyClass}>
        <h2 id="delete-document-title" className={dialogTitleClass}>
          {t("documents.delete.title")}
        </h2>
        <p className={dialogTextClass}>
          {t("documents.delete.body", { name: target?.name ?? "" })}
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
            {t("documents.delete.confirm")}
          </button>
        </div>
      </div>
    </dialog>
  );
}
