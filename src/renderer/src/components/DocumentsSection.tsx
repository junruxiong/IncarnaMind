import { type ChangeEvent, useEffect, useRef, useState } from "react";
import { useShallow } from "zustand/react/shallow";
import type {
  Document,
  DocumentFailureReason,
  DocumentStatus,
  EmbeddingModelError,
} from "../../../core/api";
import type { MessageKey } from "../../../shared/i18n";
import { endSidebarDrag, startSidebarDrag } from "../folders";
import { useT } from "../i18n";
import { selectVisibleDocuments, useAppStore } from "../store";
import { DocumentFileMenu } from "./DocumentFileMenu";
import {
  DocumentTagChips,
  DocumentTagMenu,
  TagFilter,
  TaggingStatus,
  TaggingWaitingNotice,
} from "./DocumentTags";
import { FolderTree } from "./FolderTree";
import {
  CloseIcon,
  DocumentIcon,
  FolderPlusIcon,
  PencilIcon,
  PlusIcon,
  TagIcon,
  TrashIcon,
} from "./icons";
import { embeddingProviderLabel } from "./providers/EmbeddingSettings";
import { testErrorKey } from "./providers/shared";

/** What the file picker offers. The core decides what it takes. */
const ACCEPTED_FILES = ".pdf,.docx,.pptx,.xlsx,.csv,.txt,.md,.markdown";

const statusMessages: Record<Exclude<DocumentStatus, "failed">, MessageKey> = {
  queued: "documents.status.queued",
  extracting: "documents.status.extracting",
  "waiting-for-model": "embedding.status.waiting",
  embedding: "embedding.status.embedding",
  ready: "documents.status.ready",
  "no-text": "documents.status.noText",
};

const failureMessages: Record<DocumentFailureReason, MessageKey> = {
  unreadable: "documents.failure.unreadable",
  "password-protected": "documents.failure.passwordProtected",
  "file-missing": "documents.failure.fileMissing",
  "processing-error": "documents.failure.processingError",
};

const statusTones: Record<DocumentStatus, string> = {
  queued: "text-gray-400",
  extracting: "text-gray-400 animate-pulse",
  "waiting-for-model": "text-gray-400",
  embedding: "text-gray-400 animate-pulse",
  ready: "text-gray-400",
  failed: "text-red-600",
  "no-text": "text-amber-600",
};

/**
 * The sidebar's Documents: a list with each Document's processing status, its
 * Tags and its tagging (with one notice above the list while tagging waits
 * for a model), an add button with a file picker, an "Add folder…" button
 * that links a folder, rename and delete, and a menu to open its file or
 * show it in its folder. Dropping files anywhere on the window adds them too
 * (see `FileDrop`). Above the list, the Folder tree (the Linked folders'
 * folders, as on disk) and the Tag chips filter it; Documents are tagged
 * from their Tags menu. Clicking a Document opens it in the viewer.
 */
export function DocumentsSection() {
  const t = useT();
  // Filtering makes a new array each time: compare it item by item, or React re-renders forever.
  const documents = useAppStore(useShallow(selectVisibleDocuments));
  const emptyMessage = useAppStore(
    (state): MessageKey =>
      state.tagFilter !== null
        ? "tags.filter.empty"
        : state.folderFilter !== null
          ? "folders.empty"
          : "documents.none",
  );
  const addDocuments = useAppStore((state) => state.addDocuments);
  const addLinkedFolder = useAppStore((state) => state.addLinkedFolder);
  const openTagsDialog = useAppStore((state) => state.openTagsDialog);
  const picker = useRef<HTMLInputElement>(null);
  const [deleting, setDeleting] = useState<Document | null>(null);

  const addPicked = (event: ChangeEvent<HTMLInputElement>) => {
    const picked = Array.from(event.target.files ?? []);
    event.target.value = ""; // so picking the same file again still counts as a change
    if (picked.length > 0) void addDocuments(picked);
  };

  return (
    <section aria-labelledby="documents-heading" className="pb-2">
      <div className="mx-4 mt-4 mb-1 flex items-center justify-between">
        <h2
          id="documents-heading"
          className="text-[11px] font-medium tracking-wide text-gray-400 uppercase"
        >
          {t("documents.title")}
        </h2>
        <div className="flex items-center gap-[2px]">
          <button
            type="button"
            data-testid="manage-tags"
            aria-label={t("tags.manage")}
            title={t("tags.manage")}
            onClick={openTagsDialog}
            className="rounded-[9px] p-[2px] text-gray-500 hover:bg-gray-200 hover:text-gray-700"
          >
            <TagIcon className="size-4" />
          </button>
          <button
            type="button"
            data-testid="add-linked-folder"
            aria-label={t("linkedFolders.add")}
            title={t("linkedFolders.add")}
            onClick={() => void addLinkedFolder()}
            className="rounded-[9px] p-[2px] text-gray-500 hover:bg-gray-200 hover:text-gray-700"
          >
            <FolderPlusIcon className="size-4" />
          </button>
          <button
            type="button"
            data-testid="add-documents"
            aria-label={t("documents.add")}
            title={t("documents.add")}
            onClick={() => picker.current?.click()}
            className="rounded-[9px] p-[2px] text-gray-500 hover:bg-gray-200 hover:text-gray-700"
          >
            <PlusIcon className="size-4" />
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
      <SkippedFilesNotice />
      <EmbeddingModelNotice />
      <EmbeddingRebuildNotice />
      <TaggingWaitingNotice />
      <FolderTree />
      <TagFilter />
      {documents.length === 0 ? (
        <p className="mx-4 py-[5px] text-sm text-gray-400">{t(emptyMessage)}</p>
      ) : (
        <ul className="mx-3">
          {documents.map((item) => (
            <DocumentItem key={item.id} item={item} onDelete={() => setDeleting(item)} />
          ))}
        </ul>
      )}
      <DeleteDocumentDialog target={deleting} onClose={() => setDeleting(null)} />
    </section>
  );
}

function DocumentItem({ item, onDelete }: { item: Document; onDelete(): void }) {
  const t = useT();
  const [renaming, setRenaming] = useState(false);
  const openDocument = useAppStore((state) => state.openDocument);
  const isOpen = useAppStore(
    (state) => state.viewerOpen && state.viewerTarget?.documentId === item.id,
  );
  const status =
    item.status === "failed"
      ? t("documents.status.failed", {
          reason: t(failureMessages[item.failure?.reason ?? "processing-error"]),
        })
      : t(statusMessages[item.status], { percent: Math.floor((item.progress ?? 0) * 100) });
  const actionButton = "rounded-[6px] p-[3px] text-gray-500 hover:bg-gray-200 hover:text-gray-700";
  const statusLine = (
    <p
      data-testid="document-status"
      title={item.failure?.message}
      className={`truncate text-[11px] leading-4 ${statusTones[item.status]}`}
    >
      {status}
    </p>
  );

  return (
    <li
      data-testid="document-list-item"
      data-document-id={item.id}
      data-folder-id={item.folderId ?? ""}
      data-status={item.status}
      data-file-status={item.fileStatus}
      data-tagging={item.tagging}
      draggable={!renaming}
      onDragStart={(event) => startSidebarDrag(event, { kind: "document", id: item.id })}
      onDragEnd={endSidebarDrag}
      className={`group my-[1px] flex items-start gap-[6px] rounded-[9px] px-1 py-[5px] text-sm ${
        isOpen ? "bg-gray-200" : "hover:bg-gray-100"
      }`}
    >
      <div className="min-w-0 flex-1">
        {renaming ? (
          <div className="flex items-start gap-[6px]">
            <DocumentIcon kind={item.kind} className="mt-[2px] size-4 shrink-0" />
            <div className="min-w-0 flex-1">
              <RenameInput item={item} onDone={() => setRenaming(false)} />
              {statusLine}
              <TaggingStatus item={item} />
            </div>
          </div>
        ) : (
          // Opens the Document in the viewer, replacing whatever it showed.
          <button
            type="button"
            data-testid="open-document"
            aria-current={isOpen ? "true" : undefined}
            onClick={() => openDocument({ documentId: item.id })}
            className="flex w-full min-w-0 items-start gap-[6px] text-left"
          >
            <DocumentIcon kind={item.kind} className="mt-[2px] size-4 shrink-0" />
            <span className="min-w-0 flex-1">
              <span className="block truncate text-gray-700" title={item.name}>
                {item.name}
              </span>
              {statusLine}
              <TaggingStatus item={item} />
            </span>
          </button>
        )}
        <DocumentTagChips item={item} />
      </div>
      {!renaming && (
        <div className="flex shrink-0 items-center opacity-0 group-focus-within:opacity-100 group-hover:opacity-100">
          <DocumentTagMenu item={item} buttonClassName={actionButton} />
          <button
            type="button"
            aria-label={t("documents.rename", { name: item.name })}
            title={t("documents.rename", { name: item.name })}
            onClick={() => setRenaming(true)}
            className={actionButton}
          >
            <PencilIcon className="size-[14px]" />
          </button>
          <button
            type="button"
            data-testid="delete-document"
            aria-label={t("documents.delete", { name: item.name })}
            title={t("documents.delete", { name: item.name })}
            onClick={onDelete}
            className={actionButton}
          >
            <TrashIcon className="size-[14px]" />
          </button>
          <DocumentFileMenu item={item} buttonClassName={actionButton} />
        </div>
      )}
    </li>
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
      className="w-full rounded-[6px] border border-gray-300 bg-white px-1 text-sm text-gray-700 outline-none focus:border-gray-400"
    />
  );
}

function SkippedFilesNotice() {
  const t = useT();
  const skipped = useAppStore((state) => state.skippedFiles);
  const dismiss = useAppStore((state) => state.dismissSkippedFiles);
  if (skipped.length === 0) return null;
  return (
    <div
      role="status"
      className="mx-3 mb-1 flex items-start gap-2 rounded-[9px] bg-amber-50 px-2 py-[5px] text-[12px] text-amber-800"
    >
      <p className="min-w-0 flex-1 break-words">
        {t("documents.skipped", { names: skipped.join(", ") })}
      </p>
      <button
        type="button"
        aria-label={t("error.dismiss")}
        title={t("error.dismiss")}
        onClick={dismiss}
        className="shrink-0 rounded-[6px] p-[2px] hover:bg-amber-100"
      >
        <CloseIcon className="size-3" />
      </button>
    </div>
  );
}

/** Decimal megabytes, as the model's 135 MB is quoted elsewhere. */
const MEGABYTE = 1_000_000;

const modelFailures: Record<Exclude<EmbeddingModelError["kind"], "load">, MessageKey> = {
  network: "embedding.model.failure.network",
  integrity: "embedding.model.failure.integrity",
  storage: "embedding.model.failure.storage",
};

/**
 * The built-in embedding model's download, shown while it runs and when it
 * fails (with a retry). The core starts it when the first Document needs it.
 */
function EmbeddingModelNotice() {
  const t = useT();
  const model = useAppStore((state) => state.embeddingModel);
  const retry = useAppStore((state) => state.downloadEmbeddingModel);
  if (model?.state === "downloading") {
    const share = model.totalBytes > 0 ? model.downloadedBytes / model.totalBytes : 0;
    return (
      <div
        role="status"
        data-testid="embedding-model-download"
        className="mx-3 mb-1 rounded-[9px] bg-gray-100 px-2 py-[5px] text-[12px] text-gray-600"
      >
        <p>
          {t("embedding.model.downloading", {
            downloaded: Math.floor(model.downloadedBytes / MEGABYTE),
            total: Math.ceil(model.totalBytes / MEGABYTE),
          })}
        </p>
        <div className="mt-1 h-1 overflow-hidden rounded-full bg-gray-200">
          <div className="h-full bg-gray-500" style={{ width: `${Math.round(share * 100)}%` }} />
        </div>
        <p className="mt-1 text-[11px] leading-4 text-gray-400">{t("embedding.model.note")}</p>
      </div>
    );
  }
  if (model?.state !== "failed" || !model.error) return null;
  const { kind, message } = model.error;
  return (
    <div
      role="alert"
      data-testid="embedding-model-failed"
      className="mx-3 mb-1 flex items-start gap-2 rounded-[9px] bg-amber-50 px-2 py-[5px] text-[12px] text-amber-800"
    >
      <p className="min-w-0 flex-1 break-words" title={message}>
        {kind === "load"
          ? t("embedding.model.loadFailed")
          : t("embedding.model.failed", { reason: t(modelFailures[kind]) })}
      </p>
      <button
        type="button"
        data-testid="embedding-model-retry"
        onClick={() => void retry()}
        className="shrink-0 rounded-[6px] px-1 font-medium hover:bg-amber-100"
      >
        {t("embedding.model.retry")}
      </button>
    </div>
  );
}

/**
 * After the embedding model changed: how many Documents are embedded with the
 * new one, and why it changed if local mode did it. While the chosen provider
 * can't be used, what is wrong, with a retry.
 */
function EmbeddingRebuildNotice() {
  const t = useT();
  const embedding = useAppStore((state) => state.embedding);
  const retry = useAppStore((state) => state.retryEmbedding);
  if (!embedding) return null;
  const { rebuild, error, provider } = embedding;
  if (error) {
    return (
      <div
        role="alert"
        data-testid="embedding-provider-error"
        className="mx-3 mb-1 flex items-start gap-2 rounded-[9px] bg-amber-50 px-2 py-[5px] text-[12px] text-amber-800"
      >
        <p className="min-w-0 flex-1 break-words" title={error.message}>
          {t("embeddingProviders.error.notice", {
            provider: embeddingProviderLabel(provider, t),
            reason: t(testErrorKey(error.kind)),
          })}
        </p>
        <button
          type="button"
          onClick={() => void retry()}
          className="shrink-0 rounded-[6px] px-1 font-medium hover:bg-amber-100"
        >
          {t("embeddingProviders.settings.retry")}
        </button>
      </div>
    );
  }
  if (!rebuild) return null;
  const share = rebuild.total > 0 ? rebuild.done / rebuild.total : 0;
  return (
    <div
      role="status"
      data-testid="embedding-rebuild-notice"
      className="mx-3 mb-1 rounded-[9px] bg-gray-100 px-2 py-[5px] text-[12px] text-gray-600"
    >
      {rebuild.reason === "local-mode" && (
        <p className="mb-1 text-gray-700">{t("embeddingProviders.rebuild.localMode")}</p>
      )}
      <p>{t("embeddingProviders.rebuild.title", { done: rebuild.done, total: rebuild.total })}</p>
      <div className="mt-1 h-1 overflow-hidden rounded-full bg-gray-200">
        <div className="h-full bg-gray-500" style={{ width: `${Math.round(share * 100)}%` }} />
      </div>
      <p className="mt-1 text-[11px] leading-4 text-gray-400">
        {t("embeddingProviders.rebuild.note")}
      </p>
    </div>
  );
}

/** A native modal <dialog>, like the settings. */
function DeleteDocumentDialog({ target, onClose }: { target: Document | null; onClose(): void }) {
  const t = useT();
  const dialog = useRef<HTMLDialogElement>(null);
  const deleteDocument = useAppStore((state) => state.deleteDocument);

  useEffect(() => {
    const element = dialog.current;
    if (!element) return;
    if (target && !element.open) element.showModal();
    if (!target && element.open) element.close();
  }, [target]);

  const confirm = () => {
    if (target) void deleteDocument(target.id);
    onClose();
  };

  return (
    <dialog
      ref={dialog}
      onClose={onClose}
      aria-labelledby="delete-document-title"
      className="m-auto w-96 rounded-[9px] bg-white p-4 text-gray-800 shadow-custom-focus backdrop:bg-black/20"
    >
      <h2 id="delete-document-title" className="text-lg font-semibold">
        {t("documents.delete.title")}
      </h2>
      <p className="mt-2 text-sm break-words text-gray-600">
        {t("documents.delete.body", { name: target?.name ?? "" })}
      </p>
      <div className="mt-4 flex justify-end gap-2">
        <button
          type="button"
          onClick={onClose}
          className="rounded-[9px] border border-gray-300 px-4 py-2 text-sm hover:bg-gray-100"
        >
          {t("documents.delete.cancel")}
        </button>
        <button
          type="button"
          data-testid="confirm-delete-document"
          onClick={confirm}
          className="rounded-[9px] bg-red-600 px-4 py-2 text-sm text-white hover:bg-red-700"
        >
          {t("documents.delete.confirm")}
        </button>
      </div>
    </dialog>
  );
}
