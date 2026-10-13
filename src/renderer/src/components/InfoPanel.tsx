import { type ReactNode, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import type { Document, LinkedFolder } from "../../../core/api";
import { displayName } from "../../../shared/documentNames";
import { copyText } from "../copied";
import { core, files } from "../core";
import { documentStatusText } from "../documentStatus";
import { addedDate, fileDate, KIND_LABELS, showFileLabelKey } from "../fileInfo";
import { useLanguage, useT } from "../i18n";
import { useInfo } from "../info";
import {
  fileStatusLabel,
  folderName,
  formatBytes,
  formatCount,
  linkedFolderRowState,
  rowStateLabel,
} from "../linkedFolders";
import { type Box, enterFrom, useExitToward } from "../motion";
import { useNotices } from "../notices";
import { scanningStatus } from "../scanStatus";
import { useAppStore } from "../store";
import { chipsOf } from "../tagEditing";
import { DocumentLineIcon, FolderLineIcon } from "./lineIcons";
import { TagDot } from "./TagColour";
import { buttonClass } from "./ui";

const WIDTH_PX = 340;
const EDGE_PX = 8;
const GAP_PX = 6;

const panelClass =
  "inset-auto m-0 max-h-[calc(100vh-16px)] max-w-[calc(100vw-16px)] overflow-y-auto rounded-lg border-0 bg-sheet p-0 text-ui text-ink shadow-popover outline-none";

const rowOf = (from: HTMLElement | null) => from?.closest<HTMLElement>("[data-object-row]") ?? null;

/**
 * Beside the row it is about if there is room, else under it (or over it, in
 * a short window); in the middle of the window when nothing shows the object.
 */
function place(panel: HTMLElement, row: HTMLElement | null): Box {
  const { width, height } = panel.getBoundingClientRect();
  const around = row?.getBoundingClientRect();
  const clamp = (value: number, size: number, limit: number) =>
    Math.max(EDGE_PX, Math.min(value, limit - size - EDGE_PX));
  let left = (window.innerWidth - width) / 2;
  let top = window.innerHeight / 4;
  if (around) {
    const beside = window.innerWidth - around.right >= width + EDGE_PX + GAP_PX;
    left = beside ? around.right + GAP_PX : around.left + 16;
    top = beside ? around.top : around.bottom + GAP_PX;
    if (!beside && top + height > window.innerHeight - EDGE_PX) top = around.top - GAP_PX - height;
  }
  const box = {
    left: clamp(left, width, window.innerWidth),
    top: clamp(top, height, window.innerHeight),
    width,
    height,
  };
  panel.style.left = `${box.left}px`;
  panel.style.top = `${box.top}px`;
  return box;
}

/**
 * Get info (⌘I): a small panel beside a Document's or Linked folder's row
 * with what IncarnaMind knows of it. A popover, so Esc or a click outside
 * closes it and the focus goes back to the row.
 */
export function InfoPanel() {
  const target = useInfo((state) => state.target);
  return target ? <Panel key={`${target.kind}:${target.id}`} /> : null;
}

function Panel() {
  const t = useT();
  const { target, from } = useInfo();
  const item = useAppStore((state) =>
    target?.kind === "document" ? state.documents.find((each) => each.id === target.id) : undefined,
  );
  const linked = useAppStore((state) =>
    target?.kind === "linkedFolder"
      ? state.linkedFolders.find((each) => each.id === target.id)
      : undefined,
  );
  const panel = useRef<HTMLDivElement>(null);
  const row = rowOf(from);
  useExitToward(panel, () => row ?? from);

  // biome-ignore lint/correctness/useExhaustiveDependencies: placed where it opened, not as the row moves.
  useLayoutEffect(() => {
    const element = panel.current;
    if (!element) return;
    element.showPopover();
    enterFrom(element, row, place(element, row));
    element.focus();
  }, []);

  // What it was about is gone (deleted, unlinked): nothing left to show.
  const gone = !item && !linked;
  useEffect(() => {
    if (gone) useInfo.getState().close();
  }, [gone]);

  const name = item ? displayName(item) : linked ? folderName(linked.path) : "";
  return (
    <div
      ref={panel}
      popover="auto"
      role="dialog"
      tabIndex={-1}
      aria-label={t("info.label", { name })}
      data-testid="info-panel"
      onKeyDown={(event) => {
        // Closes only this, not the Document viewer behind it.
        if (event.key !== "Escape") return;
        event.preventDefault();
        event.stopPropagation();
        panel.current?.hidePopover();
      }}
      onToggle={(event) => {
        if (event.newState !== "closed") return;
        useInfo.getState().close();
        // Back to what it was opened from, unless the click that closed it went somewhere to focus.
        if (from?.isConnected && !document.activeElement?.closest("[data-surface]")) from.focus();
      }}
      style={{ width: WIDTH_PX }}
      className={panelClass}
    >
      {item ? <DocumentInfo document={item} /> : linked && <LinkedFolderInfo linked={linked} />}
    </div>
  );
}

function Header({ icon, name, line }: { icon: ReactNode; name: string; line: string }) {
  return (
    <div className="flex items-center gap-2.5 border-b border-rule px-4 pt-3.5 pb-2.5">
      <span className="flex size-7 shrink-0 items-center justify-center text-ink-meta">{icon}</span>
      <div className="min-w-0 flex-1">
        <p data-testid="info-name" className="truncate text-ui leading-5 font-semibold text-ink">
          {name}
        </p>
        <p data-testid="info-summary" className="truncate text-[12px] leading-4 text-ink-meta">
          {line}
        </p>
      </div>
    </div>
  );
}

function Rows({ children }: { children: ReactNode }) {
  return (
    <dl className="grid grid-cols-[auto_minmax(0,1fr)] gap-x-4 gap-y-1 px-4 py-3 text-[13px] leading-5 select-text">
      {children}
    </dl>
  );
}

function Row({ label, testId, children }: { label: string; testId: string; children: ReactNode }) {
  return (
    <>
      <dt className="text-ink-meta">{label}</dt>
      <dd data-testid={testId} className="min-w-0 break-words text-ink [overflow-wrap:anywhere]">
        {children}
      </dd>
    </>
  );
}

function Actions({
  path,
  canShow,
  showLabel,
  onShow,
}: {
  path: string;
  canShow: boolean;
  showLabel: string;
  onShow(): void;
}) {
  const t = useT();
  const copied = useNotices((state) => state.confirmation?.message === "info.copied");
  return (
    <div className="flex flex-wrap gap-2 px-4 pb-3.5">
      <button
        type="button"
        data-testid="info-copy-path"
        onClick={() =>
          void copyText(path).catch((error) => useAppStore.getState().reportError(error))
        }
        className={buttonClass}
      >
        {copied ? t("info.copied") : t("info.copyPath")}
      </button>
      {canShow && (
        <button type="button" data-testid="info-show" onClick={onShow} className={buttonClass}>
          {showLabel}
        </button>
      )}
      <span role="status" aria-live="polite" className="sr-only">
        {copied ? t("info.copied") : ""}
      </span>
    </div>
  );
}

/** Its slides or sheets, counted from the text the core kept: the Document itself holds only a PDF's pages. */
function useUnitCount(
  document: Document,
): { key: "info.slides" | "info.sheets"; count: number } | null {
  const [count, setCount] = useState<number | null>(null);
  const { id, kind, contentHash } = document;
  // biome-ignore lint/correctness/useExhaustiveDependencies: counted again when the file's version changes.
  useEffect(() => {
    if (kind !== "pptx" && kind !== "xlsx") return;
    let current = true;
    core
      .readDocumentText(id)
      .then(({ pages }) => {
        if (!current) return;
        setCount(
          kind === "pptx"
            ? pages.length
            : new Set(pages.map((page) => page.label?.sheet ?? "")).size,
        );
      })
      .catch(() => undefined);
    return () => {
      current = false;
    };
  }, [id, kind, contentHash]);
  if (count === null || count === 0) return null;
  return { key: kind === "pptx" ? "info.slides" : "info.sheets", count };
}

function DocumentInfo({ document }: { document: Document }) {
  const t = useT();
  const language = useLanguage();
  const tags = useAppStore((state) => state.tags);
  const folder = useAppStore((state) => {
    const groupId = state.library?.assignments.find(
      (each) => each.documentId === document.id,
    )?.groupId;
    return groupId ? state.library?.groups.find((each) => each.id === groupId)?.name : undefined;
  });
  const linkedName = useAppStore((state) => {
    const linked = state.linkedFolders.find((each) => each.id === document.linkedFolderId);
    return linked && folderName(linked.path);
  });
  const units = useUnitCount(document);
  const chips = chipsOf(document.tags, tags);
  const file = !document.web;
  const fileState = fileStatusLabel(document.fileStatus);
  const scanning =
    document.status === "extracting" && document.scan
      ? scanningStatus(document.scan, t).full
      : null;
  const status = documentStatusText(document, t, language, scanning);

  const counted = (key: "info.pages" | "info.slides" | "info.sheets", count: number) =>
    t(`${key}.${count === 1 ? "one" : "other"}`, { count: formatCount(count, language) });
  const summary = [
    t(KIND_LABELS[document.kind]),
    document.pageCount !== null
      ? counted("info.pages", document.pageCount)
      : units && counted(units.key, units.count),
    formatBytes(document.size, t),
  ]
    .filter(Boolean)
    .join(" · ");

  return (
    <>
      <Header
        icon={<DocumentLineIcon kind={document.kind} className="size-7" />}
        name={displayName(document)}
        line={summary}
      />
      <Rows>
        {document.web ? (
          <Row label={t("info.address")} testId="info-address">
            {document.web.address}
          </Row>
        ) : (
          <Row label={t("info.where")} testId="info-path">
            {document.path}
          </Row>
        )}
        {file && (
          <Row label={t("info.from")} testId="info-from">
            {linkedName ? t("info.from.linked", { name: linkedName }) : t("info.from.own")}
          </Row>
        )}
        <Row label={t("info.folder")} testId="info-folder">
          {folder ?? t("library.unsorted")}
        </Row>
        {chips.length > 0 && (
          <Row label={t("info.tags")} testId="info-tags">
            <span className="flex flex-wrap gap-x-3 gap-y-0.5">
              {chips.map(({ tag }) => (
                <span key={tag.id} className="inline-flex items-center gap-1.5">
                  <TagDot colour={tag.colour} />
                  {tag.name}
                </span>
              ))}
            </span>
          </Row>
        )}
        {document.creationDate && (
          <Row label={t("info.created")} testId="info-created">
            {t("info.createdValue", { date: fileDate(document.creationDate, language) })}
          </Row>
        )}
        <Row label={t("info.added")} testId="info-added">
          {addedDate(document.createdAt, language)}
        </Row>
        <Row label={t("info.status")} testId="info-status">
          {status}
          {fileState && <span className="block text-ink-meta">{t(fileState.full)}</span>}
          {document.failure?.message && document.status === "failed" && (
            <span className="block text-ink-meta">{document.failure.message}</span>
          )}
        </Row>
      </Rows>
      {file && (
        <Actions
          path={document.path}
          canShow={document.fileStatus === "available"}
          showLabel={t(showFileLabelKey(navigator.platform))}
          onShow={() =>
            void files
              .showDocumentInFolder(document.id)
              .catch((error) => useAppStore.getState().reportError(error))
          }
        />
      )}
    </>
  );
}

function LinkedFolderInfo({ linked }: { linked: LinkedFolder }) {
  const t = useT();
  const language = useLanguage();
  const documents = useAppStore((state) => state.documents);
  const bytes = useMemo(
    () =>
      documents.reduce(
        (sum, each) => (each.linkedFolderId === linked.id ? sum + each.size : sum),
        0,
      ),
    [documents, linked.id],
  );
  const state = rowStateLabel(linkedFolderRowState(linked, true), t, language);
  const files = linked.progress.files;
  const count = t(`info.files.${files === 1 ? "one" : "other"}`, {
    count: formatCount(files, language),
  });
  const size = formatBytes(bytes, t);
  return (
    <>
      <Header
        icon={<FolderLineIcon className="size-7" />}
        name={folderName(linked.path)}
        line={`${count} · ${size}`}
      />
      <Rows>
        <Row label={t("info.where")} testId="info-path">
          {linked.path}
        </Row>
        <Row label={t("info.files")} testId="info-files">
          {count}
        </Row>
        <Row label={t("info.size")} testId="info-size">
          {size}
        </Row>
        <Row label={t("info.added")} testId="info-added">
          {addedDate(linked.createdAt, language)}
        </Row>
        <Row label={t("info.status")} testId="info-status">
          {state?.full ?? t("info.upToDate")}
        </Row>
      </Rows>
      <Actions
        path={linked.path}
        canShow={linked.status !== "unavailable"}
        showLabel={t(showFileLabelKey(navigator.platform))}
        onShow={() => void useAppStore.getState().showLinkedFolder(linked.id)}
      />
    </>
  );
}
