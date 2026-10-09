import { type ReactNode, type ToggleEvent, useId, useRef, useState } from "react";
import { useShallow } from "zustand/react/shallow";
import type { DocumentStatus, EmbeddingModelError } from "../../../core/api";
import type { MessageKey } from "../../../shared/i18n";
import { useT } from "../i18n";
import { rerankingModelStatus } from "../rerankStatus";
import { useAppStore } from "../store";
import { CloseLineIcon } from "./lineIcons";
import { embeddingProviderLabel } from "./providers/EmbeddingSettings";
import { testErrorKey } from "./providers/shared";
import { rowActionButtonClass } from "./sidebarRows";

/** The status row's button: a small ghost one, inside the 28px row. */
const statusActionClass =
  "inline-flex h-6 shrink-0 items-center rounded-md px-2 text-[12px] font-semibold text-ink-secondary hover:bg-rule hover:text-ink focus-visible:outline-offset-0 aria-expanded:bg-rule aria-expanded:text-ink";

/** Decimal megabytes, as the model's 135 MB is quoted elsewhere. */
const MEGABYTE = 1_000_000;

/** The gap between the "+N" button and the list of the other statuses above it. */
const GAP_PX = 4;

const modelFailures: Record<Exclude<EmbeddingModelError["kind"], "load">, MessageKey> = {
  network: "embedding.model.failure.network",
  integrity: "embedding.model.failure.integrity",
  storage: "embedding.model.failure.storage",
};

const PROCESSING: ReadonlySet<DocumentStatus> = new Set([
  "queued",
  "extracting",
  "waiting-for-model",
  "embedding",
]);

/** In progress (a blue dot), waiting for the User (amber), failed (red), or for information (grey). */
type Tone = "progress" | "waiting" | "error" | "info";

const dotTones: Record<Tone, string> = {
  progress: "bg-accent",
  waiting: "bg-attention",
  error: "bg-danger",
  info: "bg-ink-placeholder",
};

interface Status {
  testId: string;
  tone: Tone;
  /** What the row shows: short enough for one line. */
  short: string;
  /** All of it: the tooltip, and what a screen reader reads. */
  full: string;
  action?: { label: string; testId?: string; run(): void };
  dismiss?: () => void;
}

/**
 * The sidebar footer's status row, above Settings: what the app is doing
 * with the Documents (processing, tagging, the search model's download or a
 * rebuild) or what needs the User (tagging waits for a model, a failure to
 * retry, skipped files, files that were in IncarnaMind already). One line:
 * the most pressing status, and "+N" for the others, which lists them above
 * the row. Its 28px are kept even when there is nothing to say, so the tree
 * above never moves.
 */
export function SidebarStatus() {
  const [first, ...others] = useStatuses();
  return (
    <div data-testid="sidebar-status" className="flex h-7 shrink-0 flex-col">
      {first && (
        <StatusRow status={first}>
          {others.length > 0 && <MoreStatuses others={others} />}
        </StatusRow>
      )}
    </div>
  );
}

function StatusRow({ status, children }: { status: Status; children?: ReactNode }) {
  const t = useT();
  const { tone, short, full, action, dismiss } = status;
  return (
    <div
      role={tone === "error" ? "alert" : "status"}
      data-testid={status.testId}
      title={full}
      className="flex h-7 shrink-0 items-center gap-2 pr-1 pl-2 text-[13px] leading-5 text-ink-meta"
    >
      <span aria-hidden="true" className="flex w-4 shrink-0 justify-center">
        <span
          className={`size-1.5 rounded-full ${dotTones[tone]} ${
            tone === "progress" ? "motion-safe:animate-pulse" : ""
          }`}
        />
      </span>
      <span aria-hidden="true" className="min-w-0 flex-1 truncate">
        {short}
      </span>
      <span className="sr-only">{full}</span>
      {children}
      {action && (
        <button
          type="button"
          data-testid={action.testId}
          onClick={action.run}
          className={statusActionClass}
        >
          {action.label}
        </button>
      )}
      {dismiss && (
        <button
          type="button"
          aria-label={t("error.dismiss")}
          title={t("error.dismiss")}
          onClick={dismiss}
          className={rowActionButtonClass}
        >
          <CloseLineIcon className="size-3.5" />
        </button>
      )}
    </div>
  );
}

/**
 * "+N": the statuses the row has no room for, listed (each with its own
 * action) in a popover above it, so none is out of reach.
 */
function MoreStatuses({ others }: { others: Status[] }) {
  const t = useT();
  const id = useId();
  const button = useRef<HTMLButtonElement>(null);
  const [open, setOpen] = useState(false);
  const label = t("status.more.label", { count: others.length });

  /** Above the button, its left edge on the row's. */
  const place = (list: HTMLElement) => {
    const anchor = button.current
      ?.closest('[data-testid="sidebar-status"]')
      ?.getBoundingClientRect();
    if (!anchor) return;
    list.style.left = `${anchor.left}px`;
    list.style.width = `${anchor.width}px`;
    list.style.bottom = `${window.innerHeight - anchor.top + GAP_PX}px`;
  };

  return (
    <>
      <button
        ref={button}
        type="button"
        data-testid="status-more"
        popoverTarget={id}
        aria-expanded={open}
        aria-label={label}
        title={label}
        className={statusActionClass}
      >
        {t("status.more", { count: others.length })}
      </button>
      <div
        id={id}
        popover="auto"
        role="dialog"
        aria-label={label}
        data-testid="status-others"
        onBeforeToggle={(event: ToggleEvent<HTMLDivElement>) => {
          if (event.newState === "open") place(event.currentTarget);
          setOpen(event.newState === "open");
        }}
        className="inset-auto m-0 rounded-lg border-0 bg-sheet p-1 font-normal shadow-popover [&>*+*]:border-t [&>*+*]:border-rule"
      >
        {/* Only while open, so a closed list adds nothing for screen readers to announce. */}
        {open && others.map((status) => <StatusRow key={status.testId} status={status} />)}
      </div>
    </>
  );
}

/** Every status worth showing, the most pressing first. Empty when there is nothing to say. */
function useStatuses(): Status[] {
  const t = useT();
  const skipped = useAppStore((state) => state.skippedFiles);
  const alreadyAdded = useAppStore((state) => state.alreadyAdded);
  const model = useAppStore((state) => state.embeddingModel);
  const embedding = useAppStore((state) => state.embedding);
  const library = useAppStore((state) => state.library);
  const organizing = !!library?.groups.length || !!library?.settings.classifier;
  const counts = useAppStore(
    useShallow((state) => {
      let processing = 0;
      let tagging = 0;
      let waitingForTagger = 0;
      // A paused Linked folder's Documents wait for the User: its row says so, not the footer.
      const paused = new Set(
        state.linkedFolders.filter((each) => each.status === "paused").map((each) => each.id),
      );
      for (const item of state.documents) {
        if (PROCESSING.has(item.status)) {
          if (item.linkedFolderId === null || !paused.has(item.linkedFolderId)) processing++;
        } else if (item.status === "ready") {
          if (item.tagging === "pending" || item.tagging === "tagging") tagging++;
          else if (item.tagging === "waiting-for-provider") waitingForTagger++;
        }
      }
      return { processing, tagging, waitingForTagger };
    }),
  );
  const retryDownload = useAppStore((state) => state.downloadEmbeddingModel);
  const rerank = useAppStore((state) => state.rerank);
  const retryRerankingDownload = useAppStore((state) => state.downloadRerankingModel);
  const retryEmbedding = useAppStore((state) => state.retryEmbedding);
  const dismissSkipped = useAppStore((state) => state.dismissSkippedFiles);
  const dismissAlreadyAdded = useAppStore((state) => state.dismissAlreadyAdded);
  const openSettings = useAppStore((state) => state.openSettings);
  const statuses: Status[] = [];
  // The built-in reranking model, on by default, downloads once there are Documents to search.
  const rerankingStatus = rerankingModelStatus(rerank, t);
  const reranking: Status | null = rerankingStatus && {
    testId: rerankingStatus.testId,
    tone: rerankingStatus.tone,
    short: rerankingStatus.short,
    full: rerankingStatus.full,
    ...(rerankingStatus.retry && {
      action: {
        label: t("status.retry"),
        testId: "reranking-model-retry",
        run: () => void retryRerankingDownload(),
      },
    }),
  };

  if (model?.state === "failed" && model.error) {
    const { kind, message } = model.error;
    const full =
      kind === "load"
        ? t("embedding.model.loadFailed")
        : t("embedding.model.failed", { reason: t(modelFailures[kind]) });
    statuses.push({
      testId: "embedding-model-failed",
      tone: "error",
      short: t(kind === "load" ? "status.modelFailed" : "status.downloadFailed"),
      full: `${full}\n${message}`,
      action: {
        label: t("status.retry"),
        testId: "embedding-model-retry",
        run: () => void retryDownload(),
      },
    });
  }

  if (reranking?.tone === "error") statuses.push(reranking);

  if (embedding?.error) {
    const provider = embeddingProviderLabel(embedding.provider, t);
    statuses.push({
      testId: "embedding-provider-error",
      tone: "error",
      short: t("status.searchError", { provider }),
      full: `${t("embeddingProviders.error.notice", {
        provider,
        reason: t(testErrorKey(embedding.error.kind)),
      })}\n${embedding.error.message}`,
      action: { label: t("status.retry"), run: () => void retryEmbedding() },
    });
  }

  if (skipped.length > 0) {
    statuses.push({
      testId: "skipped-files",
      tone: "waiting",
      short:
        skipped.length === 1
          ? t("status.skipped.one")
          : t("status.skipped.other", { count: skipped.length }),
      full: t("documents.skipped", { names: skipped.join(", ") }),
      dismiss: dismissSkipped,
    });
  }

  if (alreadyAdded.length > 0) {
    statuses.push({
      testId: "already-added",
      tone: "info",
      short:
        alreadyAdded.length === 1
          ? t("status.alreadyAdded.one")
          : t("status.alreadyAdded.other", { count: alreadyAdded.length }),
      full: t("documents.alreadyAdded", { names: alreadyAdded.join(", ") }),
      dismiss: dismissAlreadyAdded,
    });
  }

  if (!organizing && counts.waitingForTagger > 0) {
    statuses.push({
      testId: "tagging-waiting",
      tone: "waiting",
      short: t("status.taggingWaiting"),
      full: t("jev.waiting.notice"),
      action: {
        label: t("status.setUp"),
        testId: "tagging-waiting-setup",
        run: () => openSettings("chat-model"),
      },
    });
  }

  if (model?.state === "downloading") {
    const downloaded = Math.floor(model.downloadedBytes / MEGABYTE);
    const total = Math.ceil(model.totalBytes / MEGABYTE);
    statuses.push({
      testId: "embedding-model-download",
      tone: "progress",
      short: t("status.downloading", { downloaded, total }),
      full: `${t("embedding.model.downloading", { downloaded, total })}\n${t("embedding.model.note")}`,
    });
  }

  if (reranking?.tone === "progress") statuses.push(reranking);

  const rebuild = embedding?.rebuild;
  if (rebuild) {
    const title = t("embeddingProviders.rebuild.title", {
      done: rebuild.done,
      total: rebuild.total,
    });
    const reason =
      rebuild.reason === "local-mode" ? `${t("embeddingProviders.rebuild.localMode")}\n` : "";
    statuses.push({
      testId: "embedding-rebuild-notice",
      tone: "progress",
      short: t("status.rebuilding", { done: rebuild.done, total: rebuild.total }),
      full: `${reason}${title}\n${t("embeddingProviders.rebuild.note")}`,
    });
  }

  if (counts.processing > 0) {
    const text =
      counts.processing === 1
        ? t("status.processing.one")
        : t("status.processing.other", { count: counts.processing });
    statuses.push({ testId: "processing-status", tone: "progress", short: text, full: text });
  }

  if (!organizing && counts.tagging > 0) {
    const text =
      counts.tagging === 1
        ? t("status.tagging.one")
        : t("status.tagging.other", { count: counts.tagging });
    statuses.push({ testId: "tagging-status", tone: "progress", short: text, full: text });
  }

  if (organizing && library) {
    const pending = library.assignments.filter(
      (item) => item.status === "pending" || item.status === "classifying",
    ).length;
    const attention = library.assignments.filter(
      (item) => item.status === "waiting" || item.status === "failed",
    ).length;
    if (attention)
      statuses.push({
        testId: "organization-attention",
        tone: "waiting",
        short: t("library.needsAttention"),
        full: t("library.waiting", { count: attention }),
        action: { label: t("library.review"), run: () => useAppStore.getState().openLibrary() },
      });
    if (pending)
      statuses.push({
        testId: "organization-progress",
        tone: "progress",
        short: t("library.progress", { count: pending }),
        full: t("library.progress", { count: pending }),
      });
  }
  return statuses;
}
