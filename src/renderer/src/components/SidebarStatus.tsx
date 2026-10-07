import { useShallow } from "zustand/react/shallow";
import type { DocumentStatus, EmbeddingModelError } from "../../../core/api";
import type { MessageKey } from "../../../shared/i18n";
import { useT } from "../i18n";
import { useAppStore } from "../store";
import { CloseLineIcon } from "./lineIcons";
import { embeddingProviderLabel } from "./providers/EmbeddingSettings";
import { testErrorKey } from "./providers/shared";
import { rowActionButtonClass } from "./sidebarRows";

/** The status row's button: a small ghost one, inside the 28px row. */
const statusActionClass =
  "inline-flex h-6 shrink-0 items-center rounded-md px-2 text-[12px] font-semibold text-ink-secondary outline-none hover:bg-rule hover:text-ink focus-visible:outline-2 focus-visible:outline-accent";

/** Decimal megabytes, as the model's 135 MB is quoted elsewhere. */
const MEGABYTE = 1_000_000;

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

/** In progress (a blue dot), waiting for the User (amber), or failed (red). */
type Tone = "progress" | "waiting" | "error";

const dotTones: Record<Tone, string> = {
  progress: "bg-accent",
  waiting: "bg-attention",
  error: "bg-danger",
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
 * retry, skipped files). One line, the most pressing first. Its 28px are
 * kept even when there is nothing to say, so the tree above never moves.
 */
export function SidebarStatus() {
  const status = useStatus();
  return (
    <div data-testid="sidebar-status" className="flex h-7 shrink-0 flex-col">
      {status && <StatusRow status={status} />}
    </div>
  );
}

function StatusRow({ status }: { status: Status }) {
  const t = useT();
  const { tone, short, full, action, dismiss } = status;
  return (
    <div
      role={tone === "error" ? "alert" : "status"}
      data-testid={status.testId}
      title={full}
      className="flex h-7 items-center gap-2 pr-1 pl-2 text-[13px] leading-5 text-ink-meta"
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

/** The most pressing status, or null when there is nothing to say. */
function useStatus(): Status | null {
  const t = useT();
  const skipped = useAppStore((state) => state.skippedFiles);
  const model = useAppStore((state) => state.embeddingModel);
  const embedding = useAppStore((state) => state.embedding);
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
  const retryEmbedding = useAppStore((state) => state.retryEmbedding);
  const dismissSkipped = useAppStore((state) => state.dismissSkippedFiles);
  const openSettings = useAppStore((state) => state.openSettings);

  if (model?.state === "failed" && model.error) {
    const { kind, message } = model.error;
    const full =
      kind === "load"
        ? t("embedding.model.loadFailed")
        : t("embedding.model.failed", { reason: t(modelFailures[kind]) });
    return {
      testId: "embedding-model-failed",
      tone: "error",
      short: t(kind === "load" ? "status.modelFailed" : "status.downloadFailed"),
      full: `${full}\n${message}`,
      action: {
        label: t("status.retry"),
        testId: "embedding-model-retry",
        run: () => void retryDownload(),
      },
    };
  }

  if (embedding?.error) {
    const provider = embeddingProviderLabel(embedding.provider, t);
    return {
      testId: "embedding-provider-error",
      tone: "error",
      short: t("status.searchError", { provider }),
      full: `${t("embeddingProviders.error.notice", {
        provider,
        reason: t(testErrorKey(embedding.error.kind)),
      })}\n${embedding.error.message}`,
      action: { label: t("status.retry"), run: () => void retryEmbedding() },
    };
  }

  if (skipped.length > 0) {
    return {
      testId: "skipped-files",
      tone: "waiting",
      short:
        skipped.length === 1
          ? t("status.skipped.one")
          : t("status.skipped.other", { count: skipped.length }),
      full: t("documents.skipped", { names: skipped.join(", ") }),
      dismiss: dismissSkipped,
    };
  }

  if (counts.waitingForTagger > 0) {
    return {
      testId: "tagging-waiting",
      tone: "waiting",
      short: t("status.taggingWaiting"),
      full: t("jev.waiting.notice"),
      action: {
        label: t("status.setUp"),
        testId: "tagging-waiting-setup",
        run: () => openSettings("chat-model"),
      },
    };
  }

  if (model?.state === "downloading") {
    const downloaded = Math.floor(model.downloadedBytes / MEGABYTE);
    const total = Math.ceil(model.totalBytes / MEGABYTE);
    return {
      testId: "embedding-model-download",
      tone: "progress",
      short: t("status.downloading", { downloaded, total }),
      full: `${t("embedding.model.downloading", { downloaded, total })}\n${t("embedding.model.note")}`,
    };
  }

  const rebuild = embedding?.rebuild;
  if (rebuild) {
    const title = t("embeddingProviders.rebuild.title", {
      done: rebuild.done,
      total: rebuild.total,
    });
    const reason =
      rebuild.reason === "local-mode" ? `${t("embeddingProviders.rebuild.localMode")}\n` : "";
    return {
      testId: "embedding-rebuild-notice",
      tone: "progress",
      short: t("status.rebuilding", { done: rebuild.done, total: rebuild.total }),
      full: `${reason}${title}\n${t("embeddingProviders.rebuild.note")}`,
    };
  }

  if (counts.processing > 0) {
    const text =
      counts.processing === 1
        ? t("status.processing.one")
        : t("status.processing.other", { count: counts.processing });
    return { testId: "processing-status", tone: "progress", short: text, full: text };
  }

  if (counts.tagging > 0) {
    const text =
      counts.tagging === 1
        ? t("status.tagging.one")
        : t("status.tagging.other", { count: counts.tagging });
    return { testId: "tagging-status", tone: "progress", short: text, full: text };
  }

  return null;
}
