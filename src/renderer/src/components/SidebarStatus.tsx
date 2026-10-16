import { type ReactNode, type ToggleEvent, useId, useRef, useState } from "react";
import { useShallow } from "zustand/react/shallow";
import type { DocumentStatus, EmbeddingModelError } from "../../../core/api";
import type { MessageKey } from "../../../shared/i18n";
import { organizes } from "../../../shared/libraryModels";
import { core } from "../core";
import { useT } from "../i18n";
import { useCapturing } from "../meetings";
import { downloadProgress } from "../modelDownload";
import { enterFrom, exitToward } from "../motion";
import { rerankingModelStatus } from "../rerankStatus";
import { allExamples, isOwnDocument, useAppStore } from "../store";
import { textRecognitionStatus, useTextRecognition } from "../textRecognition";
import { canTryAgain, outsideDialog, reasonKey, useWebPages } from "../webPages";
import { CloseLineIcon } from "./lineIcons";
import { embeddingProviderLabel } from "./providers/EmbeddingSettings";
import { testErrorKey } from "./providers/shared";
import { rowActionButtonClass } from "./sidebarRows";
import { useTaggingPermission } from "./useTaggingPermission";

/** The status row's button: a small ghost one, inside the 28px row. */
const statusActionClass =
  "inline-flex h-6 shrink-0 items-center rounded-md px-2 text-[12px] font-semibold text-ink-secondary hover:bg-rule hover:text-ink focus-visible:outline-offset-0 aria-expanded:bg-rule aria-expanded:text-ink";

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

/** In progress (a blue dot), waiting for the User or failed (amber), or for information (grey). */
type Tone = "progress" | "waiting" | "error" | "info";

const dotTones: Record<Tone, string> = {
  progress: "bg-accent",
  waiting: "bg-attention",
  error: "bg-attention",
  info: "bg-ink-placeholder",
};

interface Status {
  testId: string;
  tone: Tone;
  /** What the row shows: short enough for one line. It gives way to `figure`. */
  short: string;
  /** How far along it is ("3/22 MB"): shown after `short`, never cut. */
  figure?: string;
  /** All of it: the tooltip, and what a screen reader reads. */
  full: string;
  action?: { label: string; testId?: string; run(): void };
  dismiss?: () => void;
}

/**
 * The sidebar footer's status row, above Settings: what the app is doing
 * with the Documents (processing, tagging, fetching web pages, the search
 * model's download or a rebuild) or what needs the User (tagging waits for a
 * model, a failure to retry, a web page that couldn't be added, skipped
 * files, files that were in IncarnaMind already). One line:
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
  const { tone, short, figure, full, action, dismiss } = status;
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
      <span aria-hidden="true" className="flex min-w-0 flex-1 items-center gap-1">
        <span data-testid="status-label" className="min-w-0 truncate">
          {figure ? `${short} ·` : short}
        </span>
        {figure && (
          <span data-testid="status-figure" className="shrink-0 tabular-nums">
            {figure}
          </span>
        )}
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
          const list = event.currentTarget;
          if (event.newState === "open") {
            place(list);
            // Once it shows, before that frame paints: it grows from the button.
            requestAnimationFrame(() => enterFrom(list, button.current));
          } else exitToward(list, button.current);
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
  const organizing = organizes(library);
  const capturing = useCapturing();
  const counts = useAppStore(
    useShallow((state) => {
      let processing = 0;
      // Scanned pages still to read, of the Documents being read (#397).
      let scannedPages = 0;
      let tagging = 0;
      // Only the User's own Documents: the examples need no model, so they alone don't ask for one.
      let waitingForTagger = 0;
      let waitingForPlan = 0;
      let failedTagging = 0;
      let unreached = 0;
      // A paused Linked folder's Documents wait for the User: its row says so, not the footer.
      const paused = new Set(
        state.linkedFolders.filter((each) => each.status === "paused").map((each) => each.id),
      );
      for (const item of state.documents) {
        if (PROCESSING.has(item.status)) {
          if (item.linkedFolderId === null || !paused.has(item.linkedFolderId)) {
            processing++;
            if (item.scan?.of === "scanned-pages") scannedPages += item.scan.pages - item.scan.read;
          }
        } else if (item.status === "ready") {
          if (item.tagging === "pending" || item.tagging === "tagging") tagging++;
          else if (
            item.tagging === "waiting-for-provider" &&
            isOwnDocument(item, allExamples(state))
          ) {
            waitingForTagger++;
            if (item.taggingError?.kind === "plan-limit") waitingForPlan++;
          } else if (item.tagging === "failed" && isOwnDocument(item, allExamples(state))) {
            failedTagging++;
            if (item.taggingError?.kind === "network") unreached++;
          }
        }
      }
      return {
        processing,
        scannedPages,
        tagging,
        waitingForTagger,
        waitingForPlan,
        failedTagging,
        unreached,
      };
    }),
  );
  const retryDownload = useAppStore((state) => state.downloadEmbeddingModel);
  const rerank = useAppStore((state) => state.rerank);
  const retryRerankingDownload = useAppStore((state) => state.downloadRerankingModel);
  const textRecognition = textRecognitionStatus(
    useTextRecognition((state) => state.status),
    t,
  );
  const retryTextRecognition = useTextRecognition((state) => state.download);
  const retryEmbedding = useAppStore((state) => state.retryEmbedding);
  const dismissSkipped = useAppStore((state) => state.dismissSkippedFiles);
  const dismissAlreadyAdded = useAppStore((state) => state.dismissAlreadyAdded);
  const openSettings = useAppStore((state) => state.openSettings);
  const taggingWaits = !organizing && counts.waitingForTagger > 0;
  const taggingNeedsOk = useTaggingPermission(taggingWaits);
  // The dialog shows how the pages typed in it go; the rest are said here.
  const fetchingPages = useWebPages(
    (state) => state.fetching.filter((key) => outsideDialog(state, key)).length,
  );
  const pageProblems = useWebPages(
    useShallow((state) => state.problems.filter((each) => outsideDialog(state, each.key))),
  );
  const statuses: Status[] = [];
  // The built-in reranking model, on by default, downloads once there are Documents to search.
  const rerankingStatus = rerankingModelStatus(rerank, t);
  const reranking: Status | null = rerankingStatus && {
    testId: rerankingStatus.testId,
    tone: rerankingStatus.tone,
    short: rerankingStatus.short,
    figure: rerankingStatus.figure,
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

  if (textRecognition?.tone === "error") {
    statuses.push({
      ...textRecognition,
      action: {
        label: t("status.retry"),
        testId: "text-recognition-retry",
        run: () => void retryTextRecognition(),
      },
    });
  }

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

  if (!organizing && counts.failedTagging > 0) {
    const failed = useAppStore
      .getState()
      .documents.filter((each) => each.status === "ready" && each.tagging === "failed");
    statuses.push({
      testId: "tagging-failed",
      tone: "error",
      short:
        counts.failedTagging === 1
          ? t("status.taggingFailed.one")
          : t("status.taggingFailed.other", { count: counts.failedTagging }),
      full: [
        t(
          counts.unreached === counts.failedTagging
            ? "status.taggingFailed.offline"
            : "status.taggingFailed.full",
        ),
        failed[0]?.taggingError?.message ?? "",
      ]
        .filter(Boolean)
        .join("\n"),
      action: {
        label: t("status.retry"),
        testId: "tagging-failed-retry",
        run: () => void useAppStore.getState().retagDocuments(failed.map((each) => each.id)),
      },
    });
  }

  for (const problem of pageProblems) {
    const refused = !canTryAgain(problem);
    const reason = t(reasonKey(problem.reason));
    const short = problem.document
      ? t("webPages.fetchAgainFailed", { name: problem.document.name, reason })
      : t(refused ? "webPages.notAdded" : "webPages.couldNotFetch", { reason });
    const { tryAgain, dismiss } = useWebPages.getState();
    statuses.push({
      testId: `web-page-problem:${problem.key}`,
      tone: refused ? "waiting" : "error",
      short,
      full: `${short}\n${problem.address}\n${problem.message}`,
      ...(refused
        ? {}
        : {
            action: {
              label: t("webPages.tryAgain"),
              testId: "web-page-try-again",
              run: () => void tryAgain(problem),
            },
          }),
      dismiss: () => dismiss(problem),
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

  if (counts.waitingForPlan > 0) {
    statuses.push({
      testId: "tagging-paused",
      tone: "waiting",
      short: t("status.taggingPaused"),
      full: t("status.taggingPaused.full"),
    });
  } else if (taggingWaits && taggingNeedsOk) {
    statuses.push({
      testId: "tagging-waiting",
      tone: "waiting",
      short: t("status.taggingPermission"),
      full: t("jev.waiting.permission", { service: taggingNeedsOk.name }),
      action: {
        label: t("privacy.flows.allow"),
        testId: "tagging-waiting-allow",
        run: () =>
          void core
            .allowDataFlow("tagging", taggingNeedsOk.id)
            .catch((error: unknown) => useAppStore.getState().reportError(error)),
      },
    });
  } else if (taggingWaits) {
    statuses.push({
      testId: "tagging-waiting",
      tone: "waiting",
      short: t("status.taggingWaiting"),
      full: t("jev.waiting.notice"),
      action: {
        label: t("status.choose"),
        testId: "tagging-waiting-setup",
        run: () => openSettings("models"),
      },
    });
  }

  if (model?.state === "downloading") {
    const { downloaded, total, figure } = downloadProgress(model, t);
    statuses.push({
      testId: "embedding-model-download",
      tone: "progress",
      short: t("status.downloading"),
      figure,
      full: `${t("embedding.model.downloading", { downloaded, total })}\n${t("embedding.model.note")}`,
    });
  }

  if (reranking?.tone === "progress") statuses.push(reranking);
  if (textRecognition?.tone === "progress") statuses.push(textRecognition);

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
      short: t("status.rebuilding"),
      figure: `${rebuild.done}/${rebuild.total}`,
      full: `${reason}${title}\n${t("embeddingProviders.rebuild.note")}`,
    });
  }

  if (fetchingPages > 0) {
    const text =
      fetchingPages === 1
        ? t("webPages.fetching.one")
        : t("webPages.fetching.other", { count: fetchingPages });
    statuses.push({ testId: "web-pages-fetching", tone: "progress", short: text, full: text });
  }

  // While a Meeting records, Documents wait to be read, and tagging waits too (#135).
  if (capturing && (counts.processing > 0 || counts.tagging > 0)) {
    statuses.push({
      testId: "reading-paused",
      tone: "info",
      short: t("status.readingPaused"),
      full: t("status.readingPaused.full"),
    });
  } else if (counts.processing > 0) {
    // While scanned pages are read, they are what takes the time: the row says so.
    const text =
      counts.scannedPages > 0
        ? counts.scannedPages === 1
          ? t("status.scanning.one")
          : t("status.scanning.other", { count: counts.scannedPages })
        : counts.processing === 1
          ? t("status.processing.one")
          : t("status.processing.other", { count: counts.processing });
    statuses.push({ testId: "processing-status", tone: "progress", short: text, full: text });
  }

  if (!capturing && !organizing && counts.tagging > 0) {
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
