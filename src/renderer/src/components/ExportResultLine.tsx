import { type RefObject, useEffect, useRef, useState } from "react";
import { core, files } from "../core";
import { CheckStateIcon } from "../editor/icons";
import { errorMessage } from "../errors";
import {
  type ExportResult,
  exportMind,
  FORMAT_LABELS,
  revealUnverified,
  useExports,
} from "../exports";
import { useT } from "../i18n";
import { enter } from "../motion";
import { chooseOutputLocation } from "../outputFiles";
import { saveFailureReason } from "../problems";
import { CloseLineIcon, ExportLineIcon } from "./lineIcons";
import { type ProblemAction, ProblemLine } from "./ProblemLine";

const isMac = navigator.userAgent.includes("Mac");

/** How long a plain "Saved …" stays, unless the pointer or the focus is on it. */
const SAVED_SHOWN_MS = 10_000;

/**
 * How the last export of this Mind went, floating at the foot of its tab (#100), as problems
 * do (DESIGN.md, Problems): "Exporting…", then "Saved Tides.pdf · Open · Show in Finder", with
 * a second line when quotes weren't found ("1 quote not found, marked unverified · Show"), or
 * the problem and its one fix. A plain "Saved" goes by itself after a while; a problem stays
 * until it is dismissed, or the next export replaces it.
 */
export function ExportResultLine({ mindId }: { mindId: string }) {
  const result = useExports((state) => (state.result?.mindId === mindId ? state.result : null));
  const line = useRef<HTMLDivElement>(null);
  const shown = result !== null;
  useEffect(() => {
    if (shown && line.current) enter(line.current, { y: 8 });
  }, [shown]);
  if (!result) return null;
  return (
    <div className="pointer-events-none absolute inset-x-6 bottom-6 z-10 flex justify-center">
      <div
        ref={line}
        data-testid="export-result"
        data-state={result.state}
        className="pointer-events-auto flex max-w-full min-w-0 items-start gap-2 rounded-lg bg-sheet py-2 pr-2 pl-3 font-sans shadow-popover"
      >
        <ResultBody result={result} line={line} />
      </div>
    </div>
  );
}

/** Whether the pointer or the focus is on the line, which keeps a plain "Saved" showing. */
function useHeld(line: RefObject<HTMLDivElement | null>): boolean {
  const [held, setHeld] = useState(false);
  useEffect(() => {
    const element = line.current;
    if (!element) return;
    const hold = () => setHeld(true);
    const release = (event: Event) => {
      if (
        event.type === "focusout" &&
        element.contains((event as FocusEvent).relatedTarget as Node)
      )
        return;
      setHeld(element.matches(":hover"));
    };
    element.addEventListener("pointerenter", hold);
    element.addEventListener("pointerleave", release);
    element.addEventListener("focusin", hold);
    element.addEventListener("focusout", release);
    return () => {
      element.removeEventListener("pointerenter", hold);
      element.removeEventListener("pointerleave", release);
      element.removeEventListener("focusin", hold);
      element.removeEventListener("focusout", release);
    };
  }, [line]);
  return held;
}

function ResultBody({
  result,
  line,
}: {
  result: ExportResult;
  line: RefObject<HTMLDivElement | null>;
}) {
  const t = useT();
  const dismiss = useExports((state) => state.dismiss);
  const [error, setError] = useState<string | null>(null);
  const held = useHeld(line);
  const plain = result.state === "saved" && result.unverified === 0 && error === null;

  useEffect(() => {
    if (!plain || held) return;
    const timer = setTimeout(dismiss, SAVED_SHOWN_MS);
    return () => clearTimeout(timer);
  }, [plain, held, dismiss]);

  const format = t(FORMAT_LABELS[result.format]);
  const fix = (): ProblemAction | undefined => {
    if (result.state === "failed") {
      return {
        label: t("export.changeLocation"),
        testId: "export-change-location",
        onClick: () =>
          void exportMind(
            result.mindId,
            result.format,
            async () => (await chooseOutputLocation()) !== null,
          ),
      };
    }
    if (result.state === "not-allowed") {
      const { linkedFolderId } = result;
      return {
        label: t("export.allow"),
        testId: "export-allow",
        onClick: () =>
          void exportMind(result.mindId, result.format, async () => {
            await core.setLinkedFolderWriting(linkedFolderId, true);
            return true;
          }),
      };
    }
    return undefined;
  };

  const run = (act: () => Promise<void>, failed: "export.failed.open" | "export.failed.show") => {
    setError(null);
    act().catch((failure: unknown) =>
      setError(t(failed, { reason: reasonOf(errorMessage(failure), t) })),
    );
  };

  return (
    <>
      <div className="flex min-w-0 flex-col gap-0.5">
        {result.state === "saving" ? (
          <p
            role="status"
            data-testid="export-saving"
            className="flex items-center gap-2 py-1 text-[13px] leading-5 text-ink-meta"
          >
            <ExportLineIcon className="size-4 shrink-0" />
            {t("export.saving.line", { format })}
          </p>
        ) : result.state === "saved" ? (
          <>
            <div className="flex min-w-0 flex-wrap items-center gap-x-4 gap-y-1">
              <p
                role="status"
                data-testid="export-done"
                title={result.path}
                className="flex min-w-0 items-center gap-2 text-[14px] leading-5 text-ink-secondary"
              >
                <CheckIcon />
                <SavedName name={result.name} />
              </p>
              <span className="flex shrink-0 gap-1">
                <button
                  type="button"
                  data-testid="export-open"
                  onClick={() => run(() => files.openSavedFile(result.path), "export.failed.open")}
                  className="h-7 rounded-md bg-sheet px-3 text-[13px] leading-5 font-semibold text-ink shadow-[inset_0_0_0_1px_var(--color-rule-strong)] hover:bg-hover"
                >
                  {t("export.open")}
                </button>
                <button
                  type="button"
                  data-testid="export-show"
                  onClick={() => run(() => files.showSavedFile(result.path), "export.failed.show")}
                  className="h-7 rounded-md px-2.5 text-[13px] leading-5 font-semibold text-ink-secondary hover:bg-hover hover:text-ink"
                >
                  {t(isMac ? "export.showInFinder" : "export.showInFolder")}
                </button>
              </span>
            </div>
            {result.unverified > 0 && (
              <ProblemLine
                role="status"
                testId="export-unverified"
                action={{
                  label: t("export.unverified.show"),
                  testId: "export-unverified-show",
                  onClick: () => revealUnverified(result.mindId),
                }}
              >
                {/* Not found is a Citation check: its icon, in its colour. */}
                <CheckStateIcon
                  check="not-found"
                  className="mr-2 inline size-4 align-[-3px] text-warning"
                />
                {result.unverified === 1
                  ? t("export.unverified.line.one")
                  : t("export.unverified.line.other", { count: result.unverified })}
              </ProblemLine>
            )}
            {error && <ProblemLine testId="export-problem">{error}</ProblemLine>}
          </>
        ) : (
          <ProblemLine testId="export-problem" action={fix()} className="py-1">
            {t("export.failed.save", {
              reason:
                result.state === "not-allowed"
                  ? t("export.failed.notAllowed", { folder: result.folder })
                  : reasonOf(result.message, t),
            })}
          </ProblemLine>
        )}
      </div>
      {result.state !== "saving" && (
        <button
          type="button"
          data-testid="export-dismiss"
          aria-label={t("export.dismiss")}
          title={t("export.dismiss")}
          onClick={dismiss}
          className="inline-flex size-7 shrink-0 items-center justify-center rounded-md text-ink-meta hover:bg-hover hover:text-ink"
        >
          <CloseLineIcon className="size-3.5" />
        </button>
      )}
    </>
  );
}

/** "Saved Tides.pdf", the file's name in the ink, as the design shows it. */
function SavedName({ name }: { name: string }) {
  const t = useT();
  const [before = "", after = ""] = t("export.done", { name: "\u0000" }).split("\u0000");
  return (
    <span className="min-w-0 truncate">
      {before}
      <span className="font-semibold text-ink">{name}</span>
      {after}
    </span>
  );
}

function CheckIcon() {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={2}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      className="size-4 shrink-0 text-ink"
    >
      <path d="m5 12.5 4.5 4.5L19 7.5" />
    </svg>
  );
}

/** Why a file couldn't be written or opened, in a few words: ours when we know, else the system's. */
function reasonOf(message: string, t: ReturnType<typeof useT>): string {
  const key = saveFailureReason(message);
  return key ? t(key) : message;
}
