import type { Editor } from "@tiptap/core";
import { useEditorState } from "@tiptap/react";
import type { MindCheckRun } from "../../../core/api";
import { formatDateTime } from "../../../shared/dates";
import { ProblemLine } from "../components/ProblemLine";
import { runStopParts } from "../cost";
import { useLanguage, useT } from "../i18n";
import { checkMind, dismissMindCheck, useMindCheck } from "../mindCheck";
import { shownClaims } from "./claimChecks";

/**
 * Check this Mind's question (#737), on the line under the title before a
 * run billed per token: "Check this Mind's 48 sentences against this
 * Folder's Documents? · About $0.40 · 52 model calls · Check · Not now".
 */
export function MindCheckAsk({ mindId }: { mindId: string }) {
  const t = useT();
  const language = useLanguage();
  const asking = useMindCheck((state) => state.asking[mindId]);
  if (!asking) return null;
  const where = asking.inFolder ? "folder" : "all";
  const count = new Intl.NumberFormat(language).format(asking.sentences);
  return (
    <ProblemLine
      role="status"
      testId="mind-check-ask"
      className="mind-health-line"
      details={asking.estimate.parts}
      title={asking.estimate.title}
      action={{
        label: t("mindCheck.check"),
        testId: "mind-check-check",
        onClick: () => void checkMind(mindId, asking.mode, { allDocuments: asking.allDocuments }),
      }}
      also={{
        label: t("mindCheck.cancel"),
        testId: "mind-check-cancel",
        onClick: () => dismissMindCheck(mindId),
      }}
    >
      {t(asking.sentences === 1 ? `mindCheck.ask.${where}.one` : `mindCheck.ask.${where}`, {
        count,
      })}
    </ProblemLine>
  );
}

/** How far the run is: "Finding the claims to check…", then "Checking 12 of 48…". */
export function MindCheckProgress({ editor, requestId }: { editor: Editor; requestId: string }) {
  const t = useT();
  const language = useLanguage();
  const { done, total } = useEditorState({
    editor,
    selector: ({ editor: current }) => {
      const mine = shownClaims(current.state).filter((each) => each.check.request === requestId);
      return {
        total: mine.length,
        done: mine.filter((each) => each.shown.state !== "checking").length,
      };
    },
    equalityFn: (a, b) => b !== null && a.done === b.done && a.total === b.total,
  });
  const format = (value: number) => new Intl.NumberFormat(language).format(value);
  return (
    <span data-testid="mind-check-progress" className="tabular-nums">
      {total === 0
        ? t("verify.finding")
        : t("verify.checking", { done: format(done), total: format(total) })}
    </span>
  );
}

/** When the whole Mind was last checked: "Checked 48 sentences, 2 Oct, 15:20", and what stopped it. */
export function CheckedWhen({ run }: { run: MindCheckRun }) {
  const t = useT();
  const language = useLanguage();
  const when = formatDateTime(run.at, language);
  const count = new Intl.NumberFormat(language).format(run.sentences);
  const said = t(run.sentences === 1 ? "health.checkedAll.one" : "health.checkedAll", {
    count,
    when,
  });
  return (
    <>
      <span data-testid="health-checked-all" className="mind-health-when">
        {said}
      </span>
      {run.stop &&
        runStopParts(run.stop, t, language).map((part) => (
          <span key={part} className="mind-health-when">
            {part}
          </span>
        ))}
    </>
  );
}

/**
 * The health line's way to check what changed: "Check all" while some
 * sentences were never checked, "Check again" while some checks need a re-check.
 */
export function MindCheckAction({
  unchecked,
  recheck,
  onCheck,
}: {
  unchecked: number;
  recheck: number;
  onCheck: () => void;
}) {
  const t = useT();
  if (unchecked === 0 && recheck === 0) return null;
  return (
    <button
      type="button"
      data-testid="health-check"
      className="mind-verify-link mind-health-action"
      onMouseDown={(event) => event.preventDefault()}
      onClick={onCheck}
    >
      {t(unchecked > 0 ? "health.checkAll" : "claim.checkAgain")}
    </button>
  );
}
