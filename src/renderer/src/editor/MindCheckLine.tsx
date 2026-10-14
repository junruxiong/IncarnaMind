import type { Editor } from "@tiptap/core";
import { useEditorState } from "@tiptap/react";
import type { MindCheckRun } from "../../../core/api";
import { ProblemLine } from "../components/ProblemLine";
import { runStopText } from "../cost";
import { dayAndTime } from "../dates";
import { useLanguage, useT } from "../i18n";
import { checkMind, dismissMindCheck, startMindCheck, useMindCheck } from "../mindCheck";
import { shownClaims } from "./claimChecks";

/**
 * Check this Mind's question (#737), on the line under the title before a
 * run billed per token: "Check this Mind's 48 sentences against this
 * Folder's Documents? About 52 calls, … · Check · Not now".
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
        estimate: asking.estimate,
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
  const when = dayAndTime(new Date(run.at), new Date(), language);
  const count = new Intl.NumberFormat(language).format(run.sentences);
  const said = t(run.sentences === 1 ? "health.checkedAll.one" : "health.checkedAll", {
    count,
    when,
  });
  return (
    <span data-testid="health-checked-all" className="mind-health-when">
      {run.stop ? `${said} · ${runStopText(run.stop, t, language)}` : said}
    </span>
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
      className="mind-verify-link"
      onMouseDown={(event) => event.preventDefault()}
      onClick={onCheck}
    >
      {t(unchecked > 0 ? "health.checkAll" : "claim.checkAgain")}
    </button>
  );
}

/**
 * After a whole-Mind check whose claims mostly found nothing in the Mind's
 * Folder: "Check against all Documents", as #764 offers for a selection,
 * so a Folder that holds too little isn't the end of it.
 */
export function MindCheckWiden({ editor, mindId }: { editor: Editor; mindId: string }) {
  const t = useT();
  const widen = useEditorState({
    editor,
    selector: ({ editor: current }) => {
      let verdicts = 0;
      let lacking = 0;
      let inFolder = false;
      for (const { shown, check } of shownClaims(current.state)) {
        if (shown.state === "supported" || shown.state === "contradicted") verdicts++;
        else if (shown.state === "unsupported") {
          verdicts++;
          lacking++;
          if (check.scope.folderId !== null && !check.scope.widened) inFolder = true;
        }
      }
      return inFolder && verdicts >= 2 && lacking * 2 > verdicts;
    },
  });
  if (!widen) return null;
  return (
    <>
      <span className="mind-health-dot" aria-hidden="true">
        ·
      </span>
      <button
        type="button"
        data-testid="health-check-all-documents"
        className="mind-verify-link"
        onMouseDown={(event) => event.preventDefault()}
        onClick={() => void startMindCheck(mindId, "all", { allDocuments: true })}
      >
        {t("mindCheck.widen")}
      </button>
    </>
  );
}
