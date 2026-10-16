import type { Editor } from "@tiptap/core";
import { useEditorState } from "@tiptap/react";
import type { MindCheckRun } from "../../../core/api";
import { formatDateTime } from "../../../shared/dates";
import { buttonStyle } from "../components/ui";
import { runStopText } from "../cost";
import { useLanguage, useT } from "../i18n";
import { checkMind, dismissMindCheck, useMindCheck } from "../mindCheck";
import { shownClaims } from "./claimChecks";

/**
 * A question before a run billed per token, in the check badge's card: what
 * it would check, the estimate under it, then Check and Not now.
 */
export function AskBlock({
  testId,
  question,
  estimate,
  check,
  notNow,
}: {
  testId: string;
  question: string;
  estimate: string;
  check: { label: string; testId: string; onClick(): void };
  notNow: { label: string; testId: string; onClick(): void };
}) {
  return (
    <div role="status" data-testid={testId} className="check-card-ask">
      <p className="check-card-ask-title">{question}</p>
      {estimate && <p>{estimate}</p>}
      <div className="check-card-ask-actions">
        <button
          type="button"
          data-testid={check.testId}
          className={buttonStyle("primary", "sm")}
          onClick={check.onClick}
        >
          {check.label}
        </button>
        <button
          type="button"
          data-testid={notNow.testId}
          className={buttonStyle("ghost", "sm")}
          onClick={notNow.onClick}
        >
          {notNow.label}
        </button>
      </div>
    </div>
  );
}

/** Check this Mind's question (#737): "Check this Mind's 48 sentences against this Folder's Documents?" */
export function MindCheckAsk({ mindId }: { mindId: string }) {
  const t = useT();
  const language = useLanguage();
  const asking = useMindCheck((state) => state.asking[mindId]);
  if (!asking) return null;
  const where = asking.inFolder ? "folder" : "all";
  const count = new Intl.NumberFormat(language).format(asking.sentences);
  return (
    <AskBlock
      testId="mind-check-ask"
      question={t(
        asking.sentences === 1 ? `mindCheck.ask.${where}.one` : `mindCheck.ask.${where}`,
        { count },
      )}
      estimate={asking.estimate}
      check={{
        label: t("mindCheck.check"),
        testId: "mind-check-check",
        onClick: () => void checkMind(mindId, asking.mode, { allDocuments: asking.allDocuments }),
      }}
      notNow={{
        label: t("mindCheck.cancel"),
        testId: "mind-check-cancel",
        onClick: () => dismissMindCheck(mindId),
      }}
    />
  );
}

/** How many of the claims a Check this Mind run marked it has checked. */
export function useMindCheckProgress(editor: Editor, requestId: string | null) {
  return useEditorState({
    editor,
    selector: ({ editor: current }) => {
      const mine =
        requestId === null
          ? []
          : shownClaims(current.state).filter((each) => each.check.request === requestId);
      return {
        total: mine.length,
        done: mine.filter((each) => each.shown.state !== "checking").length,
      };
    },
    equalityFn: (a, b) => b !== null && a.done === b.done && a.total === b.total,
  });
}

/** How far the run is: "Finding the claims to check…", then "Checking 12 of 48…". */
export function MindCheckProgress({ editor, requestId }: { editor: Editor; requestId: string }) {
  const t = useT();
  const language = useLanguage();
  const { done, total } = useMindCheckProgress(editor, requestId);
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
    <span data-testid="health-checked-all">
      {run.stop ? `${said} · ${runStopText(run.stop, t, language)}` : said}
    </span>
  );
}

/**
 * The check card's way to check what changed: "Check all" while some
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
