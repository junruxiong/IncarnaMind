import { useState } from "react";
import type { ConversationAnswer } from "../../../../core/api";
import { checkSentences } from "../../checks";
import { shownClaims } from "../../editor/claimChecks";
import { mindEditor, whenMindEditor } from "../../editor/openEditors";
import { editorDoc } from "../../editor/proposalReview";
import { useT } from "../../i18n";
import { useAppStore } from "../../store";
import { ProblemLine } from "../ProblemLine";

const OPEN_WAIT_MS = 10_000;

async function checkLeft(mindId: string, request: string): Promise<void> {
  useAppStore.getState().openMind(mindId);
  const editor = mindEditor(mindId) ?? (await whenMindEditor(mindId, OPEN_WAIT_MS));
  const doc = editor && editorDoc(editor);
  if (!editor || !doc) return;
  const ranges = shownClaims(editor.state)
    .filter((claim) => claim.check.request === request && claim.shown.state === "not-checked")
    .map((claim) => ({ from: claim.sentence.from, to: claim.sentence.to }));
  if (ranges.length > 0) {
    await checkSentences(editor, mindId, doc, "check-against-documents", { ranges });
  }
}

/**
 * Under an Answer that wrote sentences with Citations into a Mind (#1111):
 * how many its check couldn't check, with Check again, and how many it
 * checked when there were more than one write checks, with Check the rest.
 * Either checks those left in their Mind now.
 */
export function SelfCheckLines({ answer }: { answer: Pick<ConversationAnswer, "toolCalls"> }) {
  const t = useT();
  const [asked, setAsked] = useState<ReadonlySet<string>>(new Set());
  const checks = answer.toolCalls.flatMap((call) => {
    const change = call.mindChange;
    return change?.selfCheck ? [{ mindId: change.mindId, ...change.selfCheck }] : [];
  });
  return checks.map(({ mindId, request, cited, sent, failed }) => {
    if (asked.has(request)) return null;
    const check = () => {
      setAsked((before) => new Set([...before, request]));
      checkLeft(mindId, request).catch(useAppStore.getState().reportError);
    };
    return (
      <div key={request}>
        {failed > 0 && (
          <ProblemLine
            role="status"
            testId="self-check-failed"
            action={{ label: t("claim.checkAgain"), testId: "self-check-again", onClick: check }}
          >
            {failed === 1 ? t("selfCheck.failed.one") : t("selfCheck.failed", { count: failed })}
          </ProblemLine>
        )}
        {sent < cited && (
          <ProblemLine
            role="status"
            testId="self-check-capped"
            action={{ label: t("selfCheck.checkRest"), testId: "self-check-rest", onClick: check }}
          >
            {t("selfCheck.capped", { checked: sent, count: cited })}
          </ProblemLine>
        )}
      </div>
    );
  });
}
