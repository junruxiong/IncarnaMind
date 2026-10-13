import type * as Y from "yjs";
import { type ContradictionCheckOutcome, MIND_SETTINGS_FIELD } from "../../../core/api";
import { clearContradictionWatch } from "../../../shared/contradictionWatch";
import { ProblemLine } from "../components/ProblemLine";
import {
  checkContradictions,
  dismissContradictions,
  useContradictionLine,
} from "../contradictions";
import { formatMoney } from "../cost";
import { useLanguage, useT } from "../i18n";
import { contradictionOutcomeText } from "./contradictionOutcome";

/**
 * Contradiction watch's line under the Mind's title (#112), beside its
 * health: the question before a run billed per token, with the estimate;
 * what a run on demand found; and what the last run a new version of a
 * Document started found and cost. Each closes; the last for good.
 */
export function ContradictionLine({ doc, mindId }: { doc: Y.Doc; mindId: string }) {
  const t = useT();
  const language = useLanguage();
  const shown = useContradictionLine(doc, mindId);
  const format = (value: number) => new Intl.NumberFormat(language).format(value);
  const outcome = (run: ContradictionCheckOutcome) => contradictionOutcomeText(run, t, language);

  if (shown?.kind === "asking") {
    return (
      <ProblemLine
        role="status"
        testId="contradictions-ask"
        className="mind-health-line"
        action={{
          label: t("contradictions.check"),
          testId: "contradictions-check",
          onClick: () => void checkContradictions(mindId),
        }}
        also={{
          label: t("contradictions.cancel"),
          testId: "contradictions-cancel",
          onClick: () => dismissContradictions(mindId),
        }}
      >
        {t(shown.claims === 1 ? "contradictions.ask.one" : "contradictions.ask", {
          count: format(shown.claims),
          estimate: shown.estimate,
        })}
      </ProblemLine>
    );
  }
  if (shown?.kind === "found") {
    return (
      <ProblemLine
        role="status"
        testId="contradictions-found"
        className="mind-health-line"
        action={{
          label: t("checks.close"),
          testId: "contradictions-close",
          onClick: () => dismissContradictions(mindId),
        }}
      >
        {outcome(shown.run)}
      </ProblemLine>
    );
  }
  if (shown?.kind === "watched") {
    const watched = shown.run;
    const result = t("contradictions.after", {
      document: watched.documentName,
      result: outcome(watched),
    });
    return (
      <ProblemLine
        role="status"
        testId="contradictions-watched"
        className="mind-health-line"
        action={{
          label: t("checks.close"),
          testId: "contradictions-watched-close",
          // Closed for good: the record goes from the Mind, everywhere it is open.
          onClick: () => clearContradictionWatch(doc.getMap(MIND_SETTINGS_FIELD)),
        }}
      >
        {watched.cost
          ? t("contradictions.cost", { result, cost: formatMoney(watched.cost, language) })
          : result}
      </ProblemLine>
    );
  }
  return null;
}
