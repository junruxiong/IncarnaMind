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
import { contradictionOutcomeParts } from "./contradictionOutcome";

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
  const outcome = (run: ContradictionCheckOutcome) => contradictionOutcomeParts(run, t, language);

  if (shown?.kind === "asking") {
    return (
      <ProblemLine
        role="status"
        testId="contradictions-ask"
        className="mind-health-line"
        details={shown.estimate.parts}
        title={shown.estimate.title}
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
        })}
      </ProblemLine>
    );
  }
  if (shown?.kind === "found") {
    const [first = "", ...rest] = outcome(shown.run);
    return (
      <ProblemLine
        role="status"
        testId="contradictions-found"
        className="mind-health-line"
        details={rest}
        action={{
          label: t("checks.close"),
          testId: "contradictions-close",
          onClick: () => dismissContradictions(mindId),
        }}
      >
        {first}
      </ProblemLine>
    );
  }
  if (shown?.kind === "watched") {
    const watched = shown.run;
    const [first = "", ...rest] = outcome(watched);
    const cost = watched.cost ? [formatMoney(watched.cost, language)] : [];
    return (
      <ProblemLine
        role="status"
        testId="contradictions-watched"
        className="mind-health-line"
        details={[...rest, ...cost]}
        action={{
          label: t("checks.close"),
          testId: "contradictions-watched-close",
          // Closed for good: the record goes from the Mind, everywhere it is open.
          onClick: () => clearContradictionWatch(doc.getMap(MIND_SETTINGS_FIELD)),
        }}
      >
        {t("contradictions.after", { document: watched.documentName, result: first })}
      </ProblemLine>
    );
  }
  return null;
}
