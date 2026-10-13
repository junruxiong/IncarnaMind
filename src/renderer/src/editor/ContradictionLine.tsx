import { useEffect, useState } from "react";
import type * as Y from "yjs";
import {
  type ContradictionCheckOutcome,
  type ContradictionWatchRun,
  MIND_SETTINGS_FIELD,
} from "../../../core/api";
import {
  clearContradictionWatch,
  contradictionWatchOf,
  observeContradictionWatch,
} from "../../../shared/contradictionWatch";
import { ProblemLine } from "../components/ProblemLine";
import { checkContradictions, dismissContradictions, useContradictions } from "../contradictions";
import { formatMoney, runStopText } from "../cost";
import { useLanguage, useT } from "../i18n";

/** The last check a new version of a Document started in the Mind, as it changes. */
function useWatchRun(doc: Y.Doc): ContradictionWatchRun | null {
  const [run, setRun] = useState(() => contradictionWatchOf(doc));
  useEffect(() => {
    setRun(contradictionWatchOf(doc));
    return observeContradictionWatch(doc, () => setRun(contradictionWatchOf(doc)));
  }, [doc]);
  return run;
}

/**
 * Contradiction watch's line under the Mind's title (#112), beside its
 * health: the question before a run billed per token, with the estimate;
 * what a run on demand found; and what the last run a new version of a
 * Document started found and cost. Each closes; the last for good.
 */
export function ContradictionLine({ doc, mindId }: { doc: Y.Doc; mindId: string }) {
  const t = useT();
  const language = useLanguage();
  const asking = useContradictions((state) => state.asking[mindId]);
  const found = useContradictions((state) => state.found[mindId]);
  const watched = useWatchRun(doc);
  const format = (value: number) => new Intl.NumberFormat(language).format(value);

  const outcome = (run: ContradictionCheckOutcome) => {
    const counts = { count: format(run.claims), contradicted: format(run.contradicted) };
    const said =
      run.contradicted > 0
        ? t("contradictions.found", counts)
        : t(run.claims === 1 ? "contradictions.none.one" : "contradictions.none", counts);
    return run.stop ? `${said} · ${runStopText(run.stop, t, language)}` : said;
  };

  if (asking) {
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
        {t(asking.claims === 1 ? "contradictions.ask.one" : "contradictions.ask", {
          count: format(asking.claims),
          estimate: asking.estimate,
        })}
      </ProblemLine>
    );
  }
  if (found) {
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
        {outcome(found)}
      </ProblemLine>
    );
  }
  if (watched) {
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
