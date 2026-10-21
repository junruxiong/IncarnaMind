import type * as Y from "yjs";
import { claimCheckShown } from "../../shared/claimChecks";
import { checkedSentences } from "../answers/checks/sentences";
import type { CitationAttributes } from "../api";
import type { FailedSentence } from "../mindText";

/** What the checks of a Mind's sentences are compared with, and how an export names things. */
export interface SentenceCheckContext {
  versions: ReadonlyMap<string, string>;
  folderId: string | null;
  folderName(folderId: string): string | null;
  /** A Passage's Document and Location, as a footnote says it. */
  sourceOf(passage: CitationAttributes): string;
}

/** The checks of the Mind's sentences that aren't good, by check id: as they show now, not as they were made. */
export function failedSentences(
  fragment: Y.XmlFragment,
  context: SentenceCheckContext,
): Map<string, FailedSentence> {
  const failed = new Map<string, FailedSentence>();
  for (const { check, sentence } of checkedSentences(fragment)) {
    const { id } = check;
    const shown = claimCheckShown(check, {
      text: sentence,
      versions: context.versions,
      folderId: context.folderId,
    });
    if (shown.state === "contradicted") {
      const { calculation } = check;
      failed.set(id, {
        id,
        check: "contradicted",
        says: check.says,
        worked:
          calculation && !calculation.agrees
            ? { shown: calculation.figure.shown, stated: calculation.stated }
            : null,
      });
    } else if (shown.state === "unsupported") {
      const { evidence, scope } = check;
      failed.set(id, {
        id,
        check: "unsupported",
        folder:
          !scope.widened && scope.folderId !== null ? context.folderName(scope.folderId) : null,
        nearest: evidence?.quote
          ? { quote: evidence.quote, source: context.sourceOf(evidence) }
          : null,
      });
    } else if (shown.state === "needs-recheck") {
      failed.set(id, { id, check: "needs-recheck", why: shown.why });
    } else if (shown.state === "supported" && check.citedCaseUnchecked) {
      failed.set(id, { id, check: "cited-case-unchecked" });
    }
  }
  return failed;
}
