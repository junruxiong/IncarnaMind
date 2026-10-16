import type { SentenceResult } from "../../shared/claimChecks";
import type { MessageKey } from "../../shared/i18n";
import type { CitationCheck } from "../api";
import type { RecordItem, TextOrigin } from "./read";

const SENTENCE_RESULTS: Readonly<Record<SentenceResult, MessageKey>> = {
  supported: "aiStatement.sentences.supported",
  unsupported: "aiStatement.sentences.unsupported",
  contradicted: "aiStatement.sentences.contradicted",
  needsRecheck: "aiStatement.sentences.recheck",
  notChecked: "aiStatement.sentences.notChecked",
  changed: "aiStatement.sentences.changed",
};

const CITATION_RESULTS: Readonly<Record<CitationCheck, MessageKey>> = {
  found: "aiStatement.citations.found",
  "not-found": "aiStatement.citations.notFound",
  "cant-check": "aiStatement.citations.cantCheck",
  unsupported: "aiStatement.citations.unsupported",
  contradicted: "aiStatement.citations.contradicted",
  checking: "aiStatement.citations.checking",
};

/** A result's words: the AI use statement's, so one result has one name. */
export function resultKey({ kind, result }: Pick<RecordItem, "kind" | "result">): MessageKey {
  if (kind === "sentence") return SENTENCE_RESULTS[result as SentenceResult];
  return CITATION_RESULTS[result as CitationCheck];
}

export const ORIGIN_LABELS: Readonly<Record<TextOrigin, MessageKey>> = {
  user: "checkRecord.origin.user",
  ai: "checkRecord.origin.ai",
  other: "checkRecord.origin.other",
};
