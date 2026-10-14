/**
 * The claim-check function (#96): one claim checked against the Documents
 * in a check Run, the one way Find support and Check against Documents
 * check a sentence, and later verify (#97) and contradiction watch (#112).
 *
 * The Run is the Answer engine's Tool-calling loop (ADR-0007) with the check
 * Tool set: the Documents' search and `cite`, nothing else (ADR-0013, #78).
 * Its searches leave out what a check may not use as evidence (#80): the
 * files IncarnaMind wrote, unless counted, and the document being checked.
 * The model records the Passage that decides the claim with `cite`, and its
 * quote must pass the Citation check before the verdict is taken: a
 * "supported" or "contradicted" without a quote that is found is not
 * checked. A refusal, a reply with no verdict, an empty one or a provider's
 * failure leaves the claim not checked, with the reason; never supported.
 */

import { generateText } from "ai";
import type {
  CheckedVersion,
  Citation,
  ClaimCheckFailure,
  ClaimCheckResult,
  LeftOutDocument,
  ProviderError,
  ThirdPartyReason,
} from "../../api";
import type { Evidence, EvidenceScope } from "../../documents/evidence";
import type { Language } from "../../language";
import { givesStructuredOutput } from "../../providers/capabilities";
import type { PreparedChatModel } from "../../providers/chat";
import { type AnswerDocuments, createCitationSession } from "../citations";
import {
  type AnswerEngine,
  type AnswerTools,
  answerTemperature,
  type DocumentLanguage,
} from "../engine";
import {
  CONFIRM_INSTRUCTIONS,
  checkInstructions,
  claimPrompt,
  confirmPrompt,
  EVIDENCE_INSTRUCTIONS,
  evidencePrompt,
  type ReadVerdict,
  readEvidence,
  readJudgement,
  readsAsRefusal,
  readVerdict,
} from "./prompt";

export interface ClaimCheckDependencies {
  /** The interface language: what a check's sentence is written in. */
  language(): Language;
  engine: AnswerEngine;
  evidence: Pick<Evidence, "search">;
  documents: Pick<AnswerDocuments, "citationSource" | "pageTexts" | "documentDate"> & {
    searchableCount(documentIds: readonly string[] | null): number;
    searchableLanguages(documentIds: readonly string[] | null): DocumentLanguage[];
  };
}

export interface ClaimCheckRequest {
  claim: string;
  context?: string;
  cited?: readonly { documentName: string; quote: string }[];
  scope: EvidenceScope;
  model: PreparedChatModel;
  /** The Run reads the document being checked (#97): it is third-party from its start. */
  thirdParty?: readonly ThirdPartyReason[];
  /** Names the Mind, for a provider's prompt cache. */
  cacheKey?: string;
  signal: AbortSignal;
}

/** Why a provider's failure left a claim not checked. */
function failureOf(error: ProviderError): { reason: ClaimCheckFailure; detail: string | null } {
  switch (error.kind) {
    case "rate-limit":
      return { reason: "rate-limit", detail: null };
    case "cost-cap":
      return { reason: "cost-cap", detail: null };
    case "plan-limit":
      return { reason: "plan-limit", detail: null };
    case "no-answer":
      return { reason: "empty", detail: null };
    default:
      return { reason: "failed", detail: error.message };
  }
}

export function createClaimCheck(deps: ClaimCheckDependencies) {
  return async function checkClaim(request: ClaimCheckRequest): Promise<ClaimCheckResult> {
    const { scope, signal } = request;
    const read = new Map<string, string>();
    const excluded = new Map<string, LeftOutDocument>();
    const scoped: AnswerDocuments = {
      searchableCount: () => deps.documents.searchableCount(scope.documentIds),
      async search(query, searchSignal) {
        const found = await deps.evidence.search(query, scope, searchSignal);
        for (const passage of found.passages) read.set(passage.documentId, passage.contentHash);
        for (const each of found.excluded) {
          if (excluded.has(each.documentId)) continue;
          excluded.set(each.documentId, {
            documentId: each.documentId,
            documentName: each.documentName,
            reason: each.reason,
          });
        }
        return found.passages;
      },
      citationSource: (passageId) => deps.documents.citationSource(passageId),
      pageTexts: (documentId, contentHash, from, to) =>
        deps.documents.pageTexts(documentId, contentHash, from, to),
      documentDate: (documentId) => deps.documents.documentDate(documentId),
    };
    const session = createCitationSession(scoped, { onRecord: () => undefined });
    const tools: AnswerTools = {
      get documentCount() {
        return session.tools.documentCount;
      },
      documentLanguages: deps.documents.searchableLanguages(scope.documentIds),
      searchDocuments: (query, searchSignal, options) =>
        session.tools.searchDocuments(query, searchSignal, options),
      cite: (records, options) => session.tools.cite(records, options),
      hasRecord: (marker) => session.tools.hasRecord?.(marker) ?? false,
      unfoundQuotes: () => session.tools.unfoundQuotes?.() ?? [],
      correctQuote: (marker, quote) => session.tools.correctQuote?.(marker, quote) ?? false,
    };

    const facts = () => ({
      versions: [...read].map(
        ([documentId, contentHash]): CheckedVersion => ({ documentId, contentHash }),
      ),
      excluded: [...excluded.values()].filter((each) => each.reason === "own-file"),
    });
    const notChecked = (
      reason: ClaimCheckFailure,
      detail: string | null = null,
    ): ClaimCheckResult => ({
      verdict: "not-checked",
      reason,
      detail,
      ...facts(),
    });

    let reply = "";
    let failed: ProviderError | null = null;
    let finished = false;
    const { model } = request;
    for await (const event of deps.engine.generate({
      instructions: (mode, options) => checkInstructions(mode, deps.language(), options?.passages),
      messages: [{ role: "user", content: claimPrompt({ ...request, language: deps.language() }) }],
      question: request.claim,
      model: model.model,
      documents: tools,
      tools: [],
      toolSet: "check",
      thirdParty: request.thirdParty ?? [],
      support: model.support,
      structuredOutput: givesStructuredOutput(model.capabilities),
      takesTemperature: model.capabilities.facts.temperature,
      window: model.window,
      retryQuotes: model.retryQuotes,
      cacheKey: request.cacheKey,
      signal,
    })) {
      if (event.type === "text-delta") reply += event.text;
      else if (event.type === "text-retracted") reply = reply.slice(0, -event.length);
      else if (event.type === "failed") failed = event.error;
      else if (event.type === "finished") finished = true;
    }
    if (signal.aborted) return notChecked("stopped");
    if (failed) {
      const { reason, detail } = failureOf(failed);
      return notChecked(reason, detail);
    }
    if (!finished || reply.trim() === "") return notChecked("empty");
    const verdict = readVerdict(reply);
    if (!verdict) return notChecked(readsAsRefusal(reply) ? "refused" : "malformed");
    if (verdict.verdict === "unsupported") {
      return { verdict: "unsupported", evidence: null, says: null, ...facts() };
    }
    session.check();
    const evidence =
      decidingCitation(session.finalCitation) ??
      (await askForEvidence(request, verdict, session, signal));
    if (signal.aborted) return notChecked("stopped");
    if (!evidence) return notChecked("no-evidence");
    if (verdict.verdict === "contradicted") {
      const judged = await confirmContradiction(request, evidence, session, signal);
      if (signal.aborted) return notChecked("stopped");
      if (judged === "neither") {
        return { verdict: "unsupported", evidence: null, says: null, ...facts() };
      }
      if (judged === "same") return { verdict: "supported", evidence, says: null, ...facts() };
    }
    return { verdict: verdict.verdict, evidence, says: verdict.says, ...facts() };
  };
}

/**
 * A model that gave a verdict without a record of the Passage that decides
 * it, or with a quote the check doesn't find, is asked once more, in one
 * request, for that record from the Passages it read (Claude Haiku 5.5 left
 * `cite` out of about one check in four). The record is taken as structured
 * output's are, and counts only when the check finds its quote.
 */
async function askForEvidence(
  request: ClaimCheckRequest,
  verdict: ReadVerdict,
  session: ReturnType<typeof createCitationSession>,
  signal: AbortSignal,
): Promise<Citation | null> {
  const passages = session.shownPassages();
  if (passages.length === 0) return null;
  const { model } = request;
  let reply: string;
  try {
    ({ text: reply } = await generateText({
      model: model.model,
      instructions: EVIDENCE_INSTRUCTIONS,
      prompt: evidencePrompt({ claim: request.claim, ...verdict, passages }),
      temperature: answerTemperature(model.capabilities.facts.temperature),
      maxOutputTokens: 600,
      maxRetries: 1,
      abortSignal: signal,
    }));
  } catch (error) {
    if (!signal.aborted)
      console.error(`A check's evidence couldn't be asked for again: ${messageOf(error)}`);
    return null;
  }
  const record = readEvidence(reply);
  if (!record) return null;
  session.tools.cite([{ marker: EVIDENCE_MARKER, ...record }], { structured: true });
  session.check();
  const citation = session.finalCitation(EVIDENCE_MARKER);
  return citation?.check === "found" ? citation : null;
}

/**
 * A contradiction counts only once the sentence and the quote it rests on,
 * judged alone, conflict. In a whole check a model leant to "contradicted"
 * for near misses: a figure for a related thing, another site in a parallel
 * Document (#97). A reply that can't be read, or a failed request, leaves
 * the verdict as it was.
 */
async function confirmContradiction(
  request: ClaimCheckRequest,
  evidence: Citation,
  session: ReturnType<typeof createCitationSession>,
  signal: AbortSignal,
): Promise<"conflict" | "same" | "neither" | null> {
  const passage = session.shownPassages().find((each) => each.passageId === evidence.passageId);
  const { model } = request;
  try {
    const { text } = await generateText({
      model: model.model,
      instructions: CONFIRM_INSTRUCTIONS,
      prompt: confirmPrompt({
        claim: request.claim,
        quote: evidence.quote,
        passage: passage?.text ?? evidence.quote,
      }),
      temperature: answerTemperature(model.capabilities.facts.temperature),
      maxOutputTokens: 200,
      maxRetries: 1,
      abortSignal: signal,
    });
    return readJudgement(text);
  } catch (error) {
    if (!signal.aborted)
      console.error(`A contradiction couldn't be confirmed: ${messageOf(error)}`);
    return null;
  }
}

const messageOf = (error: unknown) => (error instanceof Error ? error.message : String(error));

/** The marker the record asked for once more is given: apart from any the model gave. */
const EVIDENCE_MARKER = 99;

/** The Citation that decides a claim: marker 1's, or the first other marker's, whose quote the check found. */
function decidingCitation(finalCitation: (marker: number) => Citation | null): Citation | null {
  for (let marker = 1; marker <= 4; marker++) {
    const citation = finalCitation(marker);
    if (citation?.check === "found") return citation;
  }
  return null;
}
