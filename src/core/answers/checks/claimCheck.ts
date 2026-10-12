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
import { givesStructuredOutput } from "../../providers/capabilities";
import type { PreparedChatModel } from "../../providers/chat";
import { type AnswerDocuments, createCitationSession } from "../citations";
import type { AnswerEngine, AnswerTools, DocumentLanguage } from "../engine";
import { checkInstructions, claimPrompt, readsAsRefusal, readVerdict } from "./prompt";

export interface ClaimCheckDependencies {
  engine: AnswerEngine;
  evidence: Pick<Evidence, "search">;
  documents: Pick<AnswerDocuments, "citationSource" | "pageTexts"> & {
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
      instructions: (mode, options) => checkInstructions(mode, options?.passages),
      messages: [{ role: "user", content: claimPrompt(request) }],
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
    const evidence = decidingCitation(session.finalCitation);
    if (!evidence) return notChecked("no-evidence");
    return { verdict: verdict.verdict, evidence, says: verdict.says, ...facts() };
  };
}

/** The Citation that decides a claim: marker 1's, or the first other marker's, whose quote the check found. */
function decidingCitation(finalCitation: (marker: number) => Citation | null): Citation | null {
  for (let marker = 1; marker <= 4; marker++) {
    const citation = finalCitation(marker);
    if (citation?.check === "found") return citation;
  }
  return null;
}
