/**
 * The claim-check function (#96): one claim checked against the Documents
 * in a check Run, the one way Find support and Check against Documents
 * check a sentence, and later verify (#97) and contradiction watch (#112).
 *
 * The Run is the Answer engine's Tool-calling loop (ADR-0007) with the check
 * Tool set: the Documents' search and `cite`, nothing else (ADR-0013, #78).
 * Its searches leave out what a check may not use as evidence (#80): the
 * files IncarnaMind wrote, unless counted, the User's own workings and the
 * document being checked.
 * The model records the Passage that decides the claim with `cite`, and its
 * quote must pass the Citation check before the verdict is taken: a
 * "supported" or "contradicted" without a quote that is found is not
 * checked. A refusal, a reply with no verdict, an empty one or a provider's
 * failure leaves the claim not checked, with the reason; never supported.
 *
 * A model measured to use `calculate` (ADR-0021) is offered it too, for a
 * figure the sentence states that no Passage does but their numbers give
 * (#707). It names the sentence's number as `stated`; IncarnaMind works the
 * figure out and compares, not the model. A figure that disagrees contradicts
 * the sentence; one that agrees settles that number only, and the model's
 * verdict on the rest of the sentence still stands.
 */

import { generateText } from "ai";
import { type CaseCitation, caseCitations, isOpinionOf } from "../../../shared/caseCitations";
import type { FigureContext } from "../../../shared/figures";
import type {
  CheckedVersion,
  Citation,
  ClaimCalculation,
  ClaimCheckFailure,
  ClaimCheckResult,
  LeftOutDocument,
  PassageSearchResult,
  ProviderError,
  RecordAiUseInput,
  ThirdPartyReason,
} from "../../api";
import type { Evidence, EvidenceScope } from "../../documents/evidence";
import type { Language } from "../../language";
import { calculates, givesStructuredOutput } from "../../providers/capabilities";
import type { PreparedChatModel } from "../../providers/chat";
import {
  answerLanguage,
  type CalculationSession,
  createCalculationSession,
  documentFigureContext,
  givenList,
  whole,
} from "../calculations";
import { type AnswerDocuments, createCitationSession } from "../citations";
import {
  type AnswerEngine,
  type AnswerTools,
  answerTemperature,
  type CiteOptions,
  type DocumentLanguage,
} from "../engine";
import { agreesWith, statedFigure } from "./calculated";
import { figuresMatch, nearestLine } from "./nearest";
import {
  CONFIRM_INSTRUCTIONS,
  checkInstructions,
  claimPrompt,
  confirmPrompt,
  EVIDENCE_INSTRUCTIONS,
  evidencePrompt,
  NEAREST_INSTRUCTIONS,
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
    savedBySearch(documentId: string): boolean;
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

/** A claim check's failure that the run's other claims share: waited out, or it stops the run. */
export function runFailureOf(result: ClaimCheckResult): ProviderError | null {
  if (result.verdict !== "not-checked") return null;
  const { reason } = result;
  if (reason !== "rate-limit" && reason !== "cost-cap" && reason !== "plan-limit") return null;
  return { kind: reason, message: result.detail ?? reason };
}

/** Records a sentence a model judged as a use of AI (#906); one it couldn't judge isn't one. */
export function recordJudged(
  record: (inputs: RecordAiUseInput[]) => void,
  mindId: string,
  model: PreparedChatModel,
  result: ClaimCheckResult,
  at: string,
): void {
  if (result.verdict === "not-checked") return;
  record([
    {
      mindId,
      kind: "sentence-checked",
      providerId: model.provider.id,
      modelId: model.modelId,
      characters: 0,
      occurredAt: at,
    },
  ]);
}

/** What the model is told when it gives a verdict before searching, in the same Run (#972). */
const SEARCH_FIRST =
  "You gave a verdict without searching the User's Documents. Search them with search_documents first, then give your verdict from what you find.";

export function createClaimCheck(deps: ClaimCheckDependencies) {
  return async function checkClaim(request: ClaimCheckRequest): Promise<ClaimCheckResult> {
    const { scope, signal } = request;
    let searched = false;
    const read = new Map<string, string>();
    const excluded = new Map<string, LeftOutDocument>();
    const decisions = caseCitations(request.claim);
    const workings = new Map<string, PassageSearchResult[]>();
    const scoped: AnswerDocuments = {
      searchableCount: () => deps.documents.searchableCount(scope.documentIds),
      async search(query, searchSignal) {
        searched = true;
        const found = await deps.evidence.search(query, scope, searchSignal);
        // A web search's result never settles a citation of a court decision.
        const passages =
          decisions.length === 0
            ? found.passages
            : found.passages.filter((passage) => !deps.documents.savedBySearch(passage.documentId));
        for (const passage of passages) read.set(passage.documentId, passage.contentHash);
        for (const each of found.excluded) {
          if (each.reason === "own-workings") {
            workings.set(each.documentId, [
              ...(workings.get(each.documentId) ?? []),
              ...each.passages,
            ]);
          }
          if (excluded.has(each.documentId)) continue;
          excluded.set(each.documentId, {
            documentId: each.documentId,
            documentName: each.documentName,
            reason: each.reason,
          });
        }
        return passages;
      },
      citationSource: (passageId) => deps.documents.citationSource(passageId),
      pageTexts: (documentId, contentHash, from, to) =>
        deps.documents.pageTexts(documentId, contentHash, from, to),
      documentDate: (documentId) => deps.documents.documentDate(documentId),
    };
    const session = createCitationSession(scoped, { onRecord: () => undefined });
    const { model } = request;
    const calculating = model.support === "tools" && calculates(model.capabilities);
    const figures = createCalculationSession({
      documents: scoped,
      quotes: session.quotes,
      language: answerLanguage(request.claim),
    });
    /** The sentence's number each figure is for, as the model named it, by marker. */
    const stated = new Map<number, string>();
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
      ...(calculating && {
        calculate(given: readonly unknown[], options?: CiteOptions) {
          for (const each of given) {
            const fields =
              typeof each === "object" && each !== null ? (each as Record<string, unknown>) : {};
            const marker = whole(fields.marker);
            if (marker !== null && typeof fields.stated === "string" && fields.stated.trim() !== "")
              stated.set(marker, fields.stated.trim());
          }
          return figures.calculate(givenList(given), options);
        },
      }),
    };

    const facts = () => ({
      versions: [...read].map(
        ([documentId, contentHash]): CheckedVersion => ({ documentId, contentHash }),
      ),
      excluded: [...excluded.values()].filter((each) => each.reason === "own-file"),
    });
    const workingsSayingIt = (): LeftOutDocument[] =>
      [...workings].flatMap(([documentId, passages]) => {
        const left = excluded.get(documentId);
        const speaks = passages.some(
          (passage) =>
            figuresMatch(request.claim, passage.text) &&
            nearestLine(request.claim, [{ id: passage.passageId, text: passage.text }]),
        );
        return left && speaks ? [left] : [];
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

    let opinions: Opinions;
    try {
      opinions = await opinionsOf(decisions, scoped, signal);
    } catch (error) {
      if (signal.aborted) return notChecked("stopped");
      throw error;
    }
    const citedCase = (evidence: Citation | null) => ({
      citedCaseUnchecked: opinions.missing,
      citedCaseFound:
        !opinions.missing &&
        evidence !== null &&
        opinions.documentIds.has(evidence.documentId ?? ""),
    });

    let reply = "";
    let failed: ProviderError | null = null;
    let finished = false;
    for await (const event of deps.engine.generate({
      instructions: (mode, options) =>
        checkInstructions(mode, deps.language(), options?.passages, calculating),
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
      // A model may give its verdict before it has looked (Claude Haiku 5.5 did, about one check
      // in six of a whole Mind): it is asked once more, within the Run's steps.
      checkReply: (text) =>
        !searched && readVerdict(text) ? { claims: [], note: SEARCH_FIRST } : null,
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
    const worked = calculated(request.claim, figures, stated, (citation) =>
      documentFigureContext(
        scoped.pageTexts(citation.documentId, citation.contentHash, null, null),
      ),
    );
    // A figure that disagrees contradicts the sentence. One that agrees settles only that
    // number: the model's verdict still stands on the rest of the sentence.
    if (worked && !worked.calculation.agrees) {
      return {
        verdict: "contradicted",
        evidence: worked.evidence,
        says: null,
        calculation: worked.calculation,
        ...citedCase(worked.evidence),
        ...facts(),
      };
    }
    const byCalculation: ClaimCheckResult | null = worked && {
      verdict: "supported",
      evidence: worked.evidence,
      says: null,
      calculation: worked.calculation,
      ...citedCase(worked.evidence),
      ...facts(),
    };
    if (!finished || reply.trim() === "") return byCalculation ?? notChecked("empty");
    const verdict = readVerdict(reply);
    if (!verdict) {
      return byCalculation ?? notChecked(readsAsRefusal(reply) ? "refused" : "malformed");
    }
    if (!searched) return notChecked("unread");
    if (byCalculation && verdict.verdict !== "contradicted") return byCalculation;
    /** Unsupported, with the nearest line when it states another value for the same thing (#770). */
    const unsupported = async (): Promise<ClaimCheckResult> => {
      const nearest = await nearestDifferent(request, session, signal);
      if (signal.aborted) return notChecked("stopped");
      const left = facts();
      return {
        verdict: "unsupported",
        evidence: nearest,
        says: null,
        calculation: null,
        ...citedCase(nearest),
        ...left,
        excluded: nearest ? left.excluded : [...left.excluded, ...workingsSayingIt()],
      };
    };
    session.check();
    if (verdict.verdict === "unsupported") return unsupported();
    const evidence =
      decidingCitation(session.finalCitation) ??
      (await askForEvidence(request, verdict, session, signal));
    if (signal.aborted) return notChecked("stopped");
    if (!evidence) return byCalculation ?? notChecked("no-evidence");
    if (verdict.verdict === "contradicted") {
      const judged = await confirmContradiction(request, evidence, session, signal);
      if (signal.aborted) return notChecked("stopped");
      // The Passage is about something else: what is nearest may still say something different.
      if (judged === "neither") return byCalculation ?? unsupported();
      if (judged === "same") {
        return (
          byCalculation ?? {
            verdict: "supported",
            evidence,
            says: null,
            calculation: null,
            ...citedCase(evidence),
            ...facts(),
          }
        );
      }
    }
    return {
      verdict: verdict.verdict,
      evidence,
      says: verdict.says,
      calculation: worked?.calculation ?? null,
      ...citedCase(evidence),
      ...facts(),
    };
  };
}

/**
 * The figure the model worked out for a number of the sentence, once every
 * input's quote is found, and whether the number agrees with it. Null when
 * there is none to decide by: no figure recorded for a number the sentence
 * has, an input's quote not found, or a result that can't be compared.
 */
function calculated(
  claim: string,
  figures: CalculationSession,
  stated: ReadonlyMap<number, string>,
  contextOf: (citation: Citation) => FigureContext,
): { evidence: Citation; calculation: ClaimCalculation } | null {
  figures.check();
  for (const [marker, named] of stated) {
    const figure = figures.figure(marker);
    const first = figure?.inputs[0]?.citation;
    if (!figure || !first) continue;
    const number = statedFigure(claim, named, contextOf(first));
    if (!number) continue;
    if (figure.inputs.some((input) => input.citation.check !== "found")) continue;
    const agrees = agreesWith(number, figure);
    if (agrees === null) continue;
    return { evidence: first, calculation: { figure, stated: number.chars, agrees } };
  }
  return null;
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
  instructions = CONFIRM_INSTRUCTIONS,
): Promise<"conflict" | "same" | "neither" | null> {
  const passage = session.shownPassages().find((each) => each.passageId === evidence.passageId);
  const { model } = request;
  try {
    const { text } = await generateText({
      model: model.model,
      instructions,
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

interface Opinions {
  documentIds: ReadonlySet<string>;
  missing: boolean;
}

function startOf(documents: AnswerDocuments, documentId: string, contentHash: string): string {
  const [first] = documents.pageTexts(documentId, contentHash, 1, 1);
  return first?.text ?? documents.pageTexts(documentId, contentHash, null, null)[0]?.text ?? "";
}

async function opinionsOf(
  decisions: readonly CaseCitation[],
  documents: AnswerDocuments,
  signal: AbortSignal,
): Promise<Opinions> {
  const documentIds = new Set<string>();
  let missing = false;
  for (const decision of decisions) {
    const query = [decision.name, decision.cite].filter(Boolean).join(" ");
    const found = await documents.search(query, signal);
    const candidates = new Map(found.map((each) => [each.documentId, each]));
    const opinions = [...candidates.values()].filter((each) =>
      isOpinionOf(
        {
          name: each.documentName,
          start: startOf(documents, each.documentId, each.contentHash),
        },
        decision,
      ),
    );
    if (opinions.length === 0) missing = true;
    for (const each of opinions) documentIds.add(each.documentId);
  }
  return { documentIds, missing };
}

/**
 * For a sentence found unsupported: the nearest line of what the Run read, as
 * a Citation whose quote the check found, when judged alone it states another
 * value for the very thing the sentence says (30 days of leave against the
 * Document's 27). Never "contradicted": the verdict stays unsupported, and the
 * card shows what the Document says. Null when nothing read is near enough, or
 * the nearest line doesn't differ.
 */
async function nearestDifferent(
  request: ClaimCheckRequest,
  session: ReturnType<typeof createCitationSession>,
  signal: AbortSignal,
): Promise<Citation | null> {
  const line = nearestLine(request.claim, session.shownPassages());
  if (!line) return null;
  session.tools.cite(
    [{ marker: NEAREST_MARKER, passage: line.id, location: null, quote: line.text }],
    {
      structured: true,
    },
  );
  session.check();
  const citation = session.finalCitation(NEAREST_MARKER);
  if (citation?.check !== "found") return null;
  const judged = await confirmContradiction(
    request,
    citation,
    session,
    signal,
    NEAREST_INSTRUCTIONS,
  );
  return judged === "conflict" ? citation : null;
}

const messageOf = (error: unknown) => (error instanceof Error ? error.message : String(error));

/** The marker the nearest line is recorded under: apart from the model's and the evidence's. */
const NEAREST_MARKER = 98;

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
