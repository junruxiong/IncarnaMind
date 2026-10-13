/**
 * Checks of the User's own sentences (#96): Find support and Check against
 * Documents in a Mind, and the claim-check function they share (./claimCheck).
 *
 * 1. The sentences the selection touches are read from the Mind, whole, and
 *    each gets a "checking" mark at once (./sentences).
 * 2. They are checked one at a time against the Mind's Folder (every
 *    Document for a Mind in none, or when the User widens it), each in a
 *    check Run that can only search and cite.
 * 3. Each result is written onto its sentence's mark as it comes, with what
 *    it was made from: the sentence's text, the Document versions it read
 *    and the scope, so it shows "needs re-check" once any of them changes
 *    (src/shared/claimChecks.ts). A mark whose sentence was deleted meanwhile
 *    is simply gone.
 *
 * The evidence of a Mind's checks is retained (#79) under the Mind, so its
 * text stays for the cards and "Add as Citation" after the Document changes.
 * Stopping a request, or quitting, marks the sentences not checked yet
 * "stopped"; a "checking" mark left by a crash says so when its Mind opens.
 *
 * Nothing leaves the computer before the User allows the "checks" flow to
 * the model's service, and in local mode only a model on this computer is used.
 */
import { randomUUID } from "node:crypto";
import type * as Y from "yjs";
import { judgesUnreliably } from "../../../shared/smallModel";
import {
  type ChatModelChoice,
  type ChatReadiness,
  type CheckSentencesResult,
  type Citation,
  type CitationAttributes,
  type ClaimCheckAction,
  type ClaimCheckAttributes,
  type ClaimCheckResult,
  type ClaimCheckScope,
  claimCheckActions,
  type ExternalService,
  type SearchScope,
  type TextPlace,
} from "../../api";
import type { Consent } from "../../consent";
import type { Evidence } from "../../documents/evidence";
import { parseCheckedDocument } from "../../documents/evidence";
import type { RetainVersion, VersionRetention } from "../../documents/retention";
import { parseModel, parsePlace } from "../../edits";
import { ChatNotReadyError, ConsentDeclinedError, InvalidInputError, isRecord } from "../../errors";
import type { MindContent } from "../../mindContent";
import { parseMindId } from "../../minds";
import type { PreparedChatModel } from "../../providers/chat";
import { parseSearchScope } from "../../scope";
import type { AnswerEngine } from "../engine";
import { type ClaimCheckDependencies, createClaimCheck } from "./claimCheck";
import { markedSentences, markSentence, selectedSentences, updateCheck } from "./sentences";

/** What the "checks" flow sends: the sentences checked, with their paragraphs, and the Passages found. */
export const CHECKS_FLOW_SENDS = ["selection", "passages"] as const;

/** The longest claim one check takes, in characters: a sentence, not a page. */
const MAX_CLAIM_CHARACTERS = 2_000;

export interface ChecksOptions {
  content: Pick<MindContent, "read" | "edit">;
  /** Throws unless the Mind exists and this version may write to it. */
  requireMind(mindId: string): void;
  /** The Folder the Mind is in; null for none. */
  mindFolder(mindId: string): string | null;
  /** The default chat model, used when the Mind has none of its own. */
  defaultModel(): ChatModelChoice | null;
  chat: {
    readiness(choice: ChatModelChoice | undefined, flow: "checks"): Promise<ChatReadiness>;
    prepareModel(choice: ChatModelChoice | undefined, flow: "checks"): Promise<PreparedChatModel>;
    /** The service the default chat model sends to; null with none, or one on this computer. */
    defaultService(): ExternalService | null;
  };
  consent: Consent;
  /** Local mode: only models on this computer. */
  localOnly(): boolean;
  resolveScope(scope: SearchScope): string[] | null;
  evidence: Pick<Evidence, "scope" | "search">;
  documents: ClaimCheckDependencies["documents"];
  engine: AnswerEngine;
  retention: Pick<VersionRetention, "retain">;
  now(): Date;
}

interface ParsedCheck {
  requestId: string;
  mindId: string;
  from: TextPlace;
  to: TextPlace;
  action: ClaimCheckAction;
  allDocuments: boolean;
  model: ChatModelChoice | null;
}

function parseCheck(input: unknown): ParsedCheck {
  if (!isRecord(input)) throw new InvalidInputError("checkSentences expects an object.");
  const { requestId, action } = input;
  if (typeof requestId !== "string" || requestId === "") {
    throw new InvalidInputError("A check needs a request id.");
  }
  const known = claimCheckActions.find((each) => each === action);
  if (!known) throw new InvalidInputError(`Unknown check "${String(action)}".`);
  return {
    requestId,
    mindId: parseMindId(input.mindId),
    from: parsePlace(input.from, "from"),
    to: parsePlace(input.to, "to"),
    action: known,
    allDocuments: input.allDocuments === true,
    model: parseModel(input.model),
  };
}

function parseClaim(input: unknown) {
  if (!isRecord(input)) throw new InvalidInputError("checkClaim expects an object.");
  const claim = typeof input.claim === "string" ? input.claim.trim() : "";
  if (claim === "") throw new InvalidInputError("A claim to check must be some text.");
  if (claim.length > MAX_CLAIM_CHARACTERS)
    throw new InvalidInputError({
      key: "error.check.claimTooLong",
      values: { max: MAX_CLAIM_CHARACTERS },
    });
  const cited = Array.isArray(input.cited)
    ? input.cited.flatMap((each: unknown) =>
        isRecord(each) && typeof each.quote === "string"
          ? [{ documentName: String(each.documentName ?? ""), quote: each.quote }]
          : [],
      )
    : [];
  return {
    claim,
    context: typeof input.context === "string" ? input.context : undefined,
    cited,
    scope: parseSearchScope(input.scope),
    checked: parseCheckedDocument(input.checked),
    model: parseModel(input.model),
  };
}

/** A Citation as a mark stores it: the Passage that decides a claim. */
const evidenceOf = (citation: Citation | null): CitationAttributes | null =>
  citation && {
    passageId: citation.passageId,
    documentId: citation.documentId,
    documentName: citation.documentName,
    contentHash: citation.contentHash,
    pageFrom: citation.pageFrom,
    pageTo: citation.pageTo,
    location: citation.location,
    quote: citation.quote,
    check: citation.check,
    checkReason: citation.checkReason,
  };

/** A result written onto its mark. */
function withResult(
  check: ClaimCheckAttributes,
  result: ClaimCheckResult,
  at: string,
): ClaimCheckAttributes {
  const common = { versions: result.versions, excluded: result.excluded, checkedAt: at };
  return result.verdict === "not-checked"
    ? { ...check, ...common, state: "not-checked", reason: result.reason, detail: result.detail }
    : {
        ...check,
        ...common,
        state: result.verdict,
        reason: null,
        detail: null,
        evidence: evidenceOf(result.evidence),
        says: result.says,
      };
}

const stopped = (check: ClaimCheckAttributes): ClaimCheckAttributes =>
  check.state === "checking" ? { ...check, state: "not-checked", reason: "stopped" } : check;

export function createChecks(options: ChecksOptions) {
  const { content, chat } = options;
  /** The requests running, by id: how to stop each, and its Mind. */
  const running = new Map<string, { controller: AbortController; mindId: string }>();
  const claimCheck = createClaimCheck(options);

  options.consent.registry.register({
    id: "checks",
    sends: CHECKS_FLOW_SENDS,
    async services() {
      const service = chat.defaultService();
      return service ? [service] : [];
    },
  });

  /** The model to check with, or why there is none; the consent dialog shows before anything is sent. */
  async function prepare(
    choice: ChatModelChoice | null,
  ): Promise<
    | { ok: true; prepared: PreparedChatModel }
    | { ok: false; result: Exclude<CheckSentencesResult, { kind: "checked" }> }
  > {
    const readiness = await chat.readiness(choice ?? undefined, "checks");
    if (!readiness.ready) return { ok: false, result: { kind: "not-ready", readiness } };
    if (readiness.provider.service && options.localOnly()) {
      return { ok: false, result: { kind: "local-mode" } };
    }
    try {
      return { ok: true, prepared: await chat.prepareModel(choice ?? undefined, "checks") };
    } catch (error) {
      if (error instanceof ChatNotReadyError) {
        return { ok: false, result: { kind: "not-ready", readiness: error.readiness } };
      }
      if (error instanceof ConsentDeclinedError) {
        const now = await chat.readiness(choice ?? undefined, "checks");
        return {
          ok: false,
          result: now.ready ? { kind: "cancelled" } : { kind: "not-ready", readiness: now },
        };
      }
      throw error;
    }
  }

  /** Keeps the text of the evidence every check in the Mind shows, and lets go of the rest (#79). */
  function retainEvidence(mindId: string): void {
    const versions = content.read(mindId, (blocks) =>
      markedSentences(blocks).flatMap(({ check }): RetainVersion[] => {
        const evidence = check.evidence;
        if (!evidence?.documentId || !evidence.contentHash) return [];
        return [
          {
            documentId: evidence.documentId,
            contentHash: evidence.contentHash,
            pageFrom: evidence.pageFrom,
            pageTo: evidence.pageTo,
            location: evidence.location,
          },
        ];
      }),
    );
    options.retention.retain("check-result", mindId, versions);
  }

  /** Marks a request's sentences not checked yet "stopped". */
  function settleStopped(mindId: string, request: string): void {
    const ids = content.read(mindId, (blocks) =>
      markedSentences(blocks)
        .filter(({ check }) => check.request === request && check.state === "checking")
        .map(({ check }) => check.id),
    );
    if (ids.length === 0) return;
    content.edit(mindId, (blocks) => {
      for (const id of ids) updateCheck(blocks, id, stopped);
    });
  }

  async function checkSentences(
    input: ParsedCheck,
    signal: AbortSignal,
  ): Promise<CheckSentencesResult> {
    const { mindId } = input;
    const read = content.read(mindId, (blocks) => selectedSentences(blocks, input.from, input.to));
    if (!read.ok) return { kind: read.reason };

    const folderId = options.mindFolder(mindId);
    const scope: ClaimCheckScope = { folderId, widened: input.allDocuments };
    const searched =
      input.allDocuments || folderId === null
        ? null
        : options.resolveScope({ folderIds: [folderId], tagIds: [], documentIds: [] });
    const evidenceScope = options.evidence.scope(searched);
    if (options.documents.searchableCount(evidenceScope.documentIds) === 0) {
      return { kind: "no-documents", folderId: input.allDocuments ? null : folderId };
    }

    const ready = await prepare(input.model ?? options.defaultModel());
    if (!ready.ok) return ready.result;
    if (signal.aborted) return { kind: "cancelled" };

    const startedAt = options.now().toISOString();
    const queued = content.edit(mindId, (blocks: Y.XmlFragment) => {
      const again = selectedSentences(blocks, input.from, input.to);
      if (!again.ok) return [];
      return again.sentences.map((sentence) => {
        const check: ClaimCheckAttributes = {
          id: randomUUID(),
          request: input.requestId,
          action: input.action,
          state: "checking",
          reason: null,
          detail: null,
          text: sentence.text,
          evidence: null,
          says: null,
          scope,
          versions: [],
          excluded: [],
          checkedAt: startedAt,
        };
        markSentence(sentence.element, sentence.from, sentence.to, check);
        return {
          id: check.id,
          claim: sentence.text,
          context: sentence.context,
          cited: sentence.cited,
        };
      });
    });
    if (queued.length === 0) return { kind: "nothing" };

    let checked = 0;
    for (const sentence of queued) {
      if (signal.aborted) break;
      const result = await claimCheck({
        claim: sentence.claim,
        context: sentence.context,
        cited: input.action === "check-against-documents" ? sentence.cited : [],
        scope: evidenceScope,
        model: ready.prepared,
        cacheKey: mindId,
        signal,
      });
      if (signal.aborted) break;
      if (result.verdict !== "not-checked") checked++;
      const at = options.now().toISOString();
      const kept = content.edit(mindId, (blocks) =>
        updateCheck(blocks, sentence.id, (check) => withResult(check, result, at)),
      );
      if (kept) retainEvidence(mindId);
    }
    if (signal.aborted) {
      settleStopped(mindId, input.requestId);
      return { kind: "cancelled" };
    }
    return {
      kind: "checked",
      sentences: queued.length,
      checked,
      smallModel: judgesUnreliably(ready.prepared),
    };
  }

  return {
    async checkSentences(input: unknown): Promise<CheckSentencesResult> {
      const parsed = parseCheck(input);
      options.requireMind(parsed.mindId);
      if (running.has(parsed.requestId)) {
        throw new InvalidInputError(`The check ${parsed.requestId} is already running.`);
      }
      const controller = new AbortController();
      running.set(parsed.requestId, { controller, mindId: parsed.mindId });
      try {
        return await checkSentences(parsed, controller.signal);
      } finally {
        running.delete(parsed.requestId);
      }
    },

    async checkClaim(input: unknown): Promise<ClaimCheckResult> {
      const parsed = parseClaim(input);
      const ready = await prepare(parsed.model ?? options.defaultModel());
      if (!ready.ok) {
        const { result } = ready;
        if (result.kind === "not-ready") throw new ChatNotReadyError(result.readiness);
        throw new InvalidInputError({
          key: result.kind === "local-mode" ? "error.check.localMode" : "error.check.declined",
        });
      }
      const scope = options.evidence.scope(
        parsed.scope ? options.resolveScope(parsed.scope) : null,
        parsed.checked,
      );
      // Nothing to check against: nothing backs it, and nothing is sent.
      if (options.documents.searchableCount(scope.documentIds) === 0) {
        return { verdict: "unsupported", evidence: null, says: null, versions: [], excluded: [] };
      }
      const checksDocument =
        (parsed.checked.documentIds?.length ?? 0) + (parsed.checked.contentHashes?.length ?? 0) > 0;
      return claimCheck({
        claim: parsed.claim,
        context: parsed.context,
        cited: parsed.cited,
        scope,
        model: ready.prepared,
        // Reading the document being checked makes the Run third-party (#78).
        ...(checksDocument && { thirdParty: [{ via: "read", source: "checked-document" }] }),
        signal: new AbortController().signal,
      });
    },

    /** Stops a request: its sentences not checked yet are marked "stopped". */
    cancel(requestId: string): void {
      running.get(requestId)?.controller.abort();
    },

    /** Marks "stopped" the sentences a request that is no longer running left "checking", e.g. after a crash. */
    settleStale(mindId: string): void {
      const live = new Set(
        [...running].filter(([, run]) => run.mindId === mindId).map(([id]) => id),
      );
      const stale = content.read(mindId, (blocks) =>
        markedSentences(blocks).filter(
          ({ check }) => check.state === "checking" && !live.has(check.request),
        ),
      );
      if (stale.length === 0) return;
      const ids = stale.map(({ check }) => check.id);
      content.edit(mindId, (blocks) => {
        for (const id of ids) updateCheck(blocks, id, stopped);
      });
    },

    /** Stops every request, as IncarnaMind quits: their sentences not checked yet say "stopped". */
    stopAll(): void {
      for (const [requestId, { controller, mindId }] of running) {
        controller.abort();
        try {
          settleStopped(mindId, requestId);
        } catch (error) {
          console.error(`The check ${requestId} couldn't be marked stopped:`, error);
        }
      }
      running.clear();
    },
  };
}
