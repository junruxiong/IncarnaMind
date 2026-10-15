/**
 * Verify (#97): a PDF someone else wrote, checked claim by claim against the
 * User's Documents.
 *
 * 1. `verifyDocument` adds the PDF to the Library, or finds it there, waits
 *    until its text is read, and makes a Mind of it: a paragraph to a Block,
 *    each Block's page recorded, all of it third-party text (#78). The Mind's
 *    settings say which Document it verifies (src/shared/verification.ts).
 * 2. `findClaims` has the model name its sentences that make claims (./claims)
 *    and records them there.
 * 3. `verifyClaims` checks each claim as Check against Documents does, with
 *    the claim-check function, against the Mind's Folder, the PDF, its earlier
 *    versions and its copies left out (#80), in Runs third-party from their
 *    start. Each claim's mark shows its result as it comes; the monthly cap or
 *    a plan's limit stops the run, and the claims after it say so (#83).
 *
 * One verify run, finding or checking, goes at a time.
 */
import { randomUUID } from "node:crypto";
import { generateText } from "ai";
import { isClaim } from "../../../shared/sentences";
import { setTextOrigin } from "../../../shared/textOrigins";
import {
  readVerification,
  type Verification,
  writeVerification,
} from "../../../shared/verification";
import {
  BLOCK_ID_ATTRIBUTE,
  type ChatModelChoice,
  type ClaimCheckAttributes,
  type ClaimCheckResult,
  type Document,
  type DocumentStatus,
  type DocumentText,
  type FindClaimsResult,
  MIND_SETTINGS_FIELD,
  type Mind,
  type ProviderError,
  type VerifyClaimsResult,
  type VerifyDocumentResult,
} from "../../api";
import { parseModel } from "../../edits";
import { InvalidInputError, isRecord, NotFoundError } from "../../errors";
import type { MindContent } from "../../mindContent";
import { parseFolderChoice, parseMindId } from "../../minds";
import type { PreparedChatModel } from "../../providers/chat";
import { classifyProviderError } from "../../providers/providerErrors";
import { RunItemError, runItems } from "../../runs/stop";
import { createElement } from "../blocks";
import { answerTemperature } from "../engine";
import type { ClaimCheckRequest } from "./claimCheck";
import {
  CLAIM_BATCH,
  FIND_CLAIMS_INSTRUCTIONS,
  findClaimsPrompt,
  paragraphsOf,
  readClaimNumbers,
} from "./claims";
import { everySentence, markSentence, updateCheck } from "./sentences";

/** Claims a cloud model checks at once; a model on this computer checks one at a time. */
const CLOUD_CLAIMS_AT_ONCE = 3;

/** A Document whose text is read: its Passages can be searched and its pages read. */
const READ: readonly DocumentStatus[] = ["waiting-for-model", "embedding", "ready"];

/** The model to verify with, or why there is none (as `checkSentences` says it). */
type Prepared =
  | { ok: true; prepared: PreparedChatModel }
  | {
      ok: false;
      result: Extract<FindClaimsResult, { kind: "not-ready" | "local-mode" | "cancelled" }>;
    };

export interface VerifyOptions {
  content: Pick<MindContent, "read" | "edit" | "readSettings">;
  /** Throws unless the Mind exists and this version may write to it. */
  requireMind(mindId: string): void;
  /** Throws unless the Folder exists. */
  requireFolder(folderId: string): void;
  /** Adds the file to the Library, new ones filed in the Folder; the Document it is, or null when it wasn't added. */
  addFile(path: string, folderId: string | null): Promise<Document | null>;
  /** Throws NotFoundError for a Document that is gone. */
  document(documentId: string): Document;
  onStatus(listener: (document: Document) => void): () => void;
  readText(documentId: string): DocumentText;
  createMind(title: string, folderId: string | null): Mind;
  defaultModel(): ChatModelChoice | null;
  prepare(choice: ChatModelChoice | null): Promise<Prepared>;
  /** Where the Mind's claims are checked: no Documents, or how many, and the scope. */
  evidenceScope(
    mindId: string,
    allDocuments: boolean,
    verification: Verification,
  ): { searchable: number; folderId: string | null; check: CheckScope };
  claimCheck(request: ClaimCheckRequest): Promise<ClaimCheckResult>;
  withResult(
    check: ClaimCheckAttributes,
    result: ClaimCheckResult,
    at: string,
  ): ClaimCheckAttributes;
  /** Keeps the text of the evidence the Mind's checks show (#79). */
  retainEvidence(mindId: string): void;
  /** Marks a request's sentences not checked yet "stopped". */
  settleStopped(mindId: string, request: string): void;
  /** Records a request as running, so `cancelCheck` stops it; `done` when it ends. */
  track(requestId: string, mindId: string): { signal: AbortSignal; done(): void };
  now(): Date;
}

/** What a claim check of the Mind searches, and how its marks record it. */
export interface CheckScope {
  scope: ClaimCheckRequest["scope"];
  marked: ClaimCheckAttributes["scope"];
}

function parseRequest(input: unknown) {
  if (!isRecord(input)) throw new InvalidInputError({ key: "error.verify.request" });
  const { requestId } = input;
  if (typeof requestId !== "string" || requestId === "") {
    throw new InvalidInputError({ key: "error.verify.request" });
  }
  return {
    requestId,
    mindId: parseMindId(input.mindId),
    allDocuments: input.allDocuments === true,
    model: parseModel(input.model),
  };
}

/** A claim check's failure that the run's other claims share: waited out, or it stops the run. */
function runFailureOf(result: ClaimCheckResult): ProviderError | null {
  if (result.verdict !== "not-checked") return null;
  const { reason } = result;
  if (reason !== "rate-limit" && reason !== "cost-cap" && reason !== "plan-limit") return null;
  return { kind: reason, message: result.detail ?? reason };
}

export function createVerify(options: VerifyOptions) {
  const { content } = options;
  /** The verify run going now: one at a time. */
  let current: { requestId: string; mindId: string } | null = null;

  /** Resolves once the Document's text is read: true, or false when it can't be (no text, or it failed). */
  const textRead = (documentId: string) =>
    new Promise<boolean>((resolve) => {
      const settled = (document: Document | null): boolean => {
        const status = document?.status ?? null;
        if (status === "queued" || status === "extracting") return false;
        stop();
        resolve(status !== null && READ.includes(status));
        return true;
      };
      const stop = options.onStatus((document) => {
        if (document.id === documentId) settled(document);
      });
      try {
        settled(options.document(documentId));
      } catch (error) {
        if (!(error instanceof NotFoundError)) {
          stop();
          throw error;
        }
        settled(null);
      }
    });

  const verificationOf = (mindId: string): Verification => {
    const verification = content.readSettings(mindId, readVerification);
    if (!verification) throw new InvalidInputError({ key: "error.verify.notVerifying" });
    return verification;
  };

  /** Runs one verify request at a time, recorded so `cancelCheck` stops it. */
  async function oneAtATime<T>(
    requestId: string,
    mindId: string,
    run: (signal: AbortSignal) => Promise<T>,
  ): Promise<T | { kind: "busy"; mindId: string }> {
    if (current) return { kind: "busy", mindId: current.mindId };
    current = { requestId, mindId };
    const tracked = options.track(requestId, mindId);
    try {
      return await run(tracked.signal);
    } finally {
      tracked.done();
      current = null;
    }
  }

  async function findClaims(
    mindId: string,
    model: ChatModelChoice | null,
    signal: AbortSignal,
  ): Promise<FindClaimsResult> {
    const verification = verificationOf(mindId);
    const { searchable, folderId } = options.evidenceScope(mindId, false, verification);
    if (searchable === 0) return { kind: "no-documents", folderId };
    const ready = await options.prepare(model ?? options.defaultModel());
    if (!ready.ok) return ready.result;
    if (signal.aborted) return { kind: "cancelled" };

    const sentences = content.read(mindId, (blocks) =>
      everySentence(blocks)
        .filter((each) => each.blockId in verification.pages && isClaim(each.text))
        .map(({ blockId, text }) => ({ blockId, text })),
    );
    const { prepared } = ready;
    const named = new Set<number>();
    for (let start = 0; start < sentences.length; start += CLAIM_BATCH) {
      const batch = sentences
        .slice(start, start + CLAIM_BATCH)
        .map((each, index) => ({ number: start + index + 1, text: each.text }));
      let numbers: number[] | null = null;
      // A reply that names nothing readable is asked for once more.
      for (let tries = 0; tries < 2 && numbers === null; tries++) {
        try {
          const { text } = await generateText({
            model: prepared.model,
            instructions: FIND_CLAIMS_INSTRUCTIONS,
            prompt: findClaimsPrompt(batch),
            temperature: answerTemperature(prepared.capabilities.facts.temperature),
            maxOutputTokens: 4_000,
            maxRetries: 1,
            abortSignal: signal,
          });
          numbers = readClaimNumbers(text);
        } catch (error) {
          if (signal.aborted) return { kind: "cancelled" };
          return { kind: "failed", error: classifyProviderError(error) };
        }
      }
      if (numbers === null) {
        return {
          kind: "failed",
          error: { kind: "no-answer", message: "The model's reply named no sentences." },
        };
      }
      const first = start + 1;
      const last = start + batch.length;
      for (const number of numbers) if (number >= first && number <= last) named.add(number);
    }
    if (signal.aborted) return { kind: "cancelled" };
    const claims = sentences.filter((_, index) => named.has(index + 1));
    content.edit(mindId, (blocks) => {
      const settings = blocks.doc?.getMap<unknown>(MIND_SETTINGS_FIELD);
      const now = settings && readVerification(settings);
      if (settings && now) writeVerification(settings, { ...now, claims });
    });
    return { kind: "found", claims: claims.length };
  }

  async function verifyClaims(
    input: ReturnType<typeof parseRequest>,
    signal: AbortSignal,
  ): Promise<VerifyClaimsResult> {
    const { mindId, requestId } = input;
    const verification = verificationOf(mindId);
    if (!verification.claims || verification.claims.length === 0) return { kind: "nothing" };
    const { searchable, folderId, check } = options.evidenceScope(
      mindId,
      input.allDocuments,
      verification,
    );
    if (searchable === 0) return { kind: "no-documents", folderId };
    const ready = await options.prepare(input.model ?? options.defaultModel());
    if (!ready.ok) return ready.result;
    if (signal.aborted) return { kind: "cancelled" };

    const startedAt = options.now().toISOString();
    const claims = verification.claims;
    const queued = content.edit(mindId, (blocks) => {
      const sentences = everySentence(blocks);
      return claims.flatMap((claim) => {
        const sentence = sentences.find(
          (each) => each.blockId === claim.blockId && each.text === claim.text,
        );
        if (!sentence) return [];
        const marked: ClaimCheckAttributes = {
          id: randomUUID(),
          request: requestId,
          action: "check-against-documents",
          state: "checking",
          reason: null,
          detail: null,
          text: sentence.text,
          evidence: null,
          says: null,
          calculation: null,
          scope: check.marked,
          versions: [],
          excluded: [],
          checkedAt: startedAt,
        };
        markSentence(sentence.element, sentence.from, sentence.to, marked);
        return [{ id: marked.id, claim: sentence.text, context: sentence.context }];
      });
    });
    if (queued.length === 0) return { kind: "nothing" };

    const write = (id: string, change: (check: ClaimCheckAttributes) => ClaimCheckAttributes) => {
      if (signal.aborted) return;
      const kept = content.edit(mindId, (blocks) => updateCheck(blocks, id, change));
      if (kept) options.retainEvidence(mindId);
    };
    const notChecked =
      (reason: ClaimCheckAttributes["reason"], detail: string | null = null) =>
      (check: ClaimCheckAttributes): ClaimCheckAttributes => ({
        ...check,
        state: "not-checked",
        reason,
        detail,
        checkedAt: options.now().toISOString(),
      });

    let checked = 0;
    const { prepared } = ready;
    const { stop } = await runItems(
      queued,
      async (claim) => {
        const result = await options.claimCheck({
          claim: claim.claim,
          context: claim.context,
          scope: check.scope,
          model: prepared,
          thirdParty: [{ via: "read", source: "checked-document" }],
          cacheKey: mindId,
          signal,
        });
        const failure = runFailureOf(result);
        if (failure) throw new RunItemError(failure);
        return result;
      },
      {
        concurrency: prepared.provider.service ? CLOUD_CLAIMS_AT_ONCE : 1,
        signal,
        onOutcome(claim, outcome) {
          if (outcome.status === "done") {
            const at = options.now().toISOString();
            if (outcome.value.verdict !== "not-checked") checked++;
            write(claim.id, (each) => options.withResult(each, outcome.value, at));
          } else if (outcome.status === "not-checked") {
            write(claim.id, notChecked(outcome.stop.reason));
          } else if (outcome.status === "failed") {
            const rateLimited = outcome.error.kind === "rate-limit";
            write(
              claim.id,
              notChecked(
                rateLimited ? "rate-limit" : "failed",
                rateLimited ? null : outcome.error.message,
              ),
            );
          }
        },
      },
    );
    if (signal.aborted) {
      options.settleStopped(mindId, requestId);
      return { kind: "cancelled" };
    }
    return { kind: "checked", claims: queued.length, checked, stop };
  }

  return {
    async verifyDocument(input: unknown): Promise<VerifyDocumentResult> {
      if (!isRecord(input)) throw new InvalidInputError({ key: "error.verify.request" });
      const { path } = input;
      if (typeof path !== "string" || path.trim() === "") {
        throw new InvalidInputError({ key: "error.verify.noPath" });
      }
      const folderId = parseFolderChoice(input.folderId);
      if (folderId !== null) options.requireFolder(folderId);

      const document = await options.addFile(path, folderId);
      if (!document) return { ok: false, reason: "unreadable" };
      if (!(await textRead(document.id))) {
        const now = options.document(document.id);
        return { ok: false, reason: now.status === "no-text" ? "no-text" : "unreadable" };
      }
      const text = options.readText(document.id);
      const paragraphs = paragraphsOf(text.pages);
      if (paragraphs.length === 0) return { ok: false, reason: "no-text" };

      const { name } = options.document(document.id);
      const mind = options.createMind(name, folderId);
      content.edit(mind.id, (blocks) => {
        const doc = blocks.doc;
        if (!doc) return;
        const elements = paragraphs.map(({ text, heading }) =>
          createElement({
            type: heading ? "heading" : "paragraph",
            ...(heading && { attrs: { level: 2 } }),
            content: [{ type: "text", text }],
          }),
        );
        blocks.insert(0, elements);
        const pages: Record<string, number> = {};
        elements.forEach((element, index) => {
          const id = element.getAttribute(BLOCK_ID_ATTRIBUTE);
          const page = paragraphs[index]?.page;
          if (typeof id === "string" && page !== undefined) pages[id] = page;
        });
        setTextOrigin(doc, Object.keys(pages), "checked-document");
        writeVerification(doc.getMap(MIND_SETTINGS_FIELD), {
          documentId: document.id,
          documentName: name,
          pages,
          claims: null,
        });
      });
      return { ok: true, mind, documentId: document.id };
    },

    async findClaims(input: unknown): Promise<FindClaimsResult> {
      const parsed = parseRequest(input);
      options.requireMind(parsed.mindId);
      return oneAtATime(parsed.requestId, parsed.mindId, (signal) =>
        findClaims(parsed.mindId, parsed.model, signal),
      );
    },

    async verifyClaims(input: unknown): Promise<VerifyClaimsResult> {
      const parsed = parseRequest(input);
      options.requireMind(parsed.mindId);
      return oneAtATime(parsed.requestId, parsed.mindId, (signal) => verifyClaims(parsed, signal));
    },
  };
}
