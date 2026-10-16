/**
 * Check this Mind (#737): every sentence of a Mind that makes a claim,
 * checked in one run, as Verify checks someone else's PDF (#97).
 *
 * 1. The Mind's sentences are read, as for note health: those in paragraphs,
 *    lists, quotes and Answers, not headings or Questions.
 * 2. The model names the claims among them, leaving out questions and the
 *    User's own opinions (./findClaims). In "changed" mode it is shown only
 *    the sentences never checked since the last run, and the checks that
 *    need a re-check are made again without asking it.
 * 3. Each claim gets a "checking" mark at once, then its result, as Check
 *    against Documents gives it (./runChecks).
 * 4. The run is recorded in the Mind's settings (`MindCheckRun`): when, how
 *    many sentences have a check, and which sentences make no claim, so the
 *    next run doesn't send them again.
 */
import { randomUUID } from "node:crypto";
import { claimCheckShown } from "../../../shared/claimChecks";
import { readMindCheck, sentenceKey, writeMindCheck } from "../../../shared/mindCheck";
import { mindModelOf } from "../../../shared/mindModel";
import { isClaim } from "../../../shared/sentences";
import { judgesUnreliably } from "../../../shared/smallModel";
import type {
  ChatModelChoice,
  CheckMindInput,
  CheckMindMode,
  CheckMindResult,
  ClaimCheckAttributes,
  MindCheckPlan,
} from "../../api";
import { parseModel } from "../../edits";
import { InvalidInputError, isRecord } from "../../errors";
import type { MindContent } from "../../mindContent";
import { parseMindId } from "../../minds";
import { claimsAmong } from "./findClaims";
import { type QueuedClaim, type RunChecksOptions, runChecks } from "./runChecks";
import { everySentence, markedSentences, markSentence, type SelectedSentence } from "./sentences";
import type { CheckScope, VerifyOptions } from "./verify";

export interface MindCheckOptions extends RunChecksOptions {
  content: Pick<MindContent, "read" | "edit" | "readSettings" | "editSettings">;
  defaultModel(): ChatModelChoice | null;
  prepare: VerifyOptions["prepare"];
  unread: VerifyOptions["unread"];
  recordUnread: VerifyOptions["recordUnread"];
  /** Where the Mind's sentences are checked: no Documents, or how many, and the scope. */
  scopeOf(
    mindId: string,
    allDocuments: boolean,
  ): { searchable: number; folderId: string | null; check: CheckScope };
  /** The Folder the Mind is in; null for none. */
  mindFolder(mindId: string): string | null;
  /** Each live Document's version now, for which checks need a re-check. */
  documentVersions(): ReadonlyMap<string, string>;
  settleStopped(mindId: string, request: string): void;
  track(requestId: string, mindId: string): { signal: AbortSignal; done(): void };
  isRunning(requestId: string): boolean;
  /** Throws unless the Mind exists and this version may write to it. */
  requireMind(mindId: string): void;
}

const MODES: readonly CheckMindMode[] = ["all", "changed"];

function parseCheckMind(input: unknown): Required<Omit<CheckMindInput, "model">> & {
  model: ChatModelChoice | null;
} {
  if (!isRecord(input)) throw new InvalidInputError({ key: "error.mindCheck.request" });
  const { requestId } = input;
  if (typeof requestId !== "string" || requestId === "") {
    throw new InvalidInputError({ key: "error.mindCheck.request" });
  }
  return {
    requestId,
    mindId: parseMindId(input.mindId),
    mode: parseMode(input.mode),
    allDocuments: input.allDocuments === true,
    model: parseModel(input.model),
  };
}

function parseMode(mode: unknown): CheckMindMode {
  const known = MODES.find((each) => each === mode);
  if (!known) throw new InvalidInputError({ key: "error.mindCheck.request" });
  return known;
}

/** What a run sends: sentences to find the claims among, and sentences whose checks are made again. */
interface Sending {
  fresh: SelectedSentence[];
  again: SelectedSentence[];
  /** Every sentence that could make a claim now, by key: what the run's skipped list keeps to. */
  keys: Set<string>;
  skipped: Set<string>;
}

export function createMindCheck(options: MindCheckOptions) {
  const { content } = options;

  const modelOf = (mindId: string): ChatModelChoice | null =>
    content.readSettings(mindId, (settings) => (settings.doc ? mindModelOf(settings.doc) : null));

  /** The sentences a run in `mode` sends now. */
  function sending(mindId: string, mode: CheckMindMode): Sending {
    const skipped = new Set(content.readSettings(mindId, readMindCheck)?.skipped ?? []);
    const versions = options.documentVersions();
    const folderId = options.mindFolder(mindId);
    return content.read(mindId, (blocks) => {
      const sentences = everySentence(blocks).filter((sentence) => isClaim(sentence.text));
      const keys = new Set(sentences.map((sentence) => sentenceKey(sentence.text)));
      if (mode === "all") return { fresh: sentences, again: [], keys, skipped: new Set() };
      const marks = markedSentences(blocks);
      const markOf = (sentence: SelectedSentence) =>
        marks.find(
          (mark) =>
            mark.element === sentence.element && mark.from < sentence.to && mark.to > sentence.from,
        );
      const fresh: SelectedSentence[] = [];
      const again: SelectedSentence[] = [];
      for (const sentence of sentences) {
        const mark = markOf(sentence);
        if (!mark) {
          if (!skipped.has(sentenceKey(sentence.text))) fresh.push(sentence);
          continue;
        }
        const shown = claimCheckShown(mark.check, { text: sentence.text, versions, folderId });
        if (shown.state === "needs-recheck") again.push(sentence);
      }
      return { fresh, again, keys, skipped };
    });
  }

  function plan(mindId: string, mode: CheckMindMode, allDocuments: boolean): MindCheckPlan {
    const { fresh, again } = sending(mindId, mode);
    if (fresh.length + again.length === 0) return { kind: "nothing" };
    const { searchable, folderId } = options.scopeOf(mindId, allDocuments);
    if (searchable === 0) return { kind: "no-documents", folderId };
    return {
      kind: "ready",
      sentences: fresh.length + again.length,
      findsClaims: fresh.length > 0,
      model: modelOf(mindId),
    };
  }

  async function check(
    input: ReturnType<typeof parseCheckMind>,
    signal: AbortSignal,
  ): Promise<CheckMindResult> {
    const { mindId, requestId, mode } = input;
    const { fresh, again, keys, skipped } = sending(mindId, mode);
    if (fresh.length + again.length === 0) return { kind: "nothing" };
    const { searchable, folderId, check: scope } = options.scopeOf(mindId, input.allDocuments);
    if (searchable === 0) return { kind: "no-documents", folderId };
    const ready = await options.prepare(input.model ?? modelOf(mindId) ?? options.defaultModel());
    if (!ready.ok) return ready.result;
    if (signal.aborted) return { kind: "cancelled" };

    let claims = again;
    if (fresh.length > 0) {
      const found = await claimsAmong(
        ready.prepared,
        fresh.map((sentence) => sentence.text),
        "own",
        signal,
      );
      if (found.kind !== "found") return found;
      for (const [index, sentence] of fresh.entries()) {
        if (!found.claims.has(index)) skipped.add(sentenceKey(sentence.text));
      }
      claims = [...fresh.filter((_, index) => found.claims.has(index)), ...again];
    }

    const startedAt = options.now().toISOString();
    const unread = options.unread(scope.wanted);
    // The User may have written on while the model named the claims: each is found again by its words.
    const queued = content.edit(mindId, (blocks) => {
      const now = everySentence(blocks);
      const marked = claims.flatMap((claim): QueuedClaim[] => {
        const sentence = now.find(
          (each) => each.blockId === claim.blockId && each.text === claim.text,
        );
        if (!sentence) return [];
        const mark: ClaimCheckAttributes = {
          id: randomUUID(),
          request: requestId,
          action: "check-against-documents",
          state: "checking",
          reason: null,
          detail: null,
          providerId: ready.prepared.provider.id,
          modelId: ready.prepared.modelId,
          text: sentence.text,
          evidence: null,
          says: null,
          calculation: null,
          citedCaseUnchecked: false,
          citedCaseFound: false,
          scope: scope.marked,
          versions: [],
          excluded: [],
          checkedAt: startedAt,
        };
        markSentence(sentence.element, sentence.from, sentence.to, mark);
        return [
          { id: mark.id, claim: sentence.text, context: sentence.context, cited: sentence.cited },
        ];
      });
      if (marked.length > 0) options.recordUnread(blocks, requestId, unread);
      return marked;
    });

    const { checked, stop } = await runChecks(options, {
      mindId,
      queued,
      scope: scope.scope,
      prepared: ready.prepared,
      signal,
    });
    if (signal.aborted) {
      options.settleStopped(mindId, requestId);
      return { kind: "cancelled" };
    }
    const sentences = content.read(mindId, (blocks) => markedSentences(blocks).length);
    content.editSettings(mindId, (settings) =>
      writeMindCheck(settings, {
        at: options.now().toISOString(),
        sentences,
        stop,
        skipped: [...skipped].filter((key) => keys.has(key)),
      }),
    );
    return {
      kind: "checked",
      sentences: queued.length,
      checked,
      smallModel: judgesUnreliably(ready.prepared),
      stop,
    };
  }

  return {
    /** See `CoreApi.planMindCheck`. */
    planMindCheck(mindId: unknown, mode: unknown, allDocuments: boolean): MindCheckPlan {
      const id = parseMindId(mindId);
      options.requireMind(id);
      return plan(id, parseMode(mode), allDocuments);
    },

    /** See `CoreApi.checkMind`. */
    async checkMind(input: unknown): Promise<CheckMindResult> {
      const parsed = parseCheckMind(input);
      options.requireMind(parsed.mindId);
      if (options.isRunning(parsed.requestId)) {
        throw new InvalidInputError({ key: "error.mindCheck.request" });
      }
      const tracked = options.track(parsed.requestId, parsed.mindId);
      try {
        return await check(parsed, tracked.signal);
      } finally {
        tracked.done();
      }
    },
  };
}
