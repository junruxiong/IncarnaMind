/**
 * What the assistant writes into a Mind, checked before it lands (#1111).
 *
 * Each sentence it wrote that has Citations is checked as Check this Mind
 * would check it (./claimCheck: the Mind's Folder, the Citations' quotes
 * sent), a few at a time. The words are still the assistant's changes then,
 * not the Mind's.
 *
 * - A part its Citations don't support, as the check names it, is taken out
 *   with Edit with AI's own request (`cut-unsupported`), and the Citations
 *   stay on what is left. A cut that would lose every Citation, or add one,
 *   isn't used.
 * - A sentence still unsupported, or one a Document contradicts, lands with
 *   its check's mark, so Check this Mind doesn't find it again later.
 * - A supported sentence lands unmarked: its Citations' own checks stand for
 *   it, as after Add as Citation.
 *
 * One check per sentence, and one edit per sentence it cuts, on the Answer's
 * model. Nothing is checked when the User has turned the checks flow off, or
 * on a model on this computer too small to judge reliably (#680).
 */
import { randomUUID } from "node:crypto";
import { INLINE_NODE, sentenceSpans } from "../../../shared/sentences";
import { judgesUnreliably } from "../../../shared/smallModel";
import {
  CITATION_NODE,
  type CitedQuote,
  CLAIM_CHECK_MARK,
  type ClaimCheckAttributes,
  type ClaimCheckResult,
  type EditAction,
  type RecordAiUseInput,
} from "../../api";
import type { PreparedChatModel } from "../../providers/chat";
import { RunItemError, runItems } from "../../runs/stop";
import type { MarkJSON, NodeJSON } from "../blocks";
import { type ClaimCheckRequest, recordJudged, runFailureOf } from "./claimCheck";
import type { CheckScope } from "./verify";

const CLOUD_SENTENCES_AT_ONCE = 3;

const UNCHECKED: ReadonlySet<string> = new Set(["question", "codeBlock", "blockMath", "heading"]);

export interface WrittenCheckOptions {
  /** Whether the checks flow may send to the model's service: never once the User turned it off (ADR-0022). */
  allowed(prepared: PreparedChatModel): boolean;
  claimCheck(request: ClaimCheckRequest): Promise<ClaimCheckResult>;
  /** Where the Mind's sentences are checked, as Check this Mind checks them. */
  scopeOf(mindId: string): { searchable: number; check: CheckScope };
  /** What the Citations quote, each Document named as it is now. */
  citedNow(cited: readonly CitedQuote[]): { documentName: string; quote: string }[];
  withResult(
    check: ClaimCheckAttributes,
    result: ClaimCheckResult,
    at: string,
  ): ClaimCheckAttributes;
  retainEvidence(mindId: string): void;
  recordAiUse(inputs: RecordAiUseInput[]): void;
  /** Edit with AI's new text for a passage not in a Mind (../../edits, `editText`). */
  editText(
    prepared: PreparedChatModel,
    ask: { action: EditAction; instruction: string },
    passage: { text: string; context: string },
    signal: AbortSignal,
  ): Promise<string | null>;
  now(): Date;
}

export interface CheckWrittenInput {
  mindId: string;
  /** The Blocks as the assistant wrote them. */
  blocks: readonly NodeJSON[];
  prepared: PreparedChatModel;
  signal: AbortSignal;
  /** How many sentences are being checked, as the check starts. */
  checking(count: number): void;
  /** Puts the Blocks as they land, cut and marked, in place of those written; only when they changed. */
  write(blocks: NodeJSON[]): void;
}

/** What the check did: how many sentences it checked, each part it cut as it named it, and how many land marked. */
export interface WrittenCheck {
  sentences: number;
  cut: string[];
  unsupported: number;
  contradicted: number;
}

interface Sentence {
  /** An index into each level's content, from the top-level Block down to its paragraph. */
  path: number[];
  from: number;
  to: number;
  /** As checks read it: an inline node as U+FFFC, a line break as "\n". */
  text: string;
  context: string;
  cited: CitedQuote[];
}

function paragraphText(content: readonly NodeJSON[]): string {
  return content
    .map((node) =>
      node.type === "text" ? (node.text ?? "") : node.type === "hardBreak" ? "\n" : INLINE_NODE,
    )
    .join("");
}

function inlineLayout(content: readonly NodeJSON[]): { node: NodeJSON; start: number }[] {
  let at = 0;
  return content.map((node) => {
    const start = at;
    at += node.type === "text" ? (node.text ?? "").length : 1;
    return { node, start };
  });
}

function quoteOf(node: NodeJSON): CitedQuote | null {
  const attrs = node.attrs ?? {};
  const { quote, documentName, documentId, contentHash } = attrs;
  if (typeof quote !== "string" || quote === "") return null;
  return {
    documentName: typeof documentName === "string" ? documentName : "",
    quote,
    ...(typeof documentId === "string" && { documentId }),
    ...(typeof contentHash === "string" && { contentHash }),
  };
}

function citedSentences(blocks: readonly NodeJSON[]): Sentence[] {
  const found: Sentence[] = [];
  const visit = (nodes: readonly NodeJSON[], path: number[]) => {
    nodes.forEach((node, index) => {
      if (UNCHECKED.has(node.type)) return;
      const here = [...path, index];
      const content = node.content ?? [];
      if (node.type !== "paragraph") {
        visit(content, here);
        return;
      }
      const context = paragraphText(content);
      const layout = inlineLayout(content);
      for (const span of sentenceSpans(context)) {
        const cited = layout.flatMap(({ node: inline, start }) => {
          if (inline.type !== CITATION_NODE || start < span.from || start >= span.to) return [];
          return quoteOf(inline) ?? [];
        });
        if (cited.length === 0) continue;
        found.push({
          path: here,
          from: span.from,
          to: span.to,
          text: context.slice(span.from, span.to),
          context,
          cited,
        });
      }
    });
  };
  visit(blocks, []);
  return found;
}

const MARKER = /\[(\d{1,3})\]/g;

/** Null when the sentence holds an inline node a marker can't stand for (a formula, a footnote), which a cut would lose. */
function asMarked(
  content: readonly NodeJSON[],
  from: number,
  to: number,
): { passage: string; context: string; citations: Map<number, NodeJSON> } | null {
  const citations = new Map<number, NodeJSON>();
  let context = "";
  let passage = "";
  for (const { node, start } of inlineLayout(content)) {
    let piece: string;
    if (node.type === "text") piece = node.text ?? "";
    else if (node.type === "hardBreak") piece = "\n";
    else if (node.type === CITATION_NODE) {
      citations.set(citations.size + 1, node);
      piece = `[${citations.size}]`;
    } else if (start >= from && start < to) return null;
    else piece = "";
    context += piece;
    if (node.type === "text") {
      const text = node.text ?? "";
      passage += text.slice(Math.max(0, from - start), Math.max(0, to - start));
    } else if (start >= from && start < to) passage += piece;
  }
  return { passage: passage.trim(), context, citations };
}

/** Null when the cut keeps no Citation, adds or repeats one, or says nothing. */
function cutNodes(
  cut: string,
  sentence: readonly number[],
  citations: ReadonlyMap<number, NodeJSON>,
  marks: MarkJSON[] | undefined,
): NodeJSON[] | null {
  const used = new Set<number>();
  const nodes: NodeJSON[] = [];
  let last = 0;
  for (const match of cut.matchAll(MARKER)) {
    const number = Number(match[1]);
    const citation = citations.get(number);
    if (!citation || !sentence.includes(number) || used.has(number)) return null;
    used.add(number);
    const at = match.index ?? 0;
    const words = cut.slice(last, at);
    if (words !== "") nodes.push({ type: "text", text: words, ...(marks && { marks }) });
    nodes.push(citation);
    last = at + match[0].length;
  }
  const rest = cut.slice(last);
  if (rest !== "") nodes.push({ type: "text", text: rest, ...(marks && { marks }) });
  const said = cut.replace(MARKER, "").trim();
  return used.size === 0 || !/[\p{L}\p{N}]/u.test(said) ? null : nodes;
}

interface Landing {
  sentence: Sentence;
  /** In place of the sentence's own; null: it stays as written. */
  nodes: NodeJSON[] | null;
  mark: ClaimCheckAttributes | null;
}

function landParagraph(content: readonly NodeJSON[], landings: readonly Landing[]): NodeJSON[] {
  const sorted = [...landings].sort((a, b) => a.sentence.from - b.sentence.from);
  const landingAt = (offset: number) =>
    sorted.find(({ sentence }) => sentence.from <= offset && offset < sentence.to);
  const out: NodeJSON[] = [];
  const placed = new Set<Landing>();
  for (const { node, start } of inlineLayout(content)) {
    if (node.type !== "text") {
      const landing = landingAt(start);
      if (!landing?.nodes) out.push(node);
      else if (!placed.has(landing)) {
        placed.add(landing);
        out.push(...landing.nodes);
      }
      continue;
    }
    const text = node.text ?? "";
    const cuts = new Set([0, text.length]);
    for (const { sentence } of sorted) {
      for (const edge of [sentence.from - start, sentence.to - start]) {
        if (edge > 0 && edge < text.length) cuts.add(edge);
      }
    }
    const edges = [...cuts].sort((a, b) => a - b);
    for (let index = 0; index < edges.length - 1; index++) {
      const begin = edges[index] ?? 0;
      const end = edges[index + 1] ?? text.length;
      const piece = text.slice(begin, end);
      const landing = landingAt(start + begin);
      if (landing?.nodes) {
        if (!placed.has(landing)) {
          placed.add(landing);
          out.push(...landing.nodes);
        }
        continue;
      }
      const marks = landing?.mark
        ? [
            ...(node.marks ?? []).filter((mark) => mark.type !== CLAIM_CHECK_MARK),
            { type: CLAIM_CHECK_MARK, attrs: { ...landing.mark } },
          ]
        : node.marks;
      out.push({ ...node, text: piece, ...(marks && { marks }) });
    }
  }
  return out;
}

function land(blocks: readonly NodeJSON[], landings: readonly Landing[]): NodeJSON[] {
  const byPath = new Map<string, Landing[]>();
  for (const landing of landings) {
    const key = landing.sentence.path.join(".");
    byPath.set(key, [...(byPath.get(key) ?? []), landing]);
  }
  const visit = (nodes: readonly NodeJSON[], path: number[]): NodeJSON[] =>
    nodes.map((node, index) => {
      const here = [...path, index];
      const key = here.join(".");
      const mine = byPath.get(key);
      if (mine) return { ...node, content: landParagraph(node.content ?? [], mine) };
      const deeper = [...byPath.keys()].some((each) => each.startsWith(`${key}.`));
      return deeper && node.content ? { ...node, content: visit(node.content, here) } : node;
    });
  return visit(blocks, []);
}

function paragraphAt(blocks: readonly NodeJSON[], path: readonly number[]): NodeJSON | undefined {
  let nodes: readonly NodeJSON[] | undefined = blocks;
  let node: NodeJSON | undefined;
  for (const index of path) {
    node = nodes?.[index];
    nodes = node?.content;
  }
  return node;
}

export function createWrittenCheck(options: WrittenCheckOptions) {
  async function cutOf(
    blocks: readonly NodeJSON[],
    sentence: Sentence,
    uncovered: string,
    prepared: PreparedChatModel,
    signal: AbortSignal,
  ): Promise<NodeJSON[] | null> {
    const content = paragraphAt(blocks, sentence.path)?.content ?? [];
    const marked = asMarked(content, sentence.from, sentence.to);
    if (!marked) return null;
    const numbers = [...marked.passage.matchAll(MARKER)].map((match) => Number(match[1]));
    const cut = await options.editText(
      prepared,
      { action: "cut-unsupported", instruction: uncovered },
      { text: marked.passage, context: marked.context },
      signal,
    );
    if (cut === null || cut.replace(/\s+/g, " ") === marked.passage.replace(/\s+/g, " ")) {
      return null;
    }
    const first = inlineLayout(content).find(
      ({ node, start }) =>
        node.type === "text" &&
        start + (node.text ?? "").length > sentence.from &&
        start < sentence.to,
    );
    return cutNodes(cut.trim(), numbers, marked.citations, first?.node.marks);
  }

  /** Null when there was nothing to check, or it couldn't be checked here. */
  return async function checkWritten(input: CheckWrittenInput): Promise<WrittenCheck | null> {
    const { mindId, blocks, prepared, signal } = input;
    if (judgesUnreliably(prepared) || !options.allowed(prepared)) return null;
    const sentences = citedSentences(blocks);
    if (sentences.length === 0) return null;
    const { searchable, check: scope } = options.scopeOf(mindId);
    if (searchable === 0) return null;
    input.checking(sentences.length);
    const request = randomUUID();
    const startedAt = options.now().toISOString();
    const { outcomes } = await runItems(
      sentences,
      async (sentence) => {
        const result = await options.claimCheck({
          claim: sentence.text,
          context: sentence.context,
          cited: options.citedNow(sentence.cited),
          scope: scope.scope,
          model: prepared,
          cacheKey: mindId,
          signal,
        });
        const failure = runFailureOf(result);
        if (failure) throw new RunItemError(failure);
        const uncovered =
          result.verdict === "unsupported" && result.unsupported?.covered
            ? result.unsupported.uncovered
            : null;
        const nodes = uncovered ? await cutOf(blocks, sentence, uncovered, prepared, signal) : null;
        return { result, uncovered, nodes };
      },
      { concurrency: prepared.provider.service ? CLOUD_SENTENCES_AT_ONCE : 1, signal },
    );
    if (signal.aborted) return null;

    const checked: WrittenCheck = { sentences: 0, cut: [], unsupported: 0, contradicted: 0 };
    const landings: Landing[] = [];
    sentences.forEach((sentence, index) => {
      const outcome = outcomes[index];
      if (outcome?.status === "failed") {
        console.warn(
          `A sentence the assistant wrote in Mind ${mindId} couldn't be checked before it landed: ${outcome.error.message}`,
        );
      }
      if (outcome?.status !== "done") return;
      const { result, uncovered, nodes } = outcome.value;
      const at = options.now().toISOString();
      recordJudged(options.recordAiUse, mindId, prepared, result, at);
      if (result.verdict === "not-checked") return;
      checked.sentences++;
      if (nodes && uncovered) {
        checked.cut.push(uncovered);
        landings.push({ sentence, nodes, mark: null });
        return;
      }
      if (result.verdict !== "unsupported" && result.verdict !== "contradicted") return;
      checked[result.verdict]++;
      const mark: ClaimCheckAttributes = {
        id: randomUUID(),
        request,
        action: "check-against-documents",
        state: "checking",
        reason: null,
        detail: null,
        providerId: prepared.provider.id,
        modelId: prepared.modelId,
        text: sentence.text,
        evidence: null,
        says: null,
        unsupported: null,
        calculation: null,
        citedCaseUnchecked: false,
        citedCaseFound: false,
        scope: scope.marked,
        versions: [],
        excluded: [],
        checkedAt: startedAt,
      };
      landings.push({ sentence, nodes: null, mark: options.withResult(mark, result, at) });
    });
    if (landings.length > 0) {
      input.write(land(blocks, landings));
      if (landings.some((each) => each.mark)) options.retainEvidence(mindId);
    }
    return checked;
  };
}
