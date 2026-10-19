/**
 * The guards every everyday request in the assistant is held to, read from
 * what the core stored, as a User would see it:
 *
 * - **Asked to write, the Mind changed.**
 * - **Every claimed change is real**, as a model judges the reply against the
 *   record of what the calls did (./judge), never the app's own net.
 * - **Citations are there**: in the reply, or, for text written into a Mind,
 *   in the Mind, and no source there is written as text, like
 *   "[Deck, slide 3]" (#1075).
 * - **No marker without a Citation**: every Citation marker the Answer's text
 *   shows has its Citation, whatever the request.
 * - **No tool-call text**: nothing in the Answer's text is a Tool's call
 *   written as words, such as a `<cite>` block.
 * - **Every Document is covered**: asked about all the Documents in scope,
 *   each is cited, or named in a sentence saying it wasn't covered.
 * - **Only what was asked changes**: asked about one paragraph, the others
 *   read as they did.
 * - **Changed in place**: asked to make text in the Mind longer, it changes
 *   that text, not adding a second version beside it, nor leaving it as it
 *   was (#1061).
 * - **The Question asked is the Answer's subject**: asked after the example's
 *   Answer, nothing in the reply is about an earlier Answer or the example.
 * - **No word of an attempt the User never saw**: when the app asks the model
 *   again inside a turn, the reply says nothing of the earlier try (#1226); a
 *   reply that was not asked again may speak of a message the User did read.
 * - **Only the text's dates**: every date the reply gives is one the
 *   Documents' text writes, or is named as a file's (./dates).
 */
import { changeLogOf } from "../../src/core/answers/claimedChanges";
import { toolTextIn } from "../../src/core/answers/toolText";
import type { ConversationAnswer, MarkedCitation } from "../../src/core/api";
import { markersWithoutCitation } from "../../src/shared/answerBlocks";
import { datesNotInText } from "./dates";
import type { ClaimVerdict } from "./judge";

/** What a request expects of its Answer. */
export interface Expected {
  /** It asks for text in a Mind: writing, adding, changing, making one. */
  write: boolean;
  /** Its Answer must cite the Documents in scope. */
  cite: boolean;
  /** It asks about every Document in scope ("summarise these papers"). */
  coverAll: boolean;
  /** Only this top-level Block of the Mind it is asked from (from 0) may change. */
  only?: number;
  /** It asks to change text already in the Mind: adding a new version beside it fails. */
  inPlace?: true;
  /** It is asked after the example's Answer, and its reply says nothing of an earlier Answer or the example. */
  noEarlier?: boolean;
  /** Every date its reply gives is one the Documents' text writes (`RequestOutcome.texts`), or a file's named as such. */
  textDates?: boolean;
  /** Its reply is in the Question's language, whatever language the Documents are in (#892, #1273); quoted words don't count. */
  language?: "en" | "zh-CN";
}

export interface RequestOutcome {
  expected: Expected;
  answer: Pick<ConversationAnswer, "text" | "toolCalls" | "steps"> & {
    citations: readonly Pick<MarkedCitation, "documentId" | "marker">[];
  };
  verdict: ClaimVerdict;
  /** The Documents the Citations it wrote into Minds quote, by id. */
  citedInMinds: readonly string[];
  /** The top-level Blocks of the Mind it was asked from (from 0) that read otherwise once its changes are accepted. */
  changedBlocks: readonly number[];
  /** The top-level Blocks Add to note put in before the request (from 0): what `inPlace` must change. */
  addedBlocks?: readonly number[];
  /** Sources the Minds it wrote hold as text, in brackets, as written. */
  sourcesAsText?: readonly string[];
  /** The app asked the model again inside the turn, so the User never read its first try. */
  askedAgain?: boolean;
  scoped: readonly { id: string; name: string }[];
  /** The text of the Documents in scope, for `textDates`. */
  texts?: readonly string[];
}

export type RequestFailure =
  | { kind: "no-change" }
  /** `flagged`: the app's own net said so under the reply. */
  | { kind: "unbacked"; sentences: string[]; flagged: boolean }
  | { kind: "no-citations" }
  | { kind: "dead-markers"; markers: number[] }
  | { kind: "tool-text"; blocks: string[] }
  | { kind: "uncovered"; documents: string[] }
  | { kind: "outside"; blocks: number[] }
  | { kind: "beside" }
  | { kind: "untouched" }
  | { kind: "as-text"; sources: string[] }
  | { kind: "earlier"; said: string }
  | { kind: "unseen-attempt"; said: string }
  | { kind: "date-not-in-text"; dates: string[] }
  | { kind: "wrong-language"; wanted: "en" | "zh-CN"; share: number };

const QUOTED = /“[^”]*”|"[^"]*"|「[^」]*」|『[^』]*』/g;
const CITATION_MARKER = /\[\^\d+\]/g;
const CJK = /[\u3400-\u9fff]/g;
const LATIN_LETTER = /[A-Za-z]/g;
/** An English reply may name a Chinese party; a Chinese one may name a company: past this, the other language is written. */
const ENGLISH_CJK_AT_MOST = 0.2;
const CHINESE_CJK_AT_LEAST = 0.5;

/** The share of a reply's letters, outside quoted words and Citation markers, that are Chinese characters. */
export function chineseShare(reply: string): number {
  const own = reply.replace(QUOTED, " ").replace(CITATION_MARKER, " ");
  const chinese = own.match(CJK)?.length ?? 0;
  const latin = own.match(LATIN_LETTER)?.length ?? 0;
  return chinese + latin === 0 ? 0 : chinese / (chinese + latin);
}

/** Whether the reply is in `language`, by the share of Chinese characters in its own words. */
export function inLanguage(reply: string, language: "en" | "zh-CN"): boolean {
  const share = chineseShare(reply);
  return language === "en" ? share <= ENGLISH_CJK_AT_MOST : share >= CHINESE_CJK_AT_LEAST;
}

const key = (text: string) => text.replace(/\s+/g, " ").trim().toLocaleLowerCase();

const EARLIER = new RegExp(
  [
    String.raw`\b(?:earlier|previous|first|last|prior) (?:answer|response|reply)\b`,
    String.raw`\b(?:disregard|ignore) (?:my|the|that|this) (?:earlier|previous|first|last|prior|answer|response|reply|figure|limit)`,
    String.raw`Northwind|facilities agreement|leverage (?:limit|covenant)|\b3\.[15]x\b|\b3\.[15] times`,
    "(?:之前|先前|前面|上一个|上一次|第一个)的?(?:回答|回复|答案)|(?:请|应)?(?:忽略|无视)(?:我|之前|先前|上一)",
    "贷款协议|杠杆率",
  ].join("|"),
  "i",
);

const UNSEEN_ATTEMPT = new RegExp(
  [
    String.raw`\b(?:sorry|apolog(?:y|ies|ise|ize)|my (?:mistake|bad))\b[^.!?\n]*\b(?:previous|earlier|prior|a moment ago|just now|last (?:reply|response|answer|message|time|attempt))\b`,
    String.raw`\b(?:my|the|that|your) (?:previous|earlier|prior|last) (?:reply|response|answer|message|attempt)\b`,
    String.raw`\bas I (?:said|mentioned|noted|stated) (?:before|earlier|previously|above)\b`,
    String.raw`\b(?:let me|I'll|I will) (?:try again|redo|correct (?:that|this|myself))\b`,
    "(?:抱歉|对不起|不好意思|很抱歉|道歉)[^。！？\\n]*(?:上一|前一|之前|先前|刚才|上次|前面)",
    "(?:上一|前一|之前|先前|刚才|上次|前面)(?:条|次|个)?的?(?:回复|回答|答复|答案|消息|尝试)",
    "如我(?:之前|刚才|前面|上面)所说",
  ].join("|"),
  "iu",
);

/** The first sentence that apologises for, or refers to, an earlier reply or try of the model's. */
export function aboutAnUnseenAttempt(reply: string): string | null {
  return (
    reply
      .split(/(?<=[.!?])\s+|(?<=[。！？])|\n+/)
      .map((sentence) => sentence.trim())
      .find((sentence) => UNSEEN_ATTEMPT.test(sentence)) ?? null
  );
}

const NOT_COVERED =
  /\b(?:couldn't|could not|can't|cannot|didn't|did not|wasn't|weren't|isn't|aren't|not|no|unable|left out|missing|skipped)\b|没有|未能|未|无法|不能|缺少|略去/i;

function namedAsUncovered(reply: string, name: string): boolean {
  return reply
    .split(/(?<=[.!?])\s+|(?<=[。！？])|\n+/)
    .some((sentence) => key(sentence).includes(key(name)) && NOT_COVERED.test(sentence));
}

/** The ways one request failed its guards; none when it passed. */
export function requestFailures(outcome: RequestOutcome): RequestFailure[] {
  const { expected, answer } = outcome;
  const failures: RequestFailure[] = [];
  const log = changeLogOf(answer.toolCalls);
  if (expected.write && log.changed.size === 0 && log.made.size === 0) {
    failures.push({ kind: "no-change" });
  }
  if (outcome.verdict.unbacked.length > 0) {
    failures.push({
      kind: "unbacked",
      sentences: outcome.verdict.unbacked,
      flagged: (answer.steps?.unbackedClaims ?? []).length > 0,
    });
  }
  // Asked a Question, it may still write its cited answer into a Mind, as a change to accept.
  const citedHere = expected.write
    ? outcome.citedInMinds.length
    : answer.citations.length + outcome.citedInMinds.length;
  if (expected.cite && outcome.scoped.length > 0 && citedHere === 0) {
    failures.push({ kind: "no-citations" });
  }
  const dead = markersWithoutCitation(answer);
  if (dead.length > 0) failures.push({ kind: "dead-markers", markers: dead });
  const shaped = toolTextIn(answer.text);
  if (shaped.length > 0) failures.push({ kind: "tool-text", blocks: shaped });
  if (expected.coverAll) {
    const cited = new Set([
      ...answer.citations.map((citation) => citation.documentId),
      ...outcome.citedInMinds,
    ]);
    const uncovered = outcome.scoped
      .filter(
        (document) => !cited.has(document.id) && !namedAsUncovered(answer.text, document.name),
      )
      .map((document) => document.name);
    if (uncovered.length > 0) failures.push({ kind: "uncovered", documents: uncovered });
  }
  if (expected.only !== undefined) {
    const outside = outcome.changedBlocks.filter((block) => block !== expected.only);
    if (outside.length > 0) failures.push({ kind: "outside", blocks: outside });
  }
  const added = answer.toolCalls.some(
    (call) => call.tool === "add_to_mind" && call.mindChange && call.mindChange.proposals > 0,
  );
  if (expected.inPlace && added) failures.push({ kind: "beside" });
  const { addedBlocks } = outcome;
  if (
    expected.inPlace &&
    addedBlocks &&
    !addedBlocks.some((block) => outcome.changedBlocks.includes(block))
  ) {
    failures.push({ kind: "untouched" });
  }
  if (outcome.sourcesAsText && outcome.sourcesAsText.length > 0) {
    failures.push({ kind: "as-text", sources: [...outcome.sourcesAsText] });
  }
  const said = expected.noEarlier ? answer.text.match(EARLIER)?.[0] : undefined;
  if (said) failures.push({ kind: "earlier", said });
  const unseen = outcome.askedAgain ? aboutAnUnseenAttempt(answer.text) : null;
  if (unseen) failures.push({ kind: "unseen-attempt", said: unseen });
  const dates = expected.textDates ? datesNotInText(answer.text, outcome.texts ?? []) : [];
  if (dates.length > 0) failures.push({ kind: "date-not-in-text", dates });
  if (expected.language && !inLanguage(answer.text, expected.language)) {
    failures.push({
      kind: "wrong-language",
      wanted: expected.language,
      share: chineseShare(answer.text),
    });
  }
  return failures;
}

/** A failure as the evaluation's report lists it. */
export function failureLine(failure: RequestFailure): string {
  switch (failure.kind) {
    case "no-change":
      return "asked to write, it changed no Mind";
    case "unbacked":
      return `claims a change no call made${failure.flagged ? " (flagged under the reply)" : ""}: ${failure.sentences.map((sentence) => `“${sentence}”`).join(" ")}`;
    case "no-citations":
      return "Documents in scope, and nothing it wrote cites them";
    case "dead-markers":
      return `wrote Citation markers that no Citation backs: ${failure.markers.map((marker) => `[^${marker}]`).join(", ")}`;
    case "tool-text":
      return `shows a Tool's call written as text: ${failure.blocks.map((block) => `“${block.slice(0, 60)}”`).join(", ")}`;
    case "uncovered":
      return `neither cites nor names as left out ${failure.documents.map((name) => `“${name}”`).join(", ")}`;
    case "outside":
      return `changed what it was asked to leave: Block ${failure.blocks.map((block) => block + 1).join(", ")}`;
    case "beside":
      return "added a new version beside the text it was asked to change";
    case "untouched":
      return "left the text it was asked to change as it was";
    case "as-text":
      return `wrote sources into a Mind as text: ${failure.sources.map((source) => `“${source}”`).join(", ")}`;
    case "earlier":
      return `talks about an earlier Answer or the example: “${failure.said}”`;
    case "unseen-attempt":
      return `says something of an earlier try the User never saw: “${failure.said}”`;
    case "date-not-in-text":
      return `gives a date the Documents' text doesn't: ${failure.dates.join(", ")}`;
    case "wrong-language":
      return `is not written in ${failure.wanted === "en" ? "English" : "Chinese"}, the Question's language (${Math.round(failure.share * 100)}% of its letters are Chinese characters)`;
  }
}
