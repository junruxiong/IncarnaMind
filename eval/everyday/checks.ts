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
 * - **About the Question asked**: after the example's Answer, the reply
 *   says nothing of an earlier Answer.
 * - **Nothing lost on the way into a page**: a sentence left out of the reply
 *   as Add to note puts it, or of what a call gave to write into a Mind, is
 *   only a pointer to an earlier Answer ("covered in my earlier answer",
 *   "见上一条回答"); the pointers left there are counted.
 * - **No word of the example unasked**: asked about the User's own Documents
 *   after the example's Answer, nothing it wrote mentions the example's.
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
  /** Asked after the example's Answer: its reply, in the panel, says nothing of an earlier Answer. */
  noEarlier?: true;
  /** The report counts the pointers to an earlier Answer left in its reply as Add to note puts it, and in what it writes into a Mind. */
  standsAlone?: true;
  /** Asked about the User's own Documents after the example's Answer: nothing it writes mentions the example's. */
  noExample?: true;
  /** Every date its reply gives is one the Documents' text writes (`RequestOutcome.texts`), or a file's named as such. */
  textDates?: boolean;
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
  /** The text of the Blocks it wrote or changed in Minds, once its changes are accepted. */
  writtenInMinds?: readonly string[];
  /** Its reply as Add to note puts it in a page; the reply as written when not given. */
  addedToPage?: string;
  /** The Markdown its calls gave to write into Minds, without what search-talk filtering takes out. */
  givenToMinds?: readonly string[];
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
  | { kind: "example"; said: string }
  | { kind: "content-lost"; sentences: string[] }
  | { kind: "unseen-attempt"; said: string }
  | { kind: "date-not-in-text"; dates: string[] };

const key = (text: string) => text.replace(/\s+/g, " ").trim().toLocaleLowerCase();

const POINTS_BACK = new RegExp(
  [
    String.raw`\b(?:earlier|previous|first|last|prior|above|preceding) (?:answer|response|reply|message)s?\b`,
    String.raw`\b(?:disregard|ignore) (?:my|the|that|this) (?:earlier|previous|first|last|prior|answer|response|reply|figure|limit)`,
    String.raw`\bas (?:I )?(?:said|mentioned|noted|explained|described|covered|discussed) (?:earlier|before|above|previously)\b`,
    String.raw`\bearlier in (?:this|our|the) (?:conversation|chat)\b`,
    String.raw`\b(?:as|see|covered|described|explained|discussed|listed|noted|mentioned|set out|shown) above\b`,
    "(?:上一条|上条|前一条|上一个|上一次|之前|先前|前面|刚才|上面|第一个)的?(?:回答|回复|答复|答案)",
    "(?:见|参见|详见|请见|如|参阅)上文|上文(?:已|所述|提到)|如上所述|如前所述|前文(?:已|所述|提到)",
    "(?:请|应)?(?:忽略|无视)(?:我|之前|先前|上一)",
  ].join("|"),
  "i",
);

/** Whether text points to an earlier Answer, as `noEarlier` and `standsAlone` read it. */
export const pointsBack = (text: string): boolean => POINTS_BACK.test(text);

function proseSentences(text: string): string[] {
  let fenced = false;
  return text.split("\n").flatMap((line) => {
    if (/^\s*(?:```|~~~)/.test(line)) {
      fenced = !fenced;
      return [];
    }
    // Headings, tables, quotes and lead-ins are the page's shape, not what it says.
    if (fenced || /^\s*(?:\||#{1,6}\s|>)|[:：](?:\*\*)?\s*$|^\s*\*\*[^*]+\*\*\s*$/.test(line))
      return [];
    return line
      .replace(/^\s*(?:[-*+]|\d{1,3}[.)])\s+/, "")
      .split(/(?<=[.!?。！？]["”’)」』]*(?:\[\^\d{1,4}\])*)\s*/u)
      .map((sentence) => sentence.trim())
      .filter((sentence) => /[\p{L}\p{N}]/u.test(sentence));
  });
}

const lettersOf = (text: string) =>
  text
    .replace(/\[[\^=]?\d{1,4}\]/g, "")
    .replace(/\[[^\]\n]{1,200}\]/g, "")
    .replace(/[^\p{L}\p{N}]+/gu, "")
    .toLocaleLowerCase();

/** The sentences of `given` that `written` doesn't hold, read as letters and digits only. */
export function leftOut(given: string, written: string): string[] {
  const there = lettersOf(written);
  return proseSentences(given).filter((sentence) => !there.includes(lettersOf(sentence)));
}

// The words a pointer is made of, so what a sentence says besides is what's left.
const POINTER_WORDS = new Set(
  (
    "see refer to check for the a an my in is are was were be been has have i i've already as like " +
    "covered set out described explained discussed listed given summarised summarized detailed " +
    "addressed answered outlined shown showed noted mentioned included dealt with stated said " +
    "above earlier previous prior last first original answer answers reply response conversation " +
    "chat this that these those it they all both and or also please more further full rest other " +
    "details detail information list breakdown same which of on"
  ).split(" "),
);
const POINTER_ZH =
  /上一条|上条|前一条|上一个|上一次|之前|先前|前面|刚才|上面|回答|回复|答复|上文|前文|参见|详见|请见|参阅|参考|查看|具体|更多|信息|详情|内容|其余|其他|正如|所述|所说|已经|说明|介绍|列出|提到|一致|[的请可见如已在中里我同与，。]/gu;

function onlyAPointer(sentence: string): boolean {
  if (!POINTS_BACK.test(sentence) || /[\d"“”「」『』]/.test(sentence)) return false;
  const words = sentence.toLocaleLowerCase().match(/[a-z]+(?:['’][a-z]+)?/g) ?? [];
  const han = sentence.replace(POINTER_ZH, "").match(/\p{Script=Han}/gu) ?? [];
  return words.filter((word) => !POINTER_WORDS.has(word)).length <= 4 && han.length <= 10;
}

/**
 * Of sentences left out of a page, those that say more than a pointer to an
 * earlier Answer, by this evaluation's own reading rather than the app's: a
 * pointer names an earlier Answer, holds no figure, marker or quote, and
 * says at most four words, or ten Chinese characters, besides.
 */
export const lostContent = (sentences: readonly string[]): string[] =>
  sentences.filter((sentence) => !onlyAPointer(sentence));

/** The sentences of what goes into a page that point to an earlier Answer. */
export const pointersIn = (text: string): string[] =>
  proseSentences(text).filter((sentence) => POINTS_BACK.test(sentence));

/** The default example's Documents, figures and words (src/core/exampleSets.ts), in English and Chinese. */
const EXAMPLE =
  /Northwind|facilities agreement|credit committee|leverage (?:limit|covenant)|\b3\.[15] ?(?:x|times)\b|\b3\.5:1|贷款协议|杠杆率|授信委员会/i;

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

/** What was taken out on the way into a page: of the reply, by Add to note, and of what its calls gave to write into Minds. */
export const takenOut = (outcome: RequestOutcome): string[] => [
  ...(outcome.addedToPage === undefined ? [] : leftOut(outcome.answer.text, outcome.addedToPage)),
  ...leftOut(
    (outcome.givenToMinds ?? []).join("\n\n"),
    (outcome.writtenInMinds ?? []).join("\n\n"),
  ),
];

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
  const inMinds = outcome.writtenInMinds ?? [];
  const earlier = expected.noEarlier ? answer.text.match(POINTS_BACK)?.[0] : undefined;
  if (earlier) failures.push({ kind: "earlier", said: earlier });
  const written = [answer.text, ...inMinds].join("\n\n");
  const example = expected.noExample ? written.match(EXAMPLE)?.[0] : undefined;
  if (example) failures.push({ kind: "example", said: example });
  const lost = lostContent(takenOut(outcome));
  if (lost.length > 0) failures.push({ kind: "content-lost", sentences: lost });
  const unseen = outcome.askedAgain ? aboutAnUnseenAttempt(answer.text) : null;
  if (unseen) failures.push({ kind: "unseen-attempt", said: unseen });
  const dates = expected.textDates ? datesNotInText(answer.text, outcome.texts ?? []) : [];
  if (dates.length > 0) failures.push({ kind: "date-not-in-text", dates });
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
      return `talks about an earlier Answer: “${failure.said}”`;
    case "example":
      return `mentions the example unasked: “${failure.said}”`;
    case "content-lost":
      return `left out of a page what says more than a pointer: ${failure.sentences.map((sentence) => `“${sentence}”`).join(" ")}`;
    case "unseen-attempt":
      return `says something of an earlier try the User never saw: “${failure.said}”`;
    case "date-not-in-text":
      return `gives a date the Documents' text doesn't: ${failure.dates.join(", ")}`;
  }
}
