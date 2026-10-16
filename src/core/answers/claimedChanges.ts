/**
 * A narrow safety net under the change log, which is what the Answer reports:
 * a reply that says it changed a Mind when none of its Tool calls did (or not
 * that Mind) is asked for once more, with the record of what the calls did
 * (./toolRecord), and one that still says so is flagged. It reads only the
 * plainest reports, in English and Chinese ("I've added…", "Shortened the
 * summary in “Q3 review”…", "The summary is now twice as long", "页面已创建"),
 * and misses a vague sentence rather than flag an honest one.
 */
import type { AnswerToolCall, ClaimedChange } from "../api";
import { type MindNames, toolRecordLines } from "./toolRecord";

/** A change a sentence of the reply says was made, and the Mind it names, by its name; null for none. */
export interface Claim {
  kind: ClaimedChange;
  sentence: string;
  mind: string | null;
}

const MADE = "made|created|started|set up";
const ADDED = "added|wrote|written|put|inserted|drafted|appended";
const CHANGED =
  "updated|changed|edited|revised|rewrote|rewritten|expanded|lengthened|extended|shortened|trimmed|condensed|tightened|fixed|corrected|translated|reworded|restructured|reorganized|reorganised|polished|improved|proposed|merged|moved";
const REMOVED = "removed|deleted|replaced|dropped|took out|taken out";
const VERBS = `${MADE}|${ADDED}|${CHANGED}|${REMOVED}`;
// "Proposed in 2017, the Transformer…": these start other sentences too.
const REPORT_VERBS = VERBS.replace(/\|(?:proposed|merged|moved|put|set up|started)\b/g, "");

const KIND_OF_VERB: [RegExp, ClaimedChange][] = [
  [new RegExp(`^(?:${MADE})$`, "i"), "made"],
  [new RegExp(`^(?:${ADDED})$`, "i"), "added"],
  [new RegExp(`^(?:${CHANGED})$`, "i"), "changed"],
  [new RegExp(`^(?:${REMOVED}|replaces)$`, "i"), "removed"],
];

const MIND_WORDS =
  /\b(?:minds?|notes?|pages?|notebook|handout|draft|summary|section|paragraph|heading|outline)\b/i;
const MIND_NOUN =
  "(?:minds?|notes?|pages?|notebook|handout|draft|summary|summaries|sections?|paragraphs?|headings?|outline|text|version)";

const MADE_THING =
  /^\s+(?:(?:a|an|the|your|this|that|it|one)\s+)?(?:(?:new|separate|fresh|short|shorter|longer|cited|brief|full)\s+)*(?:minds?\b|notes?\b|pages?\b|document\b|summary\b|draft\b|handout\b|outline\b|section\b|report\b|copy\b|version\b|[“"«])/i;
const MADE_INTO =
  /^\s+(?:it|this|that|them|the\s+\w+(?:\s+\w+)?)\s+(?:longer|shorter|clearer|simpler|tighter|more\s+\w+|less\s+\w+)/i;
const NOT_A_CHANGE =
  /^\s+(?:(?:no|nothing|none|any)\b|(?:(?:a|the|my|your|some|each|every|all|several|many|more|\d+)\s+)?(?:new\s+)?(?:citations?|sources?|references?|markers?|quotes?|footnotes?)\b|(?:my|the|this)\s+(?:answer|reply|response)\b|(?:sure|a mistake|an error|that clear)\b)/i;
const IN_THE_REPLY =
  /\b(?:below|above|here)[\s.:!]*$|\bin (?:this|my) (?:reply|answer|message)\b|下面|如下|以下|本回答/;
const NEGATION = /\b(?:not|nothing|never|no)\b|n['’]t\b|\bcannot\b/i;
// "As I wrote", "was wrong to say I added…": an earlier claim reported, not made.
const REPORTED =
  /\b(?:as|say|said|says|saying|claim|claimed|claims|claiming|stated|states|reported|wrote that|told you)\s+(?:that\s+)?$/i;
const CHINESE_REPORTED = /(?:说|称|声称|讲)$/;

const FIRST_PERSON = new RegExp(
  `\\bI(?:'ve|’ve| have|'d|’d| had)?(?:\\s+(?:just|also|now|then|already|quickly))*\\s+(${VERBS})\\b`,
  "gi",
);
const HEADING_LIKE = /^(?![-*•]\s)[^.!?。！？]*$/;
const REPORT = new RegExp(`^(?:[-*•]\\s+)?(${REPORT_VERBS})\\b`, "i");
const PASSIVE = new RegExp(`\\b(?:has|have) been (${VERBS})\\b|\\b(?:is|are) now in\\b`, "i");
// "Summary page created in “X”", "The new page replaces the earlier version", "is now twice as long".
const STATED = new RegExp(
  `^(?:the |a |an |your |my |this |that )?(?:[\\p{L}-]+ ){0,3}${MIND_NOUN} (?:was |were |is |are |has been |have been |now )?(${VERBS}|replaces)\\b`,
  "iu",
);
const NOW_LONGER =
  /\b(?:is|are) now (?:about |roughly |almost )?(?:twice|three times|\d+(?:\.\d+)? times|longer|shorter|much longer|much shorter|more \w+|less \w+)\b/i;

const CHINESE_KINDS = [
  ["新建|创建|建立", "made"],
  ["添加|加入|写入|插入|补充|追加|写好|写了", "added"],
  [
    "修改|改写|更新|扩写|扩展|扩充|缩短|精简|压缩|翻译|调整|重写|润色|修正|改正|改短|改长|改成",
    "changed",
  ],
  ["删除|删去|删掉|替换|去掉|移除", "removed"],
] as const;
const CHINESE: [RegExp, ClaimedChange][] = CHINESE_KINDS.map(([verbs, kind]) => [
  new RegExp(
    `(?:我(?:已经?|刚刚?|也|还|先)?|(?:^|[，,；;：:、\\s])已经?)(?:为你|帮你|替你)?(?:(?:在[“"][^”"]{1,40}[”"](?:中|里)?)|(?:把|将)[^。！？]{1,40}?)?(?:${verbs})`,
  ),
  kind,
]);
// "总结已扩写", "页面已创建": a Mind's text said to be done, not the authors'.
const CHINESE_STATED: [RegExp, ClaimedChange][] = CHINESE_KINDS.map(([verbs, kind]) => [
  new RegExp(
    `(?:总结|摘要|页面|笔记|段落|部分|章节|内容|文字|标题|列表|表格|报告|草稿|Mind|[”"])(?:已经?|都已经?|均已)(?:被)?(?:${verbs}|完成)`,
  ),
  kind,
]);
const CHINESE_NOT = /没有|没|未能|未|无法|不能|尚未|并未/;

const kindOf = (verb: string): ClaimedChange =>
  KIND_OF_VERB.find(([pattern]) => pattern.test(verb.replace(/\s+/g, " ")))?.[1] ?? "changed";

const nameKey = (name: string) => name.replace(/\s+/g, " ").trim().toLocaleLowerCase();

function sentenceClaims(sentence: string, minds: ReadonlyMap<string, string>): Claim[] {
  if (IN_THE_REPLY.test(sentence)) return [];
  const named =
    [...sentence.matchAll(/[“"«]([^”"»]{1,120})[”"»]/g)]
      .map((quote) => minds.get(nameKey(quote[1] ?? "")))
      .find((name) => name !== undefined) ?? null;
  const namesAMind = named !== null || MIND_WORDS.test(sentence);
  const denied = NEGATION.test(sentence);
  const kinds: ClaimedChange[] = [];
  const consider = (verb: string, after: string, needsMind: boolean) => {
    if (NOT_A_CHANGE.test(after)) return;
    const kind = kindOf(verb);
    if (kind === "made" && MADE_INTO.test(after)) {
      kinds.push("changed");
      return;
    }
    if (kind === "made" && !MADE_THING.test(after)) return;
    if (needsMind && !namesAMind) return;
    kinds.push(kind);
  };
  for (const match of sentence.matchAll(FIRST_PERSON)) {
    if (REPORTED.test(sentence.slice(0, match.index))) continue;
    consider(match[1] ?? "", sentence.slice(match.index + match[0].length), false);
  }
  const trimmed = sentence.trim();
  // "Expanded two-paper summary" on a line of its own is a heading, not a report.
  const report = HEADING_LIKE.test(trimmed) ? null : REPORT.exec(trimmed);
  if (report?.[1]) consider(report[1], trimmed.slice(report[0].length), true);
  if (!denied) {
    const passive = PASSIVE.exec(sentence);
    if (passive && namesAMind) kinds.push(passive[1] ? kindOf(passive[1]) : "added");
    const stated = STATED.exec(trimmed);
    if (stated?.[1]) kinds.push(kindOf(stated[1]));
    if (NOW_LONGER.test(sentence) && namesAMind) kinds.push("changed");
  }
  for (const [pattern, kind] of [...CHINESE, ...CHINESE_STATED]) {
    const match = pattern.exec(sentence);
    const before = match
      ? sentence.slice(Math.max(0, match.index - 4), match.index + match[0].length)
      : "";
    if (!match || CHINESE_NOT.test(before)) continue;
    if (!CHINESE_REPORTED.test(sentence.slice(0, match.index))) kinds.push(kind);
  }
  return [...new Set(kinds)].map((kind) => ({ kind, sentence, mind: named }));
}

const sentencesOf = (reply: string): string[] =>
  reply
    .replace(/\[\^\d+\]|\[=\d+\]/g, "")
    .split(/(?<=[.!?])\s+|(?<=[。！？])|\n+/)
    .map((sentence) => sentence.trim())
    .filter((sentence) => sentence !== "");

const MAY_CLAIM = new RegExp(`\\bI\\b|\\b(?:now|been|${VERBS}|replaces)\\b|[“"«]|我|已`, "iu");
const REPORT_VERB_LIST = REPORT_VERBS.split("|");

/**
 * How much of a reply being written may show: up to the first sentence that
 * may claim a change, which waits until the reply is checked. Of the sentence
 * still being written, all but its last word, unless it has begun the way a
 * claim does.
 */
export function shownUpTo(text: string, minds: readonly string[]): number {
  const names = new Map(minds.map((name) => [nameKey(name), name]));
  let start = 0;
  for (const match of text.matchAll(/(?<=[.!?])\s+|(?<=[。！？])|\n+/g)) {
    const sentence = text
      .slice(start, match.index)
      .replace(/\[\^\d+\]|\[=\d+\]/g, "")
      .trim();
    if (sentence && sentenceClaims(sentence, names).length > 0) return start;
    start = match.index + match[0].length;
  }
  const rest = text.slice(start);
  if (MAY_CLAIM.test(rest)) return start;
  const word = /^\s*(\p{L}+)$/u.exec(rest)?.[1]?.toLocaleLowerCase();
  if (word && REPORT_VERB_LIST.some((verb) => verb.startsWith(word))) return start;
  const lastWord = /[\p{Script=Latin}\p{N}'’]+$/u.exec(rest);
  return lastWord ? start + lastWord.index : text.length;
}

/** `minds`: the names of the User's Minds, so a quoted name counts as a Mind's only when it is one. */
export function claimsIn(reply: string, minds: readonly string[] = []): Claim[] {
  const names = new Map(minds.map((name) => [nameKey(name), name]));
  return sentencesOf(reply).flatMap((sentence) => sentenceClaims(sentence, names));
}

/** The Minds each kind of change was made in, by id. */
export interface ChangeLog {
  made: Set<string>;
  /** Anything written: added, rewritten, revised, or made with text. */
  changed: Set<string>;
  /** Text already there rewritten, so some may be gone. */
  edited: Set<string>;
}

export function changeLogOf(calls: readonly AnswerToolCall[]): ChangeLog {
  const log: ChangeLog = { made: new Set(), changed: new Set(), edited: new Set() };
  for (const call of calls) {
    const change = call.mindChange;
    if (call.source !== "minds" || !change) continue;
    if (change.created) log.made.add(change.mindId);
    if (change.proposals > 0 || change.revised) log.changed.add(change.mindId);
    if (call.tool === "edit_mind" && (change.proposals > 0 || change.revised)) {
      log.edited.add(change.mindId);
    }
  }
  return log;
}

function backs(log: ChangeLog, kind: ClaimedChange, mindIds: readonly string[] | null): boolean {
  const done =
    kind === "made"
      ? log.made
      : kind === "removed"
        ? log.edited
        : new Set([...log.changed, ...log.made]);
  return mindIds === null ? done.size > 0 : mindIds.some((id) => done.has(id));
}

/**
 * The kinds of change a reply claims that none of `calls` made, in the Mind
 * a claim names (by its name, among `minds`), else in any. Each kind once.
 */
export function unbackedClaims(
  reply: string,
  calls: readonly AnswerToolCall[],
  minds: readonly { id: string; name: string }[] = [],
): ClaimedChange[] {
  const log = changeLogOf(calls);
  const idsOf = (name: string) =>
    minds.filter((mind) => nameKey(mind.name) === nameKey(name)).map((mind) => mind.id);
  const unbacked = claimsIn(
    reply,
    minds.map((mind) => mind.name),
  ).filter((claim) => !backs(log, claim.kind, claim.mind === null ? null : idsOf(claim.mind)));
  return [...new Set(unbacked.map((claim) => claim.kind))];
}

const CLAIMED_WORDS: Record<ClaimedChange, string> = {
  made: "made a Mind",
  added: "added text to a Mind",
  changed: "changed a Mind",
  removed: "removed or replaced text in a Mind",
};

/** The record of what the calls did, then to write the reply again from it. */
export function askAgainNote(
  claims: readonly ClaimedChange[],
  calls: readonly AnswerToolCall[],
  names: MindNames,
): string {
  const lines = toolRecordLines(calls, names);
  const said = claims.map((claim) => CLAIMED_WORDS[claim]).join(", and ");
  return [
    "<tool-record>",
    "What the Tools of this Answer really did, as IncarnaMind recorded it:",
    ...(lines.length > 0 ? lines : ["- No Tool was called."]),
    "</tool-record>",
    `Your reply says you ${said}, but no Tool did that. Write your reply again from its start, without saying what you changed: IncarnaMind lists each change under your reply. Say plainly what wasn't done and why. If the User's request still needs a change you can make now, make it with the Tools first.`,
  ].join("\n");
}
