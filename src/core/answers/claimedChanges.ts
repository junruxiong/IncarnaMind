/**
 * What an Answer's reply says it changed in a Mind, against what its Tool
 * calls really changed (#893). The assistant must never claim a change that
 * didn't happen: a reply that does is asked for once more, with the record of
 * what the calls did (./toolRecord), and one that still does is shown with a
 * line saying so.
 *
 * A claim is read from the words a reply reports a change with, in English
 * and Chinese: "I've added a section…", "Shortened the summary in “Q3
 * review”…", "The changes are waiting for your review", "我已新建…",
 * "已在“周报”中添加了…". Words about the reply itself ("I've written a
 * summary below"), about Citations, or about someone else ("the authors
 * added…") aren't claims, and neither is a change said not to have been made.
 * A claim about an earlier Answer ("as I added before") holds when an
 * earlier Answer's calls made that kind of change.
 */
import type { AnswerToolCall, ClaimedChange } from "../api";
import { type MindNames, toolRecordLines } from "./toolRecord";

/** A change a sentence of the reply says was made, and whether it says an earlier Answer made it. */
export interface Claim {
  kind: ClaimedChange;
  sentence: string;
  earlier: boolean;
}

const MADE = "made|created|started|set up";
const ADDED = "added|wrote|written|put|inserted|drafted|appended";
const CHANGED =
  "updated|changed|edited|revised|rewrote|rewritten|expanded|lengthened|extended|shortened|trimmed|condensed|tightened|fixed|corrected|translated|reworded|restructured|reorganized|reorganised|polished|improved|proposed|merged|moved";
const REMOVED = "removed|deleted|replaced|dropped|took out|taken out";
const VERBS = `${MADE}|${ADDED}|${CHANGED}|${REMOVED}`;
/** The verbs a report starts with, as the instructions' example does; not those that start other sentences too. */
const REPORT_VERBS = VERBS.replace(/\|(?:proposed|merged|moved|put|set up|started)\b/g, "");

const KIND_OF_VERB: [RegExp, ClaimedChange][] = [
  [new RegExp(`^(?:${MADE})$`, "i"), "made"],
  [new RegExp(`^(?:${ADDED})$`, "i"), "added"],
  [new RegExp(`^(?:${CHANGED})$`, "i"), "changed"],
  [new RegExp(`^(?:${REMOVED})$`, "i"), "removed"],
];

const MIND_WORDS =
  /\b(?:minds?|notes?|pages?|notebook|handout|draft|summary|section|paragraph|heading|outline)\b/i;

/** A made thing that is a Mind's, not a mistake or a point. */
const MADE_THING =
  /^\s+(?:(?:a|an|the|your|this|that|it|one)\s+)?(?:(?:new|separate|fresh|short|shorter|longer|cited|brief|full)\s+)*(?:minds?\b|notes?\b|pages?\b|document\b|summary\b|draft\b|handout\b|outline\b|section\b|report\b|copy\b|version\b)/i;

/** "Made it longer", "made the summary clearer": a change, not something made. */
const MADE_INTO =
  /^\s+(?:it|this|that|them|the\s+\w+(?:\s+\w+)?)\s+(?:longer|shorter|clearer|simpler|tighter|more\s+\w+|less\s+\w+)/i;

/** The words after a verb that say nothing was changed, or that it is about something not in a Mind. */
const NOT_A_CHANGE =
  /^\s+(?:(?:no|nothing|none|any)\b|(?:(?:a|the|my|your|some|each|every|all)\s+)?(?:citations?|sources?|references?|markers?|quotes?|footnotes?)\b|(?:my|the|this)\s+(?:answer|reply|response)\b|(?:sure|a mistake|an error|that clear)\b)/i;

/** Words saying the text is in the reply, not in a Mind. */
const IN_THE_REPLY =
  /\b(?:below|above|here|in this (?:reply|answer|message)|in my (?:reply|answer))\b|下面|如下|以下|上面|本回答/;

const EARLIER =
  /\b(?:earlier|before|previous(?:ly)?|last (?:time|reply|answer|turn|message)|in my (?:first|last|earlier|previous) (?:reply|answer|message))\b|之前|上次|刚才|先前|前面|上一/i;

const FIRST_PERSON = new RegExp(
  `\\bI(?:'ve|’ve| have|'d|’d| had)?(?:\\s+(?:just|also|now|then|already|quickly))*\\s+(${VERBS})\\b`,
  "gi",
);

/** A sentence that starts with the verb, as a report does: "Shortened the summary in “Q3 review”." */
const REPORT = new RegExp(`^(?:[-*•]\\s+)?(${REPORT_VERBS})\\b`, "i");

/** "It has been added to your Mind", "the summary is now in “Q3 review”". */
const PASSIVE = new RegExp(`\\b(?:has|have) been (${VERBS})\\b|\\b(?:is|are) now in\\b`, "i");

const WAITING =
  /\b(?:changes?|edits?|text|it|they|these|those)\s+(?:are|is)\s+(?:now\s+)?(?:waiting|ready)\s+(?:for\s+(?:you|your)|in\b)|\bwaits?\s+for\s+your\s+(?:review|approval)\b/i;

/** A first-person or subjectless Chinese report of a change: "我已添加", "已在“周报”中新建了", "我把总结改长了". */
const CHINESE: [RegExp, ClaimedChange][] = (
  [
    ["新建|创建|建立", "made"],
    ["添加|加入|写入|插入|补充|追加|写好|写了", "added"],
    [
      "修改|改写|更新|扩写|扩展|扩充|缩短|精简|压缩|翻译|调整|重写|润色|修正|改正|改短|改长|改成",
      "changed",
    ],
    ["删除|删去|删掉|替换|去掉|移除", "removed"],
  ] as const
).map(([verbs, kind]) => [
  new RegExp(
    `(?:我(?:已经?|刚刚?|也|还|先)?|(?:^|[，,；;：:、\\s])已经?)(?:为你|帮你|替你)?(?:(?:在[“"][^”"]{1,40}[”"](?:中|里)?)|(?:把|将)[^。！？]{1,40}?)?(?:${verbs})`,
  ),
  kind,
]);

const NEGATION = /\b(?:not|nothing|never|no)\b|n['’]t\b|\bcannot\b/i;

const CHINESE_NOT = /没有|没|未能|未|无法|不能|尚未|并未/;

const CHINESE_WAITING = /等待?你(?:的)?(?:确认|审阅|接受|查看|审核)|待你确认/;

const kindOf = (verb: string): ClaimedChange =>
  KIND_OF_VERB.find(([pattern]) => pattern.test(verb.replace(/\s+/g, " ")))?.[1] ?? "changed";

const nameKey = (name: string) => name.replace(/\s+/g, " ").trim().toLocaleLowerCase();

/** The claims of one sentence of a reply; `minds`: the names of the User's Minds, as `nameKey` gives them. */
function sentenceClaims(sentence: string, minds: ReadonlySet<string>): ClaimedChange[] {
  if (IN_THE_REPLY.test(sentence)) return [];
  const namesAMind =
    MIND_WORDS.test(sentence) ||
    [...sentence.matchAll(/[“"«]([^”"»]{1,120})[”"»]/g)].some((quote) =>
      minds.has(nameKey(quote[1] ?? "")),
    );
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
    // "As I wrote before…" is about the reply's own words.
    if (/\bas\s*$/i.test(sentence.slice(0, match.index))) continue;
    consider(match[1] ?? "", sentence.slice(match.index + match[0].length), false);
  }
  const trimmed = sentence.trim();
  const report = REPORT.exec(trimmed);
  if (report?.[1]) consider(report[1], trimmed.slice(report[0].length), true);
  const passive = PASSIVE.exec(sentence);
  if (passive && namesAMind && !NEGATION.test(sentence)) {
    kinds.push(passive[1] ? kindOf(passive[1]) : "added");
  }
  const denied = NEGATION.test(sentence);
  if (!denied && (WAITING.test(sentence) || CHINESE_WAITING.test(sentence))) kinds.push("changed");
  for (const [pattern, kind] of CHINESE) {
    const match = pattern.exec(sentence);
    if (
      match &&
      !CHINESE_NOT.test(sentence.slice(Math.max(0, match.index - 4), match.index + match[0].length))
    ) {
      kinds.push(kind);
    }
  }
  return [...new Set(kinds)];
}

/** A reply's sentences, its list items and lines each one, Citations' markers left out. */
const sentencesOf = (reply: string): string[] =>
  reply
    .replace(/\[\^\d+\]|\[=\d+\]/g, "")
    .split(/(?<=[.!?])\s+|(?<=[。！？])|\n+/)
    .map((sentence) => sentence.trim())
    .filter((sentence) => sentence !== "");

/**
 * The changes a reply says were made in a Mind, each with its sentence (see
 * the module's comment). `minds`: the names of the User's Minds, so a quoted
 * name counts as a Mind's only when it is one.
 */
export function claimsIn(reply: string, minds: readonly string[] = []): Claim[] {
  const names = new Set(minds.map(nameKey));
  return sentencesOf(reply).flatMap((sentence) =>
    sentenceClaims(sentence, names).map((kind) => ({
      kind,
      sentence,
      earlier: EARLIER.test(sentence),
    })),
  );
}

export interface ChangeLog {
  made: boolean;
  /** Changed anything in a Mind: added, rewrote, revised or made one with text. */
  changed: boolean;
  /** Changed text already there, so some may be gone: `edit_mind`'s Proposals or a revision. */
  edited: boolean;
}

export function changeLogOf(calls: readonly AnswerToolCall[]): ChangeLog {
  const log: ChangeLog = { made: false, changed: false, edited: false };
  for (const call of calls) {
    const change = call.mindChange;
    if (call.source !== "minds" || !change) continue;
    if (change.created) log.made = true;
    if (change.proposals > 0 || change.revised) log.changed = true;
    if (call.tool === "edit_mind" && (change.proposals > 0 || change.revised)) log.edited = true;
  }
  return log;
}

function backs(log: ChangeLog, kind: ClaimedChange): boolean {
  switch (kind) {
    case "made":
      return log.made;
    case "added":
    case "changed":
      return log.changed || log.made;
    case "removed":
      return log.edited;
  }
}

/**
 * The kinds of change a reply claims that no call made: those of this
 * Answer, or, for a claim about an earlier one, those of the conversation's
 * earlier Answers. Each kind once. `minds`: the names of the User's Minds
 * (see `claimsIn`).
 */
export function unbackedClaims(
  reply: string,
  calls: readonly AnswerToolCall[],
  {
    earlierCalls = [],
    minds = [],
  }: { earlierCalls?: readonly AnswerToolCall[]; minds?: readonly string[] } = {},
): ClaimedChange[] {
  const now = changeLogOf(calls);
  const before = changeLogOf(earlierCalls);
  const unbacked = claimsIn(reply, minds).filter(
    (claim) => !backs(now, claim.kind) && !(claim.earlier && backs(before, claim.kind)),
  );
  return [...new Set(unbacked.map((claim) => claim.kind))];
}

const CLAIMED_WORDS: Record<ClaimedChange, string> = {
  made: "made a Mind",
  added: "added text to a Mind",
  changed: "changed a Mind",
  removed: "removed or replaced text in a Mind",
};

/**
 * What the model is told when its reply claims a change no call made: the
 * record of what its calls did, then to write the reply again from it.
 */
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
    `Your reply says you ${said}, but no Tool did that. Write your reply again from its start: say only what the Tools really did, and plainly what wasn't done and why. If the User's request still needs a change you can make now, make it with the Tools first.`,
  ].join("\n");
}
