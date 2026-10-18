/**
 * Text that goes into a page stands on its own (#1172): someone who never saw
 * the conversation may read it. What sends the reader to an earlier Answer is
 * cut from it, never from the Answer in the panel. Only the pointer goes: a
 * clause ("As I said in my earlier answer, …", "I made an error in my previous
 * answer: …", "…, as covered in my earlier answer, …", "…，详见上文。", "(see
 * above)"), so its sentence keeps what it says, or a sentence that is nothing
 * but a pointer ("Notice is covered in my earlier answer.", "居家办公的规定见上一条回答。")
 * or talk of an earlier Answer ("I didn't recheck the figures in my earlier
 * answer."). A sentence with a Citation always stays, and so does what a
 * Document could say ("our first response is due", "the previous answer on the
 * form"). When in doubt, it stays.
 */

import { rewriteSentences } from "./proseSentences";

const MARKER = String.raw`\[[\^=]?\d{1,4}\]`;
const MY_EARLIER = String.raw`\bmy (?:earlier|previous|prior|last|first|original) (?:answer|reply|response)s?\b`;
const EARLIER = String.raw`(?:${MY_EARLIER}|\b(?:the|that) (?:earlier|previous|prior|last|preceding) (?:answer|reply)\b)`;
const IN_CONVERSATION = String.raw`\b(?:earlier|above) in (?:this|our|the) (?:conversation|chat)\b`;
// "Shown above.", "listed above, …", never "listed above market rates".
const ABOVE = String.raw`\babove(?=\s*(?:[,.;:!?)]|${MARKER}|$))`;
const THERE = `(?:in ${EARLIER}|${ABOVE}|${IN_CONVERSATION})`;
const SET_OUT =
  "(?:covered|set out|described|explained|discussed|listed|given|summari[sz]ed|detailed|addressed|answered|outlined|shown|noted|mentioned|included|dealt with)";
const SAID =
  "(?:said|mentioned|noted|explained|described|covered|set out|discussed|showed|shown|stated|outlined|listed)";
const I_SET_OUT =
  "I (?:have |['’]ve )?(?:already )?(?:covered|set out|described|explained|discussed|listed|gave|summari[sz]ed|detailed|addressed|answered|outlined|showed|noted|mentioned|went (?:over|through))";
const ANSWER_ZH =
  "(?:我的)?(?:(?:上一条|上条|前一条|上一个|上一次|之前|先前|前面|刚才|上面)的?回答|(?:上一条|上条|前一条|刚才)的?(?:回复|答复))";
const EARLIER_ZH = `(?:${ANSWER_ZH}|上文|前文)`;
const SET_OUT_ZH =
  "(?:说明|介绍|列出|提到|讲过|说过|讲|说|解释|列明|写明|给出|总结|回答|讨论|涵盖|概括|列举)";
const SEE_ZH = `(?:可|请)?(?:见|参见|详见|请见|参阅)(?:我)?(?:在)?${EARLIER_ZH}(?:中|里)?`;
const IN_ZH = `(?:我)?在${EARLIER_ZH}(?:中|里)?(?:已经?|都)?${SET_OUT_ZH}(?:了|过)?`;
const AS_ZH = `(?:(?:如|正如)(?:我)?(?:在)?${EARLIER_ZH}(?:中|里)?(?:所述|所说|所列|提到的|说的|讲的)|(?:正如|如)(?:上文|前文|上|前)所?述)`;
// Up to the end of the sentence: its full stop, closing quote or bracket, and markers.
const END = `(?=(?:\\s*${MARKER})*[.!?。！？]?["”’)]*(?:\\s*${MARKER})*$)`;
const PLAIN = "[^,;:\\d\\[\\]]";
const PLAIN_ZH = "[^，,；;：:。！？\\d\\[\\]]";

const REFERS = new RegExp(
  [
    EARLIER,
    IN_CONVERSATION,
    ABOVE,
    String.raw`\b${SAID} (?:earlier|before|previously)\b|\b(?:see|refer to)(?: also)? above\b`,
    EARLIER_ZH,
    "(?:上|前)(?:文)?所?述",
  ].join("|"),
  "i",
);
const BRACKETED = new RegExp(
  [
    String.raw`\s*\((?:(?:see|cf\.?)(?: also)? (?:${EARLIER}|above)|as (?:above|(?:${SET_OUT} )?in ${EARLIER}))\)`,
    `\\s*[（(](?:(?:见|参见|详见|请见|如|同)${EARLIER_ZH}(?:所述)?)[）)]`,
  ].join("|"),
  "gi",
);
const MID = new RegExp(
  [
    String.raw`,\s*(?:(?:as|which (?:is|are|was|were))(?: already)? ${SET_OUT} ${THERE}|as (?:above|in ${EARLIER})),(?=\s)`,
    `[，,](?:${AS_ZH}|${SEE_ZH})[，,]`,
  ].join("|"),
  "gi",
);
const LEADING = new RegExp(
  [
    String.raw`^(?:as|like) (?:I )?${SAID} (?:in ${EARLIER}|${IN_CONVERSATION}|(?:earlier|above|before|previously)\b),\s*`,
    String.raw`^as ${EARLIER} (?:said|says|showed|shows|noted|explained),\s*`,
    String.raw`^${EARLIER} (?:said|says|showed|noted|explained|found|stated) that\s+`,
    String.raw`^I (?:have |['’]ve )?(?:already )?${SAID} (?:in ${EARLIER}|above|earlier) that\s+`,
    String.raw`^[^:;.!?\[\]]{0,80}?${EARLIER}[^:;.!?\[\]]{0,40}:\s+`,
    String.raw`^(?:to correct|correcting|unlike|contrary to|updating|to update|revising|further to|following on from|adding to|compared with|compared to) ${EARLIER},\s*`,
    String.raw`^(?:${PLAIN}{1,80}? (?:is|are|was|were|has been|have been)(?: already| all| both)? ${SET_OUT}|${I_SET_OUT}${PLAIN}{0,80}?) ${THERE},\s*(?:and|but|while)\s+`,
    `^${AS_ZH}[，,]\\s*`,
    `^[^：:；;。！？\\[\\]]{0,30}?${ANSWER_ZH}[^：:；;。！？\\[\\]]{0,20}[：:]\\s*`,
    `^(?:(?:更正|纠正|修正)(?:一下)?(?:我)?${ANSWER_ZH}(?:中的错误|的错误)?|(?:与|和|跟)(?:我)?${ANSWER_ZH}(?:不同|相比))[，,]\\s*`,
    `^${PLAIN_ZH}{0,30}?(?:${SEE_ZH}|${IN_ZH})[，,；;]\\s*(?:但|而|不过)?`,
  ].join("|"),
  "i",
);
const TRAILING = new RegExp(
  [
    String.raw`,?\s+(?:(?:as|which (?:is|are|was|were))(?: already)? ${SET_OUT} ${THERE}|as (?:above|in ${EARLIER}))${END}`,
    String.raw`,\s+(?:(?:both|all|each|also|already) )*${SET_OUT} ${THERE}${END}`,
    String.raw`,\s+(?:and|but|while) (?:${PLAIN}{1,60}? (?:is|are|was|were|has been|have been)(?: already| all| both| also)? ${SET_OUT}|${I_SET_OUT}${PLAIN}{0,60}?) ${THERE}${END}`,
    String.raw`[,;]\s+(?:please )?see (?:${EARLIER}|above|${IN_CONVERSATION})(?: for ${PLAIN}{0,60})?${END}`,
    `[，,；;]\\s*(?:具体|详情|详细内容|其余|其他|另外)?${PLAIN_ZH}{0,30}?(?:${SEE_ZH}|${IN_ZH})${END}`,
    `[，,；;]${AS_ZH}${END}`,
  ].join("|"),
  "i",
);
const PURE = [
  new RegExp(
    String.raw`^(?:please )?(?:see|refer to)(?: also)? (?:${EARLIER}|above|${IN_CONVERSATION})(?: for (?<subject>[^.!?;:\[\]]{0,80}))?${END}`,
    "i",
  ),
  new RegExp(
    String.raw`^(?<subject>[^;:\[\]]{0,100}?)\s(?:is|are|was|were|has been|have been)(?: already| all| both| also)? (?:${SET_OUT} )?${THERE}${END}`,
    "i",
  ),
  new RegExp(String.raw`^${I_SET_OUT}\b(?<subject>[^;:\[\]]{0,80}?)\s?${THERE}${END}`, "i"),
  new RegExp(`^(?<subject>[^，,；;：:\\[\\]]{0,40}?)(?:${SEE_ZH}|${IN_ZH})${END}`),
  new RegExp(
    `^(?<subject>[^，,；;：:\\[\\]]{0,40}?)${EARLIER_ZH}(?:中|里)?(?:已经?|都)${SET_OUT_ZH}(?:了|过)?${END}`,
  ),
];
// A subject that says something of its own: "The pay is 18,000, and the rest…".
const CLAIMS =
  /\d|\b(?:is|are|was|were|has|have|had|will|would|can|could|may|might|must|shall|should|do|does|did)\b/i;
const ABOUT_EARLIER = new RegExp(
  [
    String.raw`^(?:(?:I\b|I['’](?:ve|m|d)\b|This (?:summary|answer|note|page|version|draft)\b).*(?:${MY_EARLIER}|${IN_CONVERSATION})|${MY_EARLIER})`,
    `^(?:我|本摘要|这份摘要|此摘要|这次|本次).*${ANSWER_ZH}`,
  ].join("|"),
  "i",
);
const CITED = /\[[\^=]\d{1,4}\]/;

const ASKS_ABOUT_EARLIER = new RegExp(
  [
    String.raw`\b(?:my|your|the|our|an?|that) (?:earlier|previous|prior|last|first|above|preceding|original) (?:answer|reply|response|message)s?\b`,
    IN_CONVERSATION,
    String.raw`\byou (?:said|wrote|mentioned|told|listed|gave|answered)\b|\bdid you (?:say|write|mention|tell|list|give)\b`,
    EARLIER_ZH,
    "(?:你|您)(?:刚才|之前|上面|前面|上次)(?:说|写|提到|列|讲|回答)",
  ].join("|"),
  "i",
);

/** Whether a Question asks about an earlier Answer, so what is written for it may point to one. */
export const asksAboutEarlier = (question: string): boolean => ASKS_ABOUT_EARLIER.test(question);

const pure = (sentence: string) =>
  PURE.some((shape) => {
    const match = shape.exec(sentence);
    return match !== null && !CLAIMS.test(match.groups?.subject ?? "");
  });

// One clause with no figure outside brackets: a sentence with more says something of its own.
function aboutEarlier(sentence: string): boolean {
  const bare = sentence.replace(/\([^)]*\)|（[^）]*）/g, "");
  return !/[\d,;:，；：]/.test(bare) && ABOUT_EARLIER.test(bare);
}

// "the total is 45" becomes "The total is 45", but "eBay" and "x86" stay as written.
const capitalised = (text: string) =>
  text.replace(
    /^\p{Ll}+(?=[\s,;:.!?'’]|$)/u,
    (word) => word.charAt(0).toLocaleUpperCase() + word.slice(1),
  );

function standingAlone(sentence: string): string {
  if (!REFERS.test(sentence)) return sentence;
  let cut = sentence.replace(BRACKETED, "").replace(MID, (clause) => clause.charAt(0));
  const lead = LEADING.exec(cut)?.[0] ?? "";
  const rest = cut.slice(lead.length);
  if (lead && /[\p{L}\p{N}]/u.test(rest)) cut = capitalised(rest);
  if (CITED.test(cut)) return cut.replace(TRAILING, "");
  if (pure(cut)) return "";
  cut = cut.replace(TRAILING, "");
  return aboutEarlier(cut) ? "" : cut;
}

/**
 * Markdown without what points to an earlier Answer, and how many sentences
 * changed or went. With nothing left, it is given back as it was.
 */
export function withoutPointersBack(markdown: string): { markdown: string; left: number } {
  const kept = rewriteSentences(markdown, standingAlone);
  if (!/[\p{L}\p{N}]/u.test(kept.markdown)) return { markdown, left: 0 };
  return { markdown: kept.markdown, left: kept.changed };
}

/** Markdown as it goes into a page: without pointers to an earlier Answer, unless its Question asked about one. */
export const forPage = (markdown: string, question: string): string =>
  asksAboutEarlier(question) ? markdown : withoutPointersBack(markdown).markdown;
