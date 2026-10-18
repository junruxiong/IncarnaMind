/**
 * Text written for a page stands on its own (#1172): it may be added to the
 * User's page and read by someone who never saw the conversation. A sentence
 * that sends the reader to an earlier Answer ("Notice and holiday are covered
 * in my earlier answer.", "居家办公的规定见上一条回答。") or speaks of one
 * ("I didn't recheck the figures in my earlier answer.") goes, and a clause or
 * bracket that points back ("As I said in my earlier answer, …", "…，详见上文。",
 * "(see above)") is cut from its sentence, which keeps its words. A sentence
 * with a Citation is never taken out: its claim stays.
 */

import { rewriteSentences } from "./proseSentences";

const EARLIER = String.raw`(?:my|your|the|our|an?|that) (?:earlier|previous|prior|last|first|above|preceding|original) (?:answer|reply|response|message)s?\b`;
const IN_CONVERSATION = String.raw`(?:earlier|above) in (?:this|our|the) (?:conversation|chat)\b`;
const EARLIER_ZH =
  "(?:我的)?(?:上一条|上条|前一条|上一个|上一次|之前|先前|前面|刚才|上面)的?(?:回答|回复|答复)|上文|前文";
const SET_OUT =
  "(?:covered|set out|described|explained|discussed|listed|given|summari[sz]ed|detailed|addressed|answered|outlined|shown|noted|mentioned|included|dealt with)";
const SET_OUT_ZH =
  "(?:说明|介绍|列出|提到|讲过|说过|讲|说|解释|列明|写明|给出|总结|回答|讨论|涵盖|概括|列举)";
const SEE_ZH = "(?:见|参见|详见|请见|请参见|参阅|请参阅)";
// "Shown above.", "listed above, …", never "listed above market rates".
const ABOVE = String.raw`above(?=\s*(?:[,.;:!?)]|$))`;
const THERE = `(?:in ${EARLIER}|${ABOVE}|${IN_CONVERSATION})`;
const END = `(?=[.!?。！？]?["”’)]*$)`;

const LEADING = new RegExp(
  [
    String.raw`^(?:as|like) (?:I )?(?:said|mentioned|noted|explained|described|covered|set out|discussed|showed)(?: in ${EARLIER}| (?:earlier|above|before|previously)\b),\s*`,
    `^(?:如|正如)(?:我)?(?:在)?(?:${EARLIER_ZH})(?:中|里)?(?:所述|所说|所列|提到的|说的|讲的)[，,]\\s*`,
    "^如(?:上|前)所述[，,]\\s*",
  ].join("|"),
  "i",
);
const TRAILING = new RegExp(
  [
    String.raw`,?\s+(?:as|which (?:is|are|was|were))(?: already)? ${SET_OUT} ${THERE}${END}`,
    `[，,]\\s*(?:具体|详情|详细内容|其余|其他|另外)?(?:可)?${SEE_ZH}(?:我)?(?:在)?(?:${EARLIER_ZH})(?:中)?${END}`,
  ].join("|"),
  "i",
);
const BRACKETED = new RegExp(
  [
    String.raw`\s*\((?:see|cf\.?|as (?:in|covered in|set out in)) (?:${EARLIER}|above)\)`,
    `\\s*[（(]${SEE_ZH}(?:${EARLIER_ZH})[）)]`,
  ].join("|"),
  "gi",
);
const POINTS_BACK = new RegExp(
  [
    String.raw`\b(?:my|our) (?:earlier|previous|prior|last|first|original) (?:answer|reply|response)s?\b`,
    "(?:上一条|上条|前一条|上一个|上一次|之前|先前|前面|刚才|上面)的?(?:回答|回复|答复)",
    `^(?:please )?(?:see|refer to)(?: also)? (?:${EARLIER}|${ABOVE}|${IN_CONVERSATION})`,
    String.raw`\b(?:is|are|was|were|has been|have been)(?: already| all| both)? ${SET_OUT} ${THERE}`,
    String.raw`\bI (?:have )?(?:already )?(?:covered|set out|described|explained|discussed|listed|gave|summari[sz]ed|detailed|addressed|answered|outlined|showed|noted|mentioned)\b[^.!?]* ${THERE}`,
    `${SEE_ZH}(?:我)?(?:在)?(?:${EARLIER_ZH})(?:中)?${END}`,
    `在(?:${EARLIER_ZH})(?:中|里)?(?:已经?)?${SET_OUT_ZH}(?:了|过)?${END}`,
    `(?:${EARLIER_ZH})(?:中|里)?已经?${SET_OUT_ZH}(?:了|过)?${END}`,
  ].join("|"),
  "i",
);
const CITED = /\[[\^=]\d{1,4}\]/;

const ASKS_ABOUT_EARLIER = new RegExp(
  [
    `\\b${EARLIER}`,
    `\\b${IN_CONVERSATION}`,
    String.raw`\byou (?:said|wrote|mentioned|told|listed|gave|answered)\b|\bdid you (?:say|write|mention|tell|list|give)\b`,
    EARLIER_ZH,
    "(?:你|您)(?:刚才|之前|上面|前面|上次)(?:说|写|提到|列|讲|回答)",
  ].join("|"),
  "i",
);

/** Whether a Question asks about an earlier Answer, so its Answer may point to it. */
export const asksAboutEarlier = (question: string): boolean => ASKS_ABOUT_EARLIER.test(question);

function standingAlone(sentence: string): string {
  const cut = sentence.replace(BRACKETED, "").replace(TRAILING, "");
  const lead = LEADING.exec(cut)?.[0] ?? "";
  const rest = cut.slice(lead.length);
  const kept = lead ? rest.charAt(0).toLocaleUpperCase() + rest.slice(1) : rest;
  return POINTS_BACK.test(kept) && !CITED.test(kept) ? "" : kept;
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
