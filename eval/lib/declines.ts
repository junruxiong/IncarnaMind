/**
 * The phrase rule for an unanswerable Question's Answer: does it say the
 * Documents don't answer the Question? It starts from how the app's answer
 * prompt tells the model to decline ("If the Documents don't cover the
 * Question, say so plainly"; "isn't in what you were given") and adds how
 * models word it. It is a first pass: a reviewer marks the declines it misses
 * on the sheet. A nearby fact cited without saying so isn't a decline.
 */

const ADVERBS =
  "(?:explicitly|directly|specifically|clearly|actually|even|anywhere|seem to|appear to)";
const SAYS =
  "(?:say|state|mention|specify|cover|contain|address|answer|discuss|report|give|provide|include|have|disclose|reveal|appear|exist)";
const DOCUMENT =
  "(?:documents?|docs?|passages?|sources?|files?|materials?|texts?|reports?|papers?|articles?|excerpts?|notes?|pdfs?)";
const WHERE =
  "(?:in|found|covered|included|mentioned|stated|specified|given|provided|addressed|available|present|reported|listed|contained|discussed|answered|documented|recorded)";

const ENGLISH = [
  // "the Documents don't say", "the report does not mention", "none of the Documents cover"
  new RegExp(
    `\\b${DOCUMENT}\\b[^.!?\\n]{0,60}?\\b(?:do(?:es)?n't|do(?:es)? not|never)\\s+(?:${ADVERBS}\\s+)*${SAYS}\\b`,
  ),
  new RegExp(`\\bnone of (?:the|your|these|those) (?:\\w+ ){0,2}${DOCUMENT}\\b`),
  // "isn't in your Documents", "was not specified"
  new RegExp(`\\b(?:is|are|was|were)(?:n't| not)\\s+(?:${ADVERBS}\\s+)*${WHERE}\\b`),
  // "there is no information about…"
  /\bno (?:\w+ ){0,2}(?:information|mention|data|record|figures?|references?|details?|evidence|statement|indication|answer)\b/,
  // "I couldn't find", "can't be determined from"
  /\b(?:could(?:n't| not)|can(?:'t|not)|can not|did(?:n't| not)|was(?:n't| not) able to|unable to)\s+(?:\w+\s+){0,2}?(?:find|found|located?|determined?|answered?|confirmed?|tell|say|identified?|established?)\b/,
  /\bthe closest (?:they|it|that (?:they|it)) comes?\b/,
  /\bi (?:don't|do not) (?:know|have)\b|\bi(?:'m| am) not (?:sure|certain)\b/,
];

const NEGATION = "(?:并没有|并未|并不|没有|没能|未能|未曾|未|没)";
const SAYS_ZH =
  "(?:明确|直接|具体|专门|任何)?(?:提到|提及|说明|涉及|记载|给出|提供|包含|披露|载明|描述|介绍|找到|发现|查到|讨论|说到|写明|列出|显示|涵盖|回答|相关|答案|信息|数据)";

const CHINESE = [
  // 文档中没有提到, 文档未说明, 没有找到相关
  new RegExp(`${NEGATION}${SAYS_ZH}`),
  /无法(?:从[^，。,.]{0,15})?(?:确定|回答|得知|得出|判断|找到|提供|告诉|说明|确认)/,
  /(?:找不到|查不到|不在(?:您|你)?(?:的|已添加的)?(?:文档|资料|文件))/,
  /最接近/,
  /(?:我|本人)?(?:不知道|不确定|不清楚)/,
];

/** The text without citation markers ("[^1]"), with straight apostrophes, in lower case. */
function readable(text: string): string {
  return text
    .replace(/\[\^\d+\]/g, "")
    .replace(/[‘’]/g, "'")
    .toLowerCase();
}

/** Whether an Answer says the Documents don't answer the Question, by the phrase rule in English and Chinese. */
export function declines(text: string): boolean {
  const plain = readable(text);
  return [...ENGLISH, ...CHINESE].some((phrase) => phrase.test(plain));
}
