/**
 * Whether a request in the assistant plainly asks for a change in a Mind,
 * from its words, in English and Chinese: "edit" text there, "write" new
 * text, or "unclear", which is every other request, questions among them.
 * `newMind`: a new page or note is asked for.
 *
 * Words can't tell every request apart, so a request routes only when the
 * whole of it is one of a few plain instructions: "make my summary shorter",
 * "write a summary page", "add this to my notes", "把第二段改得更正式",
 * "写一个总结页面". Anything before or after it ("…, does that work",
 * "这篇论文讲了什么？…", "…，怎么样") leaves it unclear. A follow-up with no
 * object ("longer", "make it shorter", "简单一点") routes only when the
 * conversation's last Answer changed a Mind (`followsAChange`). An unclear
 * request is answered as before: the model chooses, and its changes wait for
 * the User.
 */
export type RequestKind = "edit" | "write" | "unclear";

export interface RequestRoute {
  kind: RequestKind;
  newMind: boolean;
}

// Never a doubt: "I mean write a summary…", "no, keep it shorter", "帮我润色一下这段".
const FILLER =
  /^(?:(?:please|pls|i mean|ok|okay|now|then|no|yes|so|just|actually|great|thanks|let's|help me|go ahead and|i'd like you to|i would like you to)[,!]?\s+)*/;
const ZH_FILLER = /^(?:请|麻烦你?|帮我|好的，?|那就|就)*/;
// Question words and hedges are a closed class: anywhere, even in free text ("add which part to
// my notes", "把哪些内容加到我的笔记里"), they leave the request unclear.
const ASKS =
  /\b(?:who|whom|whose|what|which|where|when|why|how|whether|if|maybe|perhaps|possibly|probably|unsure|idk|wdyt|lmk|eh|huh|can|could|would|should|do|does|did|you|think|reckon|guess|suppose)\b/;
const ZH_ASKS =
  /谁|什么|啥|哪|多少|几|何|如何|怎|是否|可否|能否|如果|要是|或许|也许|大概|可能|吗|呢|吧|么|嘛|(\p{Script=Han})[不没]\1/u;

const WORD =
  "(?!(?:or|and|but|unless|then|is|will|i|ok|okay|right|yes|yeah|no|not|thoughts|idea|worth|fine)\\b)[\\p{L}\\p{N}'’-]+";
const WORDS = `(?:${WORD} ?){1,8}`;
const LANGUAGE = "(?:chinese|english|french|german|spanish|japanese|korean|plain english)";
const SOME = "(?:these|the|both|this|all|my)(?: \\d+| two)?";
const DOCUMENTS = `${SOME} (?:papers?|documents?|reports?|notes)`;

const EDIT_VERB =
  "(?:shorten|lengthen|rewrite|tighten|simplify|translate|delete|remove|cut|fix|proofread|polish|rephrase|reword|condense|expand|edit|revise|correct|tidy up|clean up)";
const COMPARED =
  "(?:longer|shorter|simpler|clearer|tighter|briefer|better|punchier|smoother|crisper|plainer|friendlier|stronger)";
const MORE =
  "(?:more|less) (?:detail|details|examples|context|formal|casual|concise|precise|specific|technical|accessible|readable|natural|professional|persuasive|neutral|polished|wordy|repetitive|jargon)";
const OWN_TEXT =
  "(?:this|the|my) (?:(?:first|second|third|fourth|fifth|last|opening|closing|whole) )?(?:paragraph|section|sentence|summary(?: section)?|intro|introduction|conclusion|page|draft|note|notes|list|table|text|heading|bullet points|typos|wording|part about (?:" +
  WORD +
  " ?){1,5})";
const PLACE_OF_IT = "(?: (?:in|on) (?:my|this|the) (?:notes?|page|summary|draft))?";
const CHANGE =
  "(?:(?:a bit|a little|much|even|slightly) )?(?:" +
  COMPARED +
  "|" +
  MORE +
  "|formal|informal|concise|simple|short|clear|readable|(?:flow|read|sound) better|(?:about )?\\d+ words|(?:in|into) " +
  LANGUAGE +
  "|into (?:a|one) (?:table|list|paragraph)|as a (?:table|list)|to be (?:" +
  COMPARED +
  "|" +
  MORE +
  "))";
const LEAVE_THE_REST =
  "(?:,? and (?:leave|keep) the (?:other|others|rest)(?: paragraphs?| sections?)?(?: as (?:they are|it is)| unchanged| alone)?)?";

const EDITS = [
  new RegExp(`^(?:make|turn) ${OWN_TEXT}${PLACE_OF_IT} ${CHANGE}${LEAVE_THE_REST}$`, "u"),
  new RegExp(`^${EDIT_VERB} ${OWN_TEXT}${PLACE_OF_IT}(?: ${CHANGE})?${LEAVE_THE_REST}$`, "u"),
  new RegExp(`^write ${OWN_TEXT} (?:in|into) ${LANGUAGE}$`, "u"),
];
// "Delete my notes" may mean the Mind itself, which no edit removes.
const WHOLE_MIND = /^(?:delete|remove|cut) (?:this|the|my) (?:whole )?(?:notes?|page|draft|text)$/;

const WRITES = [
  new RegExp(
    `^(?:write|make|create|draft|start|prepare)(?: me)? (?:a|an)(?: new)?(?: (?:short|brief|one-page|summary|separate))? (?:page|note|handout|summary page)(?: (?:of|on|about|for) (?:${DOCUMENTS}|${WORDS}))?(?: with ${WORDS})?(?: called ${WORDS})?(?: in ${LANGUAGE})?$`,
    "u",
  ),
  new RegExp(`^(?:make|take) notes (?:on|about|from) ${DOCUMENTS}$`, "u"),
  new RegExp(
    `^(?:write|draft|prepare)(?: me)? (?:a|an)(?: short| brief| one-page)? (?:summary|overview|outline|brief|memo)(?: (?:of|about|on) ${DOCUMENTS})?(?: in ${LANGUAGE})?(?: (?:in|on|into) (?:a )?new (?:note|page))?$`,
    "u",
  ),
  /^(?:write|draft) (?:the|my) (?:conclusion|introduction|intro|summary|abstract|next section|next paragraph)$/,
  new RegExp(
    `^(?:add|put|save|copy|insert|append) ${WORDS} (?:to|into|in) (?:my|the) notes?$`,
    "u",
  ),
  new RegExp(
    `^(?:add|insert|append) (?:a|an|another) (?:[\\p{L}'’-]+ ){0,3}(?:section|paragraph|table|list)(?: (?:on|about) ${WORDS})?$`,
    "u",
  ),
  new RegExp(`^add (?:a|an)(?: short)? note(?: (?:to|about|on) ${WORDS})?$`, "u"),
  new RegExp(
    `^summari[sz]e ${DOCUMENTS}(?: and (?:add|put|save) (?:it|this|them))? (?:to|into|in|on) (?:my notes|(?:a )?new (?:note|page)|a (?:note|page))$`,
    "u",
  ),
];

const FOLLOW_UPS = [
  new RegExp(
    `^(?:(?:make|keep) it )?(?:(?:a bit|a little|much|even) )?(?:${COMPARED}|${MORE}|short|simple|formal|concise|(?:about )?\\d+ words)$`,
    "u",
  ),
  new RegExp(
    `^(?:expand|shorten|simplify|translate|rewrite|tighten|condense|fix|polish|lengthen)(?: (?:it|this|that))?(?: up)?(?: a bit)?(?: (?:in|into) ${LANGUAGE})?$`,
    "u",
  ),
  new RegExp(`^(?:write|put) it (?:in|into) (?:${LANGUAGE}|a table|a list|bullet points)$`, "u"),
  /^(?:remove|delete) it$/,
  /^turn (?:it|this) into a (?:table|list)$/,
];
const CONTINUE = /^(?:continue|go on|keep going)$/;

const ZH_OWN_TEXT =
  "(?:这段|这一段|这一节|这节|这页|这部分|这一部分|这句话?|这一句|第[一二三四五六七八九十\\d]+段|最后一段|我的(?:总结|笔记|段落|草稿)|关于[^，。？！?吗呢吧么]{1,10}的部分)";
const ZH_EDIT_VERB =
  "(?:改得|写得|改成|改为|改短|改长|写长|写短|缩短|扩写|翻译成|翻译为|翻译|润色|精简|删掉|删除|删去|去掉|简化|改)";
const ZH_CHANGE =
  "(?:更?(?:正式|简洁|简单|清楚|详细|通顺|口语化)|[长短]|中文|英文)?(?:一点|一些|些)?";
const ZH_LEAVE_THE_REST = "(?:，?其他(?:段落|部分|的)?(?:保持不变|不变|不要动|别动))?";
const ZH_FREE = "[^，。？！?吗呢吧么]{1,20}";

const ZH_EDITS = [
  new RegExp(
    `^(?:在(?:我的)?笔记(?:里|中))?把${ZH_OWN_TEXT}${ZH_EDIT_VERB}${ZH_CHANGE}${ZH_LEAVE_THE_REST}$`,
  ),
  new RegExp(
    `^(?:缩短|删掉|删除|润色(?:一下)?|精简(?:一下)?|翻译(?:一下)?|改一下|扩写|简化)${ZH_OWN_TEXT}$`,
  ),
];
const ZH_WRITES = [
  /^写一[个份篇](?:总结|概述)?(?:页面|页)$/,
  /^写一[份篇](?:关于)?(?:这两篇论文|这些文档|这篇论文|这些论文)的(?:总结|概述)$/,
  /^(?:新建|创建)一[篇个份](?:叫[“"][^”"]{1,20}[”"]的)?(?:新的?)?(?:笔记|页面)$/,
  new RegExp(
    `^把${ZH_FREE}(?:加到|添加到|放到|记到|写到|写进|存到)(?:我的)?(?:笔记|页面)(?:里|中)?$`,
  ),
  new RegExp(
    `^(?:在(?:我的)?笔记(?:里|中))?(?:添加|加上|加|插入|补充)一[个段节]${ZH_FREE}?(?:部分|章节|段落|小节)$`,
  ),
  /^总结一下(?:这篇论文|这两篇论文|这些文档|这些论文)，?(?:并|然后)?(?:放到|加到|记到)(?:我的)?笔记(?:里|中)$/,
  /^写一[份篇个](?:简短的)?(?:总结|概述|摘要)，?(?:并|然后)?(?:放到|加到|记到|写到)(?:我的)?笔记(?:里|中)$/,
  /^(?:新建|创建)一[篇个份](?:新的?)?笔记，?(?:总结|概括)(?:一下)?(?:这两篇论文|这篇论文|这些文档|这些论文)$/,
];
// What the last Answer said, put into a Mind: "add it to my notes", "加到我的笔记里".
const ADD_IT =
  /^(?:add|put|save|copy) (?:this|that|it|these|them|your answer|the answer) (?:to|into|in) (?:my|the) notes?$/;
const ZH_ADD_IT =
  /^(?:把(?:这个|这些|它|它们|刚才的回答|这个回答))?(?:加到|添加到|放到|记到|存到)(?:我的)?(?:笔记|页面)(?:里|中)?$/;
const ZH_FOLLOW_UPS = [
  /^(?:再|更)?(?:长|短|详细|简单|正式|简洁|具体)一?(?:点|些)?$/,
  /^更(?:正式|简洁|详细|简单)(?:一点|一些)?$/,
  /^把它(?:写|改)得?更?(?:长|短|简单|正式|简洁|详细)一?[点些]?$/,
  /^(?:写得|改得|总结得|说得)(?:再|更)?(?:长|短|详细|简单|正式|简洁)一?[点些]?$/,
  /^改一下$/,
  /^(?:翻译成|改成)(?:中文|英文)$/,
];
const ZH_CONTINUE = /^(?:继续|接着写|继续写)$/;

const NEW_MIND =
  /\b(?:new|separate|another|fresh|an?)\s+(?:[\p{L}-]+\s+){0,2}(?:page|note|handout)\b|\b(?:make|take) notes\b|新建|创建|一[个份篇](?:总结|概述)?(?:页面|页)$/u;
// "…to Project plan": a Mind the User has, by its name.
const NAMES_A_MIND =
  /\b(?:to|into|in)\s+[“"]?(?!(?:English|Chinese|French|German|Spanish|Japanese|Korean)\b)\p{Lu}/u;

const matches = (patterns: readonly RegExp[], text: string) =>
  patterns.some((pattern) => pattern.test(text));

export interface RequestContext {
  /** The conversation's last Answer changed a Mind: "longer" means what it wrote. */
  followsAChange: boolean;
  /** The conversation has an Answer before: "add it to my notes" means it. */
  followsAnAnswer: boolean;
}

function plainKind(question: string, { followsAChange, followsAnAnswer }: RequestContext) {
  const trimmed = question.trim();
  // "Make my summary shorter..." trails off.
  if (/(?:\.{2,}|…|。{2,})$/.test(trimmed)) return null;
  if (/\p{Script=Han}/u.test(trimmed)) {
    const chinese = trimmed
      .replace(/[。！]+$/, "")
      .replace(/，?谢谢$/, "")
      .replace(ZH_FILLER, "");
    if (ZH_ASKS.test(chinese)) return null;
    if (matches(ZH_EDITS, chinese)) return "edit";
    if (ZH_ADD_IT.test(chinese)) return followsAnAnswer ? "write" : null;
    if (matches(ZH_WRITES, chinese)) return "write";
    if (!followsAChange) return null;
    if (matches(ZH_FOLLOW_UPS, chinese)) return "edit";
    return ZH_CONTINUE.test(chinese) ? "write" : null;
  }
  const english = trimmed
    .replace(/\s+/g, " ")
    .toLocaleLowerCase()
    .replace(/[.!]$/, "")
    .replace(/,? (?:thanks|thank you|then)$/, "")
    .replace(/,? (?:please|pls)$/, "")
    .replace(FILLER, "");
  if (ASKS.test(english)) return null;
  if (matches(EDITS, english)) return WHOLE_MIND.test(english) ? null : "edit";
  if (ADD_IT.test(english)) return followsAnAnswer ? "write" : null;
  if (matches(WRITES, english)) return "write";
  if (!followsAChange) return null;
  if (matches(FOLLOW_UPS, english)) return "edit";
  return CONTINUE.test(english) ? "write" : null;
}

export function routeRequest(question: string, context: RequestContext): RequestRoute {
  const kind = plainKind(question, context) ?? "unclear";
  const newMind =
    kind === "write" && !NAMES_A_MIND.test(question) && NEW_MIND.test(question.toLocaleLowerCase());
  return { kind, newMind };
}
