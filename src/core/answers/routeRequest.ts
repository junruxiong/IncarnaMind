/**
 * Whether a Question in the assistant asks for an answer or for text in a
 * Mind, from its words alone, in English and Chinese: "write" new text,
 * "edit" text already there, "answer" in the reply and change no Mind, or
 * "unclear", whose changes wait for the User instead of applying. `newMind`:
 * it asks for a new page or note.
 *
 * What the words are about decides, more than the verb: "translate the
 * abstract of the paper" or "clarify what the authors mean" is about a
 * Document, so it is answered; "make this paragraph more formal" or
 * "把它写长一点" is about the User's own text, so it is an edit, never a
 * second copy; "add it to my notes" names where it goes, so it is written. An
 * edit applies at once (#880), so whatever can't be told is "unclear", never
 * an edit.
 */
export type RequestKind = "answer" | "write" | "edit" | "unclear";

export interface RequestRoute {
  kind: RequestKind;
  newMind: boolean;
}

const LEAD =
  "(?:(?:ok(?:ay)?|so|now|then|and|also|just|please|pls|i mean|actually|instead|go ahead and|let's|help me|i want you to|i'd like you to|i would like you to|can you|could you|would you|will you)[,]?\\s+)*";

const WRITE =
  "write|draft|compose|create|make|add|insert|put|append|jot down|note down|generate|produce|prepare|save|turn (?:it|this|that|these|them) into";
const EDIT =
  "shorten|lengthen|condense|trim|tighten|simplify|rewrite|rephrase|reword|paraphrase|translate|fix|correct|proofread|polish|improve|edit|revise|restructure|reorgani[sz]e|reformat|remove|delete|cut|drop|replace|change|update|rename|merge|split|move|tidy up|clean up|expand|extend";
// Asked of a Document's text, these explain it; only the User's own text is rewritten by them.
const EXPLAIN = "clarify|elaborate on|elaborate|expand on|explain|put simply|in plain words|unpack";
const VERB = new RegExp(`^${LEAD}(${EXPLAIN}|${EDIT}|${WRITE}|summari[sz]e)\\b`, "u");
const IS_EDIT = new RegExp(`^(?:${EDIT})$`, "u");
const IS_EXPLAIN = new RegExp(`^(?:${EXPLAIN})$`, "u");

/** "make it longer", "make my summary 200 words": a change, not something made. */
const MAKE_INTO = new RegExp(
  `^${LEAD}make (?:it|this|that|them|these|those|(?:the|my|your|this|that) [\\p{L}\\p{N}'’ -]{1,40}?) (?:longer|shorter|clearer|simpler|tighter|briefer|more \\p{L}+|less \\p{L}+|formal|informal|casual|concise|readable|about \\d+ words|\\d+ words)\\b`,
  "u",
);
/** "longer please", "more formal": what was just written, said again. */
const MORE_OF =
  /^(?:(?:a bit|a little|much|even|slightly)\s+)?(?:longer|shorter|more \p{L}+|less \p{L}+|simpler|clearer|tighter)(?:\s+(?:please|pls|again|still))?$/u;

const ASKS =
  /^(?:what|why|how|when|where|who|whom|which|whose|is|are|was|were|does|do|did|should|shall|may|might|must|has|have|had)\b/;
const MAYBE = /^(?:should|shall|may|might|could|would|can) (?:i|we)\b/;

const DOCUMENT =
  /\b(?:the|this|that|these|those|its)\s+(?:paper|papers|article|articles|study|studies|abstract|authors?|tutorial|filing|filings|10-k|book|chapter|pdf|pdfs|document|documents|report|reports)\b|\bthe authors?\b|\bwhat (?:the authors?|they|it) (?:means?|say|says|claims?)\b|\b(?:paper|article|study|document|report)'s\b/;
// Named outright: "this paragraph", "my summary", "the typos".
const OWN_NAMED =
  /\b(?:this|that|the|my|your|our)\s+(?:(?:first|second|third|fourth|last|next|opening|closing|whole)\s+)?(?:paragraphs?|sentences?|sections?|summary|intro|introduction|conclusion|page|notes?|draft|text|heading|list|bullets?|bullet points?|table|selection|wording|writing|parts?|typos?|spelling|grammar|formatting|tone)\b/;
const OWN = new RegExp(`${OWN_NAMED.source}|\\b(?:it|them)\\b|^(?:this|that)\\b|\\b(?:this|that)$`);
const DESTINATION =
  /\b(?:to|into|in|on)\s+(?:my|the|this|a|a new|your|our)\s+(?:new\s+)?(?:notes?|page|mind|notebook|doc)\b|\b(?:new|separate|another|fresh)\s+(?:[\p{L}-]+\s+){0,2}(?:page|note|notes|mind|doc)\b|\bsummary page\b|\b(?:an?)\s+(?:[\p{L}-]+\s+){0,2}(?:page|note|handout)\b|\bmake notes\b|\btake notes\b/u;

const NEW_MIND =
  /\b(?:new|separate|another|fresh|an?)\s+(?:[\p{L}-]+\s+){0,2}(?:page|note|notes|mind|document|doc|handout|report|file)\b|\b(?:make|take|write) notes\b/u;

const ZH_WRITE =
  "写|撰写|起草|生成|整理成|总结成|汇总成|做成|新建|创建|添加|加上|加入|加到|补充|插入|记下|记到|放到|存到";
const ZH_EDIT =
  "缩短|改短|改长|加长|写长|写短|扩写|扩展|扩充|精简|压缩|简化|改写|重写|润色|翻译|修改|改成|改为|改得|写得|改一下|调整|删除|删掉|删去|去掉|移除|替换|更新|校对|修正|纠正|重新组织";
const ZH_EXPLAIN = "解释|说明|阐述|讲讲|讲解|展开讲|展开说";
const ZH_VERB = new RegExp(
  `^(?:请|麻烦你?|帮我|给我|你?(?:可以|能不能|能否)(?:帮我)?|再|接着|然后|我想让你|那就|就)*(?:在[^，。！？]{1,20}?(?:里|中|内|上|后面|前面|末尾|最后))?(?:把[^，。！？]{1,40}?)?(${ZH_EXPLAIN}|${ZH_EDIT}|${ZH_WRITE})`,
);
const ZH_IS_EDIT = new RegExp(`^(?:${ZH_EDIT})$`);
const ZH_IS_EXPLAIN = new RegExp(`^(?:${ZH_EXPLAIN})$`);
const ZH_ASKS = /^(?:什么|为什么|怎么|怎样|如何|哪|谁|多少|是否|是不是|有没有)/;
const ZH_DOCUMENT = /论文|文章|作者|原文|文中|这篇|这份文档|这份报告|文档的|报告的/;
const ZH_OWN =
  /它|这段|这一段|第[一二三四五六七八九十\d]+段|这部分|这一部分|这句|我的(?:总结|笔记|段落|摘要|草稿)|总结|笔记|页面|段落|(?:关于|有关)[^，。！？]{1,20}的部分/;
const ZH_OWN_NAMED =
  /这段|这一段|第[一二三四五六七八九十\d]+段|这部分|这一部分|这句|我的(?:总结|笔记|段落|摘要|草稿)/;
const ZH_DESTINATION =
  /(?:加到|添加到|放到|写到|写进|记到|存到|加入)(?:我的|这个|这份)?(?:笔记|页面|Mind)|(?:到|在|进)(?:我的)?笔记(?:里|中)|新建|一[篇份个](?:新的?)?(?:总结)?(?:笔记|页面|文档)/;
const ZH_MORE_OF =
  /^(?:再)?(?:长|短|正式|简洁|简单|清楚|详细)(?:一点|一些|些)(?:吧|啊)?$|^(?:更|再)(?:长|短|正式|简洁|简单|清楚|详细)(?:一点|一些|些)?(?:吧)?$/;
const ZH_NEW_MIND =
  /新建|新的?(?:笔记|页面|文档|Mind)|一[篇份个](?:新的?)?(?:总结)?(?:笔记|页面|文档)|另建/;

const clausesOf = (question: string) =>
  question
    .toLocaleLowerCase()
    .split(
      /[.!?;\n。！？；]+|,\s*(?:and|then)\s+|\s+and\s+(?=(?:then\s+)?(?:add|put|write|save|make|create|insert|append)\b)|，(?:然后|再|并且?)?/u,
    )
    .map((clause) => clause.trim())
    .filter(Boolean);

function englishKind(clause: string): RequestKind | "asks" | null {
  if (MORE_OF.test(clause)) return "edit";
  if (MAKE_INTO.test(clause)) return DOCUMENT.test(clause) ? "answer" : "edit";
  const verb = VERB.exec(clause)?.[1];
  if (!verb) {
    if (MAYBE.test(clause)) return "unclear";
    return ASKS.test(clause) ? "asks" : null;
  }
  const document = DOCUMENT.test(clause);
  const own = OWN.test(clause);
  if (DESTINATION.test(clause)) return "write";
  if (/^summari[sz]e$/.test(verb)) return "answer";
  // "Put simply, what does it mean?" asks; "clarify this paragraph" may ask to rewrite it.
  if (IS_EXPLAIN.test(verb)) return OWN_NAMED.test(clause) && !document ? "unclear" : "answer";
  if (IS_EDIT.test(verb)) return document && !own ? "answer" : own ? "edit" : "unclear";
  // "write it longer", "put this more simply": the User's own text, rewritten.
  if (
    own &&
    !document &&
    /\b(?:longer|shorter|more \p{L}+|less \p{L}+|simpler|clearer)\b/u.test(clause)
  ) {
    return "edit";
  }
  return "write";
}

function chineseKind(clause: string): RequestKind | "asks" | null {
  if (ZH_MORE_OF.test(clause)) return "edit";
  const verb = ZH_VERB.exec(clause)?.[1];
  if (!verb) return ZH_ASKS.test(clause) || /[吗呢]$/.test(clause) ? "asks" : null;
  const document = ZH_DOCUMENT.test(clause);
  const own = ZH_OWN.test(clause);
  if (ZH_DESTINATION.test(clause)) return "write";
  if (ZH_IS_EXPLAIN.test(verb))
    return ZH_OWN_NAMED.test(clause) && !document ? "unclear" : "answer";
  if (ZH_IS_EDIT.test(verb)) return document && !own ? "answer" : own ? "edit" : "unclear";
  // "把它写长一点": 写 of the User's own text is a rewrite.
  if (own && !document && verb === "写") return "edit";
  return "write";
}

const clauseKind = (clause: string): RequestKind | "asks" | null =>
  /\p{Script=Han}/u.test(clause) ? chineseKind(clause) : englishKind(clause);

export function routeRequest(question: string): RequestRoute {
  const kinds = clausesOf(question).map(clauseKind);
  const asked = kinds.filter((kind): kind is RequestKind => kind !== null && kind !== "asks");
  const kind: RequestKind = asked.includes("edit")
    ? "edit"
    : asked.includes("write")
      ? "write"
      : asked.includes("unclear")
        ? "unclear"
        : "answer";
  const newMind =
    kind === "write" && (NEW_MIND.test(question.toLocaleLowerCase()) || ZH_NEW_MIND.test(question));
  return { kind, newMind };
}
