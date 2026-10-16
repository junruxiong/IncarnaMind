/**
 * Whether a Question in the assistant asks for an answer or for text in a
 * Mind, from its words, in English and Chinese: "write" new text, "edit"
 * text already there, "answer" in the reply, or "unclear". `newMind`: it
 * asks for a new page or note.
 *
 * Words can't tell every request apart, so only the plain ones route: an
 * instruction, not a question, that names the User's own text ("this
 * paragraph", "my summary", "第二段") or where new text goes ("a summary
 * page", "add this to my notes", "加到我的笔记里"). A follow-up with no
 * object ("make it longer", "简单一点") is an edit only when the
 * conversation's last Answer changed a Mind (`followsAChange`); after a
 * question in the same request it is about the answer. "unclear" behaves as
 * an Answer always has: the model chooses, and its changes wait for the User.
 */
export type RequestKind = "answer" | "write" | "edit" | "unclear";

export interface RequestRoute {
  kind: RequestKind;
  newMind: boolean;
}

// "none" asks for nothing ("thanks") and, unlike "answer", leaves a later "make it longer" alone.
type ClauseKind = RequestKind | "follow-up" | "continue" | "polite" | "none";

interface ClauseRead {
  kind: ClauseKind;
  asks: boolean;
  document: boolean;
}

const LEAD =
  /^(?:(?:ok(?:ay)?|so|now|then|and|also|just|please|pls|i mean|actually|instead|no|nope|yes|yeah|thanks|thank you|great|good|nice|perfect|cool|go ahead and|let's|let us|help me|i want you to|i'd like you to|i would like you to)[,!]?\s+)*/;
const POLITE = /^(?:can|could|would|will) you\s+(?:please\s+)?/;
const ASKS =
  /^(?:what|what's|why|how|when|where|who|whom|which|whose|is|isn't|are|aren't|was|were|does|doesn't|do|don't|did|didn't|should|shall|may|might|must|has|have|had|would|could|can|will)\b/;
const IDIOM =
  /^(?:make sense of|cut to the chase|drop the jargon|correct me|update me|walk me through|fill me in|bring me up to speed|tell me|show me|give me|remind me)\b/;
const REPLY =
  /\b(?:your|the|that|this)\s+(?:last\s+|previous\s+|first\s+)?(?:answer|reply|response|explanation)\b|\bwhat you (?:just\s+)?(?:said|wrote)\b|^(?:say|put) (?:that|it|this) (?:more |less )?(?:simply|differently|another way|in other words|in plain|plainly|clearly|briefly)/;
const MORE_OF =
  /^(?:(?:a bit|a little|much|even|slightly|way)\s+)?(?:longer|shorter|simpler|clearer|tighter|briefer|more \p{L}+|less \p{L}+)(?:,?\s+(?:please|pls|again|still|then))?$/u;
const CONTINUE = /^(?:continue|go on|keep going)(?:,?\s+please)?$/;
const KEEP_IT = /^keep (?:it|this|that|them)\b/;

const CREATE = "write|draft|compose|create|make|generate|produce|prepare|start";
const PLACE_IT = "add|insert|append|put|save|copy|jot down|note down";
const EDIT =
  "shorten|lengthen|condense|trim|tighten|simplify|rewrite|rephrase|reword|paraphrase|translate|fix|correct|proofread|polish|improve|edit|revise|restructure|reorgani[sz]e|reformat|remove|delete|cut|drop|replace|change|update|rename|merge|split|move|tidy up|clean up|expand|extend|turn";
const EXPLAIN = "clarify|elaborate on|elaborate|expand on|explain|put simply|unpack";
const VERB = new RegExp(
  `^(${EXPLAIN}|summari[sz]e|${EDIT}|${CREATE}|${PLACE_IT})\\b\\s*(.*)$`,
  "u",
);
const IS_EXPLAIN = new RegExp(`^(?:${EXPLAIN})$`);
const IS_CREATE = new RegExp(`^(?:${CREATE}|${PLACE_IT})$`);
const IS_WRITE = /^(?:write|draft|compose|create|generate|produce|prepare|start)$/;
const IS_REMOVE = /^(?:delete|remove|cut|drop)$/;
const WHOLE = /^(?:my|the|this|our)\s+(?:whole\s+)?(?:notes?|draft|text|page|writing)$/;
const MANNER =
  /\b(?:more|less|longer|shorter|simpler|clearer|better|again|differently|in (?:english|chinese|french|german|spanish|japanese|plain \p{L}+)|as (?:an?)\b)/u;
// "Translate this sentence: …" brings its own text, not the Mind's.
const INLINE =
  /\b(?:this|these|the following)\s+(?:sentences?|paragraphs?|text|lines?|passage|words?)\s*[:：]/;

const DOCUMENT =
  /\b(?:the|this|that|these|those|its)\s+(?:paper|papers|article|articles|study|studies|abstract|authors?|tutorial|filing|filings|10-k|book|chapter|pdf|pdfs|document|documents|report|reports)\b|\bthe authors?\b|\bwhat (?:the authors?|they|it) (?:means?|say|says|claims?)\b|\b(?:paper|article|study|document|report)'s\b/;
const OWN_NAMED =
  /\b(?:this|that|the|my|our)\s+(?:(?:first|second|third|fourth|fifth|last|next|opening|closing|whole)\s+)?(?:paragraphs?|sentences?|sections?|lines?|summary|intro|introduction|conclusion|notes?|draft|text|headings?|list|bullets?|bullet points?|table|selection|wording|writing|parts?|typos?|spelling|grammar|formatting|tone)\b/;
const BARE_OBJECT =
  /^(?:(?:it|this|that|them|these|those)\b(?!\s+(?:paper|papers|article|study|document|documents|report|reports)\b)|(?:into|in|to|a bit|a little|more|less|again|please)\b|$)/;
const INTO_PLACE =
  /\b(?:to|into|onto)\s+(?:my|the|this|a|a new|our)\s+(?:new\s+)?(?:notes?|page|mind|notebook)\b|\b(?:in|on)\s+(?:a\s+)?new\s+(?:page|note|mind)\b/;
const IN_PLACE = /\b(?:in|on)\s+(?:my|the|this|our)\s+(?:notes?|page|mind|notebook)\b/;
const NEW_PAGE =
  /\b(?:an?|new|separate|another|fresh)\s+(?:[\p{L}-]+\s+){0,2}(?:page|note|handout)\b|^(?:make|take|write) notes\b/u;
const PIECE =
  /^(?:write|draft|compose|prepare)\s+(?:me\s+)?(?:an?|one)\s+(?:[\p{L}-]+\s+){0,2}(?:summary|overview|outline|brief|briefing|memo|handout|write-?up)\b/u;
const ADD_SECTION =
  /^(?:add|insert|append)\s+(?:an?|another|one more|a new)\s+(?:[\p{L}\p{N}'’-]+\s+){0,3}(?:section|paragraph|heading|bullet|bullet points|list|table)\b/u;
const NEW_MIND =
  /\b(?:new|separate|another|fresh|an?)\s+(?:[\p{L}-]+\s+){0,2}(?:page|note|notes|mind|document|doc|handout|report|file)\b(?!\s+(?:to|in|into|inside)\s+(?!(?:an?|another|new)\b))|\b(?:make|take|write) notes\b/u;

const ZH_LEAD = /^(?:请|麻烦你?|帮我|给我|帮忙|再|接着|然后|我想让你|那就|就|你)*/;
const ZH_POLITE =
  /^你?(?:可以|能不能|能否|可不可以|能)(?:帮我|帮忙)?|(?<![更会很不太得])(?:可以|行|好)吗$/;
const ZH_ASKS =
  /^(?:什么|为什么|怎么|怎样|如何|哪|谁|多少|是否|是不是|有没有)|[吗呢]$|会不会|是不是|要不要|好不好|对不对|行不行|有没有|对吗|怎么样$|如何$/;
const ZH_REPLY = /刚才的回答|你的回答|上面的回答|这个回答|上一个回答|你刚才说的|你的解释/;
const ZH_MORE_OF =
  /^(?:写得|改得|总结得|说得)?(?:再|更)?(?:长|短|正式|简洁|简单|清楚|详细|具体)(?:一点|一些|些)?(?:吧|啊)?$/;
const ZH_CONTINUE = /^(?:继续|接着写|继续写)(?:吧)?$/;
const ZH_SUMMARISE = "总结成|整理成|汇总成|归纳成|总结|概括|归纳|汇总|整理";
const ZH_CREATE = "写|撰写|起草|生成|做成|新建|创建";
const ZH_PLACE_IT = "添加|加上|加入|加到|补充|插入|记下|记到|放到|存到|加";
const ZH_EDIT =
  "缩短|改短|改长|加长|写长|写短|扩写|扩展|扩充|精简|压缩|简化|改写|重写|润色|翻译|修改|改成|改为|改得|写得|改一下|调整|删除|删掉|删去|去掉|移除|替换|更新|校对|修正|纠正|重新组织|改";
const ZH_EXPLAIN = "解释|说明|阐述|讲讲|讲解|展开讲|展开说";
const ZH_VERB = new RegExp(
  `^(?:在[^，。！？]{1,20}?(?:里|中|内|上|后面|前面|末尾|最后))?(把[^，。！？]{1,40})?(${ZH_EXPLAIN}|${ZH_SUMMARISE}|${ZH_EDIT}|${ZH_CREATE}|${ZH_PLACE_IT})(.*)$`,
);
const ZH_IS_EXPLAIN = new RegExp(`^(?:${ZH_EXPLAIN})$`);
const ZH_IS_SUMMARISE = new RegExp(`^(?:${ZH_SUMMARISE})$`);
const ZH_IS_CREATE = new RegExp(`^(?:${ZH_CREATE}|${ZH_PLACE_IT})$`);
const ZH_DOCUMENT = /论文|文章|作者|原文|文中|这篇|这份文档|这份报告|文档的|报告的/;
const ZH_OWN_NAMED =
  /这段|这一段|第[一二三四五六七八九十\d]+段|最后一段|这部分|这一部分|这句|这一句|我的(?:总结|笔记|段落|摘要|草稿|文章|报告)|选中的|(?:关于|有关)[^，。！？]{1,20}的部分/;
const ZH_BARE_OBJECT = /^把(?:它们?|这个|这些|其)$/;
const ZH_BARE_REST =
  /^(?:一下|一点|一些|下|得[^，。！？]{0,6}|[长短]一?[点些]|成中文|成英文|为中文|为英文)?(?:吧|啊)?$/;
const ZH_INTO_PLACE =
  /(?:加到|添加到|放到|写到|写进|记到|存到|加入|放进|放在|写在|记在)(?:我的|这个|这份)?(?:笔记|页面|Mind)|新建|一[篇份个](?:新的?)?(?:总结)?(?:笔记|页面|文档)/;
const ZH_IN_PLACE = /(?:在|进)(?:我的)?(?:笔记|页面)(?:里|中|上)/;
const ZH_PIECE = /^写一[份篇个][^，。！？]{0,30}?(?:总结|概述|报告|备忘|讲义|笔记|页面)/;
const ZH_ADD_SECTION =
  /^(?:添加|加上|加入|补充|插入|加)一[个段节][^，。！？]{0,10}?(?:部分|章节|段落|小节|段)/;
const ZH_NEW_MIND =
  /新建|新的?(?:笔记|页面|文档|Mind)|一[篇份个](?:新的?)?(?:总结)?(?:笔记|页面|文档)|另建/;

function clausesOf(question: string): { text: string; asked: boolean }[] {
  const sentences =
    question.toLocaleLowerCase().match(/[^.!?;\n。！？；]+[.!?;\n。！？；]*/gu) ?? [];
  return sentences.flatMap((sentence) => {
    const asked = /[?？]/.test(sentence);
    return sentence
      .replace(/[.!?;\n。！？；]+$/u, "")
      .split(
        /,\s*(?:and|then)\s+|\s+and\s+(?=(?:then\s+)?(?:add|put|write|save|make|create|insert|append)\b)|，(?:然后|再|并且?)?/u,
      )
      .map((text) => text.trim())
      .filter(Boolean)
      .map((text) => ({ text, asked }));
  });
}

function englishImperative(rest: string, document: boolean): ClauseKind {
  const [, verb = "", object = ""] = VERB.exec(rest) ?? [];
  if (verb && !IS_EXPLAIN.test(verb) && INTO_PLACE.test(rest)) return "write";
  if (IDIOM.test(rest) || REPLY.test(rest)) return "answer";
  if (MORE_OF.test(rest) || KEEP_IT.test(rest)) return "follow-up";
  if (CONTINUE.test(rest)) return "continue";
  if (!verb) return "none";
  if (INLINE.test(rest)) return "unclear";
  const own = !document && OWN_NAMED.test(rest.replace(IN_PLACE, " "));
  if (IS_EXPLAIN.test(verb)) return own ? "unclear" : "answer";
  if (IS_CREATE.test(verb) && (NEW_PAGE.test(rest) || PIECE.test(rest) || ADD_SECTION.test(rest)))
    return "write";
  if (/^summari[sz]e$/.test(verb)) return "answer";
  if (own) {
    // "Delete my notes" may mean the Mind itself, which no edit removes.
    if (IS_REMOVE.test(verb) && WHOLE.test(object)) return "unclear";
    // "Write the conclusion" asks for new text; "write the summary in Chinese" changes it.
    return IS_WRITE.test(verb) && !MANNER.test(object) ? "write" : "edit";
  }
  if (IS_CREATE.test(verb) && IN_PLACE.test(rest)) return "write";
  if (document) return "answer";
  return BARE_OBJECT.test(object) ? "follow-up" : "unclear";
}

function englishClause(clause: string, asked: boolean): ClauseRead {
  const document = DOCUMENT.test(clause);
  let rest = clause.replace(LEAD, "");
  const polite = POLITE.test(rest);
  if (polite) rest = rest.replace(POLITE, "").replace(LEAD, "");
  else if (asked || ASKS.test(rest)) return { kind: "answer", asks: true, document };
  const kind = englishImperative(rest, document);
  if (!polite) return { kind, asks: false, document };
  return { kind: kind === "answer" || kind === "none" ? "answer" : "polite", asks: true, document };
}

function chineseImperative(clause: string, document: boolean): ClauseKind {
  const [, object = "", verb = "", after = ""] = ZH_VERB.exec(clause) ?? [];
  if (verb && !ZH_IS_EXPLAIN.test(verb) && ZH_INTO_PLACE.test(clause)) return "write";
  if (ZH_REPLY.test(clause)) return "answer";
  if (ZH_MORE_OF.test(clause)) return "follow-up";
  if (ZH_CONTINUE.test(clause)) return "continue";
  if (!verb) return "none";
  const own = ZH_OWN_NAMED.test(clause.replace(ZH_IN_PLACE, "")) && !document;
  if (ZH_IS_EXPLAIN.test(verb)) return own ? "unclear" : "answer";
  if (ZH_IS_CREATE.test(verb) && (ZH_PIECE.test(clause) || ZH_ADD_SECTION.test(clause)))
    return "write";
  if (ZH_IS_SUMMARISE.test(verb)) return "answer";
  if (own) return verb === "写" && !/[得成长短]/.test(after) ? "write" : "edit";
  if (ZH_IS_CREATE.test(verb) && ZH_IN_PLACE.test(clause)) return "write";
  if (document) return "answer";
  const bare = object ? ZH_BARE_OBJECT.test(object) : ZH_BARE_REST.test(after);
  return bare ? "follow-up" : "unclear";
}

function chineseClause(clause: string, asked: boolean): ClauseRead {
  const document = ZH_DOCUMENT.test(clause);
  let rest = clause.replace(ZH_LEAD, "");
  const polite = ZH_POLITE.test(rest);
  if (polite) {
    rest = rest
      .replace(ZH_POLITE, "")
      .replace(ZH_LEAD, "")
      .replace(/[吗呢]$/, "");
  } else if (asked || ZH_ASKS.test(rest)) {
    return { kind: "answer", asks: true, document };
  }
  const kind = chineseImperative(rest, document);
  if (!polite) return { kind, asks: false, document };
  return { kind: kind === "answer" || kind === "none" ? "answer" : "polite", asks: true, document };
}

const readClause = ({ text, asked }: { text: string; asked: boolean }): ClauseRead =>
  /\p{Script=Han}/u.test(text) ? chineseClause(text, asked) : englishClause(text, asked);

export function routeRequest(
  question: string,
  { followsAChange }: { followsAChange: boolean },
): RequestRoute {
  const kinds: ClauseKind[] = [];
  let afterAQuestion = false;
  for (const read of clausesOf(question).map(readClause)) {
    // "What does the abstract say? Translate it": "it" is the answer, or the Document.
    kinds.push(afterAQuestion && read.kind !== "write" ? "answer" : read.kind);
    afterAQuestion ||= read.asks || read.document || read.kind === "answer";
  }
  const kind: RequestKind = kinds.includes("edit")
    ? "edit"
    : kinds.includes("write")
      ? "write"
      : kinds.includes("follow-up") && followsAChange
        ? "edit"
        : kinds.includes("continue") && followsAChange
          ? "write"
          : kinds.some((each) => each === "follow-up" || each === "polite" || each === "unclear")
            ? "unclear"
            : "answer";
  const newMind =
    kind === "write" && (NEW_MIND.test(question.toLocaleLowerCase()) || ZH_NEW_MIND.test(question));
  return { kind, newMind };
}
