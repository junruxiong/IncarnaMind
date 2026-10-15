/**
 * Whether a Question in the assistant asks for an answer or for text in a
 * Mind, from its words alone, in English and Chinese: "write" new text,
 * "edit" text already there, "answer" in the reply and change no Mind, or
 * "unclear", whose changes wait for the User instead of applying. `newMind`:
 * it asks for a new page or note.
 */
export type RequestKind = "answer" | "write" | "edit" | "unclear";

export interface RequestRoute {
  kind: RequestKind;
  newMind: boolean;
}

const WRITE =
  "write|draft|compose|create|make|add|insert|put|append|jot down|note down|generate|produce|prepare|start|turn (?:it|this|that|these|them) into|summari[sz]e(?: [\\p{L}\\p{N}-]+){0,5} (?:into|as) |summari[sz]e(?: [\\p{L}\\p{N}-]+){0,5} in (?:a|an|my|the|this|that) (?:new )?(?:page|note|mind|doc|document|section)";
const EDIT =
  "shorten|lengthen|expand|extend|elaborate on|condense|trim|tighten|simplify|clarify|rewrite|rephrase|reword|paraphrase|translate|fix|correct|proofread|polish|improve|edit|revise|restructure|reorgani[sz]e|reformat|remove|delete|cut|drop|replace|change|update|rename|merge|split|move|tidy up|clean up";
/** "make it longer": a change, not something made. */
const MAKE_INTO =
  /^make (?:it|this|that|them|these|those|the [\p{L}\p{N} -]{1,40}?) (?:longer|shorter|clearer|simpler|tighter|briefer|more \p{L}+|less \p{L}+|formal|informal|casual|concise|readable)\b/u;

const LEAD =
  "(?:(?:ok(?:ay)?|so|now|then|and|also|just|please|pls|i mean|actually|instead|go ahead and|let's|help me|i want you to|i'd like you to|i would like you to|can you|could you|would you|will you)[,]?\\s+)*";
const VERB = new RegExp(`^${LEAD}(${EDIT}|${WRITE})\\b`, "u");
const IS_EDIT = new RegExp(`^(?:${EDIT})$`, "u");

/** A clause that asks, rather than tells: a polite "can you…" is a request, not a question. */
const ASKS =
  /^(?:what|why|how|when|where|who|whom|which|whose|is|are|was|were|does|do|did|should|shall|may|might|must|has|have|had)\b/;
const MAYBE = /^(?:should|shall|may|might|could|would|can) (?:i|we)\b/;

const NEW_MIND =
  /\b(?:new|separate|another|fresh|an?)\s+(?:[\p{L}-]+\s+){0,2}(?:page|note|notes|mind|document|doc|handout|report|file)\b|\b(?:make|take|write) notes\b/u;

const ZH_WRITE =
  "写|撰写|起草|生成|整理成|总结成|汇总成|做成|新建|创建|添加|加上|加入|补充|插入|记下";
const ZH_EDIT =
  "缩短|改短|改长|加长|扩写|扩展|扩充|展开|精简|压缩|简化|改写|重写|润色|翻译|修改|改成|改为|改一下|调整|删除|删掉|删去|去掉|移除|替换|更新|校对|修正|纠正|重新组织";
const ZH_VERB = new RegExp(
  `^(?:请|麻烦你?|帮我|给我|你?(?:可以|能不能|能否)(?:帮我)?|再|接着|然后|我想让你|那就|就)*(?:在[^，。！？]{1,20}?(?:里|中|内|上|后面|前面|末尾|最后))?(?:把[^，。！？]{1,40}?)?(${ZH_EDIT}|${ZH_WRITE})`,
);
const ZH_IS_EDIT = new RegExp(`^(?:${ZH_EDIT})$`);
const ZH_ASKS = /^(?:什么|为什么|怎么|怎样|如何|哪|谁|多少|是否|是不是|有没有)/;
const ZH_NEW_MIND = /新建|新的?(?:笔记|页面|文档|Mind)|一[篇份个](?:新的?)?(?:笔记|页面|文档)|另建/;

const clausesOf = (question: string) =>
  question
    .toLocaleLowerCase()
    .split(/[.!?;\n。！？；]+|,\s*(?:and|then)\s+|，(?:然后|再|并且?)/u)
    .map((clause) => clause.trim())
    .filter(Boolean);

function clauseKind(clause: string): RequestKind | "asks" | null {
  if (MAKE_INTO.test(clause.replace(new RegExp(`^${LEAD}`, "u"), ""))) return "edit";
  const verb = VERB.exec(clause)?.[1] ?? ZH_VERB.exec(clause)?.[1];
  if (verb) return IS_EDIT.test(verb) || ZH_IS_EDIT.test(verb) ? "edit" : "write";
  if (MAYBE.test(clause)) return "unclear";
  if (ASKS.test(clause) || ZH_ASKS.test(clause) || /[吗呢]$/.test(clause)) return "asks";
  return null;
}

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
