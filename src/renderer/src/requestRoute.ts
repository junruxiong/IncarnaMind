/** Kept to one signature with the core's routing, which takes its place once it lands. */
export type RequestKind = "answer" | "write" | "edit" | "unclear";

export interface RequestRoute {
  kind: RequestKind;
  newMind: boolean;
}

const UNDERSTAND =
  /^(?:please\s+)?(?:make sense of|explain|tell me|help me understand)\b|^(?:解释|帮我理解)/i;
const NEW_MIND = /\bnew (?:note|page|mind|document)\b|新(?:笔记|页面|文档)|新建/i;
const WRITE =
  /^(?:please\s+)?(?:write|draft|summari[sz]e|make an? (?:list|summary|table|outline|note|page)\b)|^(?:写|起草|总结|列出)/i;
const QUESTION =
  /[?？]\s*$|^(?:what|why|how|who|when|where|which|is|are|does|do|did|can|could|should|would)\b|[吗呢]\s*[?？]?\s*$/i;
const EDIT =
  /^(?:please\s+)?(?:make|replace|shorten|lengthen|expand|rewrite|rephrase|reword|translate|fix|correct|tighten|simplify|polish|improve|edit|change|turn|remove|delete|cut|trim|condense|clarify)\b|^(?:请|帮我)?(?:(?:把|将).+?(?:改|缩短|精简|扩写|重写|翻译|润色|删|去掉|简化)|缩短|精简|扩写|重写|改写|翻译|润色|删除|删掉|去掉|简化|改得)/iu;
const EDIT_WORD =
  /\b(?:make|replace|shorten|rewrite|rephrase|translate|fix|correct|tighten|simplify|polish|improve|edit|change|remove|delete|cut)\b|改|缩短|重写|翻译|润色|删/iu;

export function routeRequest(question: string): RequestRoute {
  const text = question.trim();
  if (UNDERSTAND.test(text)) return { kind: "answer", newMind: false };
  const newMind = NEW_MIND.test(text);
  if (newMind || WRITE.test(text)) return { kind: "write", newMind };
  if (QUESTION.test(text)) {
    return { kind: EDIT_WORD.test(text) ? "unclear" : "answer", newMind: false };
  }
  return { kind: EDIT.test(text) ? "edit" : "answer", newMind: false };
}
