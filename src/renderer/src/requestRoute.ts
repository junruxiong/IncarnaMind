/**
 * Whether what the User typed asks for an answer, for new text, or for a
 * change to text already in a Mind; "unclear" when it can't be told, and
 * then nothing applies at once (#880). In English and Chinese.
 */
export type RequestKind = "answer" | "write" | "edit" | "unclear";

export interface RequestRoute {
  kind: RequestKind;
  /** It asks for a new page, note or document. */
  newMind: boolean;
}

const QUESTION =
  /[?？]\s*$|^(?:what|why|how|who|when|where|which|is|are|does|do|did|can|could|should|would)\b/i;
const EDIT =
  /^(?:please\s+)?(?:make|shorten|lengthen|expand|rewrite|rephrase|reword|translate|fix|correct|tighten|simplify|polish|improve|edit|change|turn|remove|delete|cut|trim|condense|clarify)\b/i;
const EDIT_ZH =
  /^(?:请|帮我)?(?:(?:把|将).+?(?:改|缩短|精简|扩写|重写|翻译|润色|删|去掉|简化)|缩短|精简|扩写|重写|改写|翻译|润色|删除|删掉|去掉|简化|改得)/u;

export function routeRequest(question: string): RequestRoute {
  const text = question.trim();
  if (QUESTION.test(text)) return { kind: "answer", newMind: false };
  if (EDIT.test(text) || EDIT_ZH.test(text)) return { kind: "edit", newMind: false };
  return { kind: "answer", newMind: false };
}
