/**
 * "Create a new note called Fieldwork plan" (新建一篇叫“田野计划”的笔记) asks
 * for an empty Mind with that title, and nothing else: it is made at once,
 * with no AI step (#202). A request that asks for more than the title, such
 * as what to write in it ("…called Plan with three steps", "…summarising
 * this"), goes to the assistant, which writes it. Only data.
 */

const ENGLISH =
  /^(?:please\s+)?(?:create|make|start|add|open)\s+(?:me\s+)?(?:a\s+|an\s+)?(?:new\s+)?(?:empty\s+|blank\s+)?(?:note|mind|page)\s+(?:called|named|titled)\s+(.+?)[.!]?$/iu;

const CHINESE_NAMED_FIRST =
  /^(?:请)?(?:帮我)?(?:新建|创建|建)(?:一(?:篇|个|页))?(?:叫|叫做|名为|标题为)(.+?)的\s*(?:新)?(?:空白)?(?:笔记|Mind|页面)[。！]?$/u;

const CHINESE_NAMED_AFTER =
  /^(?:请)?(?:帮我)?(?:新建|创建|建)(?:一(?:篇|个|页))?\s*(?:新的?)?(?:空白)?(?:笔记|Mind|页面)[，,]?\s*(?:叫|叫做|名为|标题为|标题是)(.+?)[。！]?$/u;

/** Words in a title that ask for content as well: the assistant writes that. */
const ASKS_FOR_MORE =
  /\b(?:with|about|from|summari[sz]ing|summari[sz]e|listing|covering|and|that|which|containing|based on)\b|关于|总结|概括|整理|包含|并且|并|内容|里面|写上/iu;

const QUOTES = /^[“"'‘「《](.*)[”"'’」》]$/u;

/** The title a request for an empty Mind gives, or null when it asks for anything more. */
export function newNoteTitle(text: string): string | null {
  const line = text.trim();
  if (line === "" || line.includes("\n")) return null;
  const match =
    ENGLISH.exec(line) ?? CHINESE_NAMED_FIRST.exec(line) ?? CHINESE_NAMED_AFTER.exec(line);
  const named = match?.[1]?.trim();
  if (!named) return null;
  const quoted = QUOTES.exec(named);
  const title = (quoted ? (quoted[1] ?? "") : named).trim();
  if (title === "" || title.length > 80) return null;
  // Quoted, the title is all of it; unquoted, words that ask for content mean it isn't.
  if (!quoted && ASKS_FOR_MORE.test(title)) return null;
  return title;
}
