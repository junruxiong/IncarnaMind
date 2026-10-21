/**
 * The one Document "this" means in a request (#1334), the same way in English and Chinese:
 * "this document", "this one", "这份", "这个", "该合同". What the User has in view decides: the
 * Document open in the focused group, the one the selected text cites, the one the Mind in focus
 * is the working copy of, the one the Mind cites, the one the last Answer cited. The words decide
 * only whether the request says "this", and which Document when what follows is in its name
 * ("this agreement" in "Employment agreement", "这份劳动合同" in "劳动合同").
 */
import * as Y from "yjs";
import { CITATION_NODE } from "../api";

export interface NamedDocument {
  id: string;
  name: string;
}

export type DocumentMeant =
  | { kind: "one"; document: NamedDocument }
  /** These fit the words equally, or none at all, and nothing in view tells which: the User is asked. */
  | { kind: "which"; documents: NamedDocument[] };

/** What the User has in view as they ask, by Document id. */
export interface InView {
  /** The Documents shown in the window's groups, the focused group's first. */
  open: readonly string[];
  /** What the selected text cites; null with nothing selected. */
  selected: readonly string[] | null;
  /** The Document the Mind in focus is the working copy of. */
  original: string | null;
  /** What the Mind in focus cites. */
  mind: readonly string[];
  /** What the conversation's last Answer cited. */
  lastAnswer: readonly string[];
}

// The words that say a Document, the same in both: a "this" before one of them, or before a
// Document's name, points at a Document; "this" alone may be the last Answer, or the User's text.
const EN_DOCUMENT =
  /^(?:document|doc|file|pdf|agreement|contract|report|statement|spreadsheet|sheet|invoice|memo|letter|deck|presentation|paper|article|minutes|lease|policy|proposal|plan|brief|budget|filing|résumé|resume|cv)s?$/i;
const ZH_DOCUMENT =
  "(?:文件|文档|协议|合同|报告|报表|表格|表|发票|备忘录|函|信|演示|幻灯片|论文|文章|纪要|租约|政策|方案|计划|诉状|预算|简历|说明书|手册|章程)";
const EN_THIS = /\bthis\b((?:\s+[\p{L}\p{N}'’-]+){0,3})/iu;
// 这份 and its kind; 此 not in 因此; 该 and 本 only before a Document's word, so not in 我该, 本周 or 成本.
const ZH_THIS = new RegExp(
  `(?:这(?:份|篇|个|张|本|封)|(?<![因如从彼借据至对凭特])此(?![外时后前次刻])|(?:(?<![我你他她们应本活谁])该|(?<![基成日根资样课书剧版文脚账])本)(?=(?:(?!的)[\\p{Script=Han}]){0,4}?${ZH_DOCUMENT}))([\\p{Script=Han}A-Za-z0-9]{0,16})`,
  "u",
);
// What "this" means when it isn't a Document, the same in both: a folder, the User's own text, the chat, a time.
const EN_NOT =
  /^(?:folder|directory|paragraph|section|sentence|page|note|notes|question|answer|chat|week|month|year|quarter|time)\b/i;
const ZH_NOT = /^(?:文件夹|目录|段|节|句|页|笔记|问题|回答|对话|周|月|年|季度|次|时候)/;
const EN_STOP =
  /^(?:one|is|was|are|a|an|the|and|or|with|of|to|for|in|on|at|by|from|about|that|it|as|than|me|us)$/i;
const SHORTEST_ZH_NAME = 2;

const escaped = (text: string) => text.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

interface AfterThis {
  words: string[];
  chinese: boolean;
  /** A Document's word follows "this": "this agreement", "这份合同". */
  saysDocument: boolean;
}

function wordsAfterThis(question: string): AfterThis | null {
  const english = EN_THIS.exec(question);
  if (english) {
    const after = (english[1] ?? "").trim();
    if (EN_NOT.test(after)) return null;
    const words: string[] = [];
    for (const word of after.split(/\s+/)) {
      if (word === "" || EN_STOP.test(word)) break;
      words.push(word.toLocaleLowerCase());
    }
    return { words, chinese: false, saysDocument: words.some((word) => EN_DOCUMENT.test(word)) };
  }
  const chinese = ZH_THIS.exec(question);
  if (!chinese) return null;
  const run = chinese[1] ?? "";
  if (ZH_NOT.test(run)) return null;
  return {
    words: run === "" ? [] : [run],
    chinese: true,
    saysDocument: new RegExp(`^(?:(?!的)[\\p{Script=Han}]){0,4}?${ZH_DOCUMENT}`, "u").test(run),
  };
}

function namedBy(
  { words, chinese }: AfterThis,
  documents: readonly NamedDocument[],
): NamedDocument[] {
  if (chinese) {
    const run = words[0] ?? "";
    for (let length = run.length; length >= SHORTEST_ZH_NAME; length--) {
      const part = run.slice(0, length).toLocaleLowerCase();
      const named = documents.filter((each) => each.name.toLocaleLowerCase().includes(part));
      if (named.length > 0) return named;
    }
    return [];
  }
  for (let count = words.length; count > 0; count--) {
    const phrase = new RegExp(
      `(?:^|[^\\p{L}\\p{N}])${escaped(words.slice(0, count).join(" "))}s?(?=$|[^\\p{L}\\p{N}])`,
      "iu",
    );
    const named = documents.filter((each) => phrase.test(each.name));
    if (named.length > 0) return named;
  }
  return [];
}

/** Every Document the Citations in these Blocks cite, by id, each once. */
export function citedIn(blocks: readonly Y.XmlElement[]): string[] {
  const found = new Set<string>();
  const visit = (element: Y.XmlElement) => {
    if (element.nodeName === CITATION_NODE) {
      const id = element.getAttribute("documentId");
      if (typeof id === "string") found.add(id);
      return;
    }
    for (const child of element.toArray()) if (child instanceof Y.XmlElement) visit(child);
  };
  for (const block of blocks) visit(block);
  return [...found];
}

/**
 * The Document `question` means by "this", among `documents` (those in scope); null when it
 * doesn't say "this" of one, or nothing tells which. `folder`: the scope is a Folder, where
 * "this" with nothing in view means its Documents, not one to ask about.
 */
export function documentMeant(
  question: string,
  {
    documents,
    inView,
    folder,
  }: { documents: () => readonly NamedDocument[]; inView: InView; folder: boolean },
): DocumentMeant | null {
  const after = wordsAfterThis(question);
  // Selected text that cites nothing is what "this" means.
  if (!after || inView.selected?.length === 0) return null;
  const all = documents();
  const byId = new Map(all.map((each) => [each.id, each]));
  const among = (ids: readonly string[]) =>
    ids.flatMap((id) => {
      const found = byId.get(id);
      return found ? [found] : [];
    });
  const only = (ids: readonly string[]) => {
    const [one, more] = among(ids);
    return more ? undefined : one;
  };
  const open = among(inView.open);
  const named = namedBy(after, all);
  const [first, second] = named;
  if (first && !second) return { kind: "one", document: first };
  // "This" alone, with nothing on screen, is the last Answer or the User's own text.
  const pointsAtDocument = after.saysDocument || first !== undefined;
  const viewed = [
    ...(open.length === 1 ? open : []),
    only(inView.selected ?? []),
    ...(pointsAtDocument
      ? [
          inView.original ? byId.get(inView.original) : undefined,
          only(inView.mind),
          only(inView.lastAnswer),
        ]
      : []),
  ].filter((each): each is NamedDocument => each !== undefined);
  if (first) {
    const seen = [...open, ...viewed].find((each) => named.includes(each));
    if (seen) return { kind: "one", document: seen };
    return folder ? null : { kind: "which", documents: named };
  }
  if (open.length > 1) return { kind: "which", documents: open };
  const [inSight] = viewed;
  if (inSight) return { kind: "one", document: inSight };
  if (!pointsAtDocument) return null;
  const [alone, another] = all;
  if (alone && !another) return { kind: "one", document: alone };
  return alone && !folder ? { kind: "which", documents: [] } : null;
}

const ASKED_WHICH_AT_MOST = 5;

/** What the model is told of the Document a request means by "this"; "" when none. */
export function documentMeantInstructions(meant: DocumentMeant | null): string {
  if (meant === null) return "";
  if (meant.kind === "one") {
    return `By "this", the User means the Document "${meant.document.name}": answer or write about it, and search it by name (search_documents with its document). When the Question also asks about their other Documents, such as one like it or a comparison, search those too.`;
  }
  const names = meant.documents.map((each) => `"${each.name}"`);
  if (names.length === 0) {
    return `By "this", the User may mean one of their Documents, and nothing they have open or cite shows which. If more than one could be meant, ask which in one line and write nothing else; otherwise answer as asked.`;
  }
  const fit =
    names.length <= ASKED_WHICH_AT_MOST
      ? ` ${names.slice(0, -1).join(", ")} and ${names.at(-1)} fit equally.`
      : "";
  return `By "this", the User means one Document, but not plainly which.${fit} Ask them which they mean, in one line${fit ? " naming those" : ""}, and write nothing else: no Mind changes until they say.`;
}
