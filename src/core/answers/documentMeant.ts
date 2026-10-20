/**
 * The one Document a request points at (#1334), in English and Chinese: "this document", "this
 * one", "this agreement", "这份劳动合同", "这篇论文", "这个文件". It is, in turn: the Document the
 * Mind in focus is the working copy of; the only Document in scope; the one whose name holds the
 * words pointed with ("this lease" in "Lease 3", "这份劳动合同" in "劳动合同"); the one the
 * conversation's last Answer cited. Two or more that fit equally, or nothing that tells, leave it
 * to the User to say which.
 */

export interface NamedDocument {
  id: string;
  name: string;
}

export type DocumentMeant =
  | { kind: "one"; document: NamedDocument }
  /** Ask the User which: these fit equally; none, nothing tells. */
  | { kind: "which"; documents: NamedDocument[] };

const EN_NOUN =
  "(?:document|doc|file|pdf|one|paper|report|agreement|contract|lease|memo|letter|deck|presentation|article|study|résumé|resume|cv|transcript|spreadsheet|workbook|invoice|policy|filing|thesis|book|chapter|minutes|proposal|nda|judgment|judgement|ruling|prospectus)";
const EN_WORD =
  "(?!(?:is|was|are|a|an|the|and|or|of|to|for|in|on|at|by|with|from|that|it|my|your|our|their)\\b)[\\p{L}\\p{N}'’-]+";
const EN_POINTER = new RegExp(
  `\\bthis\\s+((?:${EN_WORD}\\s+){0,2}?${EN_NOUN})(?![\\p{L}\\p{N}-])`,
  "iu",
);
const EN_GENERIC = /^(?:document|doc|file|pdf|one)$/;

// 这个 points at anything ("这个问题"): only at a file by these words.
const ZH_POINTER =
  /这(?:份|篇|本|封|个(?=文件|文档|合同|协议|报告|论文|简历|PDF|pdf))([\p{Script=Han}A-Za-z0-9]{1,16})/u;
// "这份总结", "这篇笔记", "这篇里": the User's own text.
const ZH_OWN =
  /^(?:笔记|总结|摘要|页面|页|段|草稿|回答|答案|部分|章节|表格|图表|内容|问题|文字|里|中)/;
const ZH_GENERIC = /^(?:文件|文档|资料|材料|pdf)/i;
const SHORTEST_ZH_NAME = 2;

const escaped = (text: string) => text.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

/** The Documents whose names hold the words a request points with; empty when it names none. */
type ByName = (documents: readonly NamedDocument[]) => NamedDocument[];

function pointerOf(question: string): ByName | null {
  const english = EN_POINTER.exec(question);
  if (english) {
    const words = (english[1] ?? "").toLocaleLowerCase().split(/\s+/);
    if (EN_GENERIC.test(words.at(-1) ?? "")) return () => [];
    return (documents) => {
      for (let from = 0; from < words.length; from++) {
        const tail = new RegExp(
          `(?:^|[^\\p{L}\\p{N}])${escaped(words.slice(from).join(" "))}s?(?=$|[^\\p{L}\\p{N}])`,
          "iu",
        );
        const named = documents.filter((each) => tail.test(each.name));
        if (named.length > 0) return named;
      }
      return [];
    };
  }
  const chinese = ZH_POINTER.exec(question);
  const run = chinese?.[1] ?? "";
  if (!chinese || ZH_OWN.test(run)) return null;
  if (ZH_GENERIC.test(run)) return () => [];
  return (documents) => {
    for (let length = run.length; length >= SHORTEST_ZH_NAME; length--) {
      const words = run.slice(0, length).toLocaleLowerCase();
      const named = documents.filter((each) => each.name.toLocaleLowerCase().includes(words));
      if (named.length > 0) return named;
    }
    return [];
  };
}

/**
 * The Document `question` points at, among `documents` (those in scope); null when it points at
 * none. `original`: the Document the Mind in focus is the working copy of, if in scope. `cited`:
 * the Documents the conversation's last Answer cited, by id.
 */
export function documentMeant(
  question: string,
  {
    documents,
    original,
    cited,
  }: {
    documents: () => readonly NamedDocument[];
    original: NamedDocument | null;
    cited: readonly string[];
  },
): DocumentMeant | null {
  const byName = pointerOf(question);
  if (!byName) return null;
  if (original) return { kind: "one", document: original };
  const all = documents();
  const [only] = all;
  if (only === undefined) return null;
  if (all.length === 1) return { kind: "one", document: only };
  const wasCited = (each: NamedDocument) => cited.includes(each.id);
  const named = byName(all);
  const fits = named.length > 0 ? named : all.filter(wasCited);
  const [first, second] = fits;
  if (first && !second) return { kind: "one", document: first };
  const [lastCited, alsoCited] = fits.filter(wasCited);
  if (lastCited && !alsoCited) return { kind: "one", document: lastCited };
  return { kind: "which", documents: fits };
}

const ASKED_WHICH_AT_MOST = 5;

/** What the model is told of the Document a request points at; "" when none. */
export function documentMeantInstructions(meant: DocumentMeant | null): string {
  if (meant === null) return "";
  if (meant.kind === "one") {
    return `The User points at one Document, "${meant.document.name}", so this Question searches it alone. Answer, or write, about it alone, never about their other Documents.`;
  }
  const names = meant.documents.map((each) => `"${each.name}"`);
  const fit =
    names.length >= 2 && names.length <= ASKED_WHICH_AT_MOST
      ? ` ${names.slice(0, -1).join(", ")} and ${names.at(-1)} fit equally.`
      : "";
  return `The User points at one Document, but not plainly at which.${fit} Ask them which they mean, in one line${fit ? " naming those" : ""}, and write nothing else: no Mind changes until they say.`;
}
