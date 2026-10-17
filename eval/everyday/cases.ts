/**
 * The everyday requests (README.md): what Users ask the assistant every day,
 * in English and Chinese, beside a Mind or not, about their Documents or
 * their own text. The owner's own conversation comes first. Each says what
 * its Answer must do; `checks.ts` holds every Answer to it.
 */
import type { Expected } from "./checks";

/** The Documents a request is asked over: two arXiv papers, the repository's seven sample PDFs, or none. */
export type LibraryName = "papers" | "reports";

export interface EverydayCase {
  id: string;
  language: "en" | "zh-CN";
  library: LibraryName;
  /** The Mind it is asked from, its Blocks in Markdown-ish lines ("## " a heading); null: none in focus. */
  mind: { title: string; blocks: readonly string[] } | null;
  /** The whole Mind is selected as it is asked. */
  selectAll?: boolean;
  /** Asked first, in the same conversation, and not checked. */
  before?: readonly string[];
  /** The last of `before` is put into the Mind with Add to note, as a whole Answer, before the request. */
  addToNote?: true;
  request: string;
  expected: Expected;
  /** Hard enough to ask Claude Sonnet 5.5 as well (INCARNAMIND_EVAL_EVERYDAY=hard). */
  hard?: boolean;
}

/** The share of its runs a request must pass: two in three, one in three for a hard one. */
export const floorOf = (each: EverydayCase): number => (each.hard ? 1 / 3 : 2 / 3);

const ANSWER = { write: false, cite: true, coverAll: false } as const;
const ANSWER_ALL = { write: false, cite: true, coverAll: true } as const;
const WRITE = { write: true, cite: true, coverAll: false } as const;
const WRITE_ALL = { write: true, cite: true, coverAll: true } as const;
const EDIT = { write: true, cite: false, coverAll: false } as const;
const NOTHING = { write: false, cite: false, coverAll: false } as const;

const READING_GROUP = {
  title: "Reading group",
  blocks: [
    "## Notes",
    "Wei asked for something short we can hand out before Thursday's reading group.",
    "Still need: what each paper actually claims, and whether the GraphST results hold up.",
  ],
};
const Q3 = {
  title: "Q3 review",
  blocks: [
    "Sales grew in every region this quarter, led by the north.",
    "Margins were thin in the north because shipping got more expensive, and we haven't fixed that yet.",
    "Hiring was slow, and the new office opened late in the year.",
  ],
};
const WEEKLY = {
  title: "周报",
  blocks: [
    "这周我们完成了新版本的测试，大部分问题都修好了。",
    "下周要跟客户开会，聊聊合同的事，顺便看看他们对新功能的想法。",
    "团队里有两个人请假，进度可能会慢一点。",
  ],
};
const TYPOS = {
  title: "Trip notes",
  blocks: [
    "We visted the harbour on tuesday and saw the tide go out compleatly.",
    "The guide said spring tides come twice a month, wich surprised everyone.",
  ],
};
const PENSIONS = {
  title: "Benefits memo",
  blocks: [
    "Staff get 25 days of leave a year, plus bank holidays.",
    "The pension scheme matches contributions up to 5% of salary.",
    "Health cover starts after three months.",
  ],
};
const OWNER = ["write a summary page", "make the summary section longer", "no citations?"];

export const CASES: readonly EverydayCase[] = [
  // The owner's own conversation, 2026-10-15.
  {
    id: "owner-1",
    language: "en",
    library: "papers",
    mind: READING_GROUP,
    request: OWNER[0] ?? "",
    expected: WRITE_ALL,
  },
  {
    id: "owner-2",
    language: "en",
    library: "papers",
    mind: READING_GROUP,
    before: OWNER.slice(0, 1),
    request: OWNER[1] ?? "",
    expected: WRITE,
  },
  {
    id: "owner-3",
    language: "en",
    library: "papers",
    mind: READING_GROUP,
    before: OWNER.slice(0, 2),
    request: OWNER[2] ?? "",
    expected: NOTHING,
  },
  {
    id: "owner-4",
    language: "en",
    library: "papers",
    mind: READING_GROUP,
    before: OWNER,
    request: "I mean write a summary about these 2 papers",
    expected: WRITE_ALL,
    hard: true,
  },
  // Only the paragraph asked about changes (#880's hand check changed the first).
  {
    id: "only-second-zh",
    language: "zh-CN",
    library: "papers",
    mind: WEEKLY,
    selectAll: true,
    request: "把第二段改得更正式，其他段落保持不变",
    expected: { ...EDIT, only: 1 },
    hard: true,
  },
  {
    id: "only-second-en",
    language: "en",
    library: "papers",
    mind: Q3,
    selectAll: true,
    request: "Make the second paragraph more formal and leave the other paragraphs as they are",
    expected: { ...EDIT, only: 1 },
    hard: true,
  },
  // Questions about the Documents: answered, cited, nothing changed.
  {
    id: "ask-tutorial",
    language: "en",
    library: "papers",
    mind: READING_GROUP,
    request: "What does the tutorial claim?",
    expected: ANSWER,
  },
  {
    id: "ask-tutorial-zh",
    language: "zh-CN",
    library: "papers",
    mind: READING_GROUP,
    request: "这篇关于随机神经网络的教程主要讲了什么？",
    expected: ANSWER,
  },
  {
    id: "ask-compare",
    language: "en",
    library: "papers",
    mind: READING_GROUP,
    request: "Compare the two papers' methods",
    expected: ANSWER_ALL,
  },
  {
    id: "ask-summarise",
    language: "en",
    library: "papers",
    mind: READING_GROUP,
    request: "Summarise these papers",
    expected: ANSWER_ALL,
  },
  {
    id: "ask-summarise-zh",
    language: "zh-CN",
    library: "papers",
    mind: READING_GROUP,
    request: "总结一下这两篇论文",
    expected: ANSWER_ALL,
  },
  {
    id: "ask-translate-doc",
    language: "en",
    library: "papers",
    mind: READING_GROUP,
    request: "Translate the abstract of the GraphST paper into Chinese",
    expected: ANSWER,
  },
  {
    id: "ask-clarify",
    language: "en",
    library: "papers",
    mind: READING_GROUP,
    request: "Clarify what the authors of the tutorial mean by product form",
    expected: ANSWER,
  },
  {
    id: "ask-datasets",
    language: "en",
    library: "papers",
    mind: READING_GROUP,
    request: "Which datasets does GraphST use?",
    expected: ANSWER,
  },
  {
    id: "ask-datasets-zh",
    language: "zh-CN",
    library: "papers",
    mind: READING_GROUP,
    request: "GraphST 在哪些数据集上做了实验？",
    expected: ANSWER,
  },
  // Text asked for in a Mind.
  {
    id: "write-section",
    language: "en",
    library: "papers",
    mind: READING_GROUP,
    request: "Add a short section on GraphST's results to my notes",
    expected: WRITE,
  },
  {
    id: "write-section-zh",
    language: "zh-CN",
    library: "papers",
    mind: READING_GROUP,
    request: "把教程的主要观点加到我的笔记里",
    expected: WRITE,
  },
  {
    id: "write-new-note",
    language: "en",
    library: "papers",
    mind: READING_GROUP,
    request: "Make a new note with the key numbers from the GraphST paper",
    expected: WRITE,
  },
  {
    id: "write-page-zh",
    language: "zh-CN",
    library: "papers",
    mind: READING_GROUP,
    request: "写一个总结页面",
    expected: WRITE_ALL,
  },
  {
    id: "write-page-all",
    language: "en",
    library: "papers",
    mind: READING_GROUP,
    request: "Write a summary page of these documents",
    expected: WRITE_ALL,
  },
  // The User's own text, changed.
  {
    id: "edit-shorten",
    language: "en",
    library: "papers",
    mind: Q3,
    request: "Shorten the second paragraph",
    expected: { ...EDIT, only: 1 },
  },
  {
    id: "edit-shorten-zh",
    language: "zh-CN",
    library: "papers",
    mind: WEEKLY,
    request: "把第三段缩短",
    expected: { ...EDIT, only: 2 },
  },
  {
    id: "edit-typos",
    language: "en",
    library: "papers",
    mind: TYPOS,
    request: "Fix the typos in my notes",
    expected: EDIT,
  },
  {
    id: "edit-translate",
    language: "en",
    library: "papers",
    mind: TYPOS,
    request: "Translate my notes into Chinese",
    expected: EDIT,
  },
  {
    id: "edit-formal",
    language: "en",
    library: "papers",
    mind: Q3,
    request: "Make the first paragraph more formal",
    expected: { ...EDIT, only: 0 },
  },
  {
    id: "edit-remove",
    language: "en",
    library: "papers",
    mind: PENSIONS,
    request: "Remove the part about pensions",
    expected: { ...EDIT, only: 1 },
  },
  // After a write, its follow-ups change the same text.
  {
    id: "follow-longer",
    language: "en",
    library: "papers",
    mind: READING_GROUP,
    before: ["Add a short section on GraphST's results to my notes"],
    request: "make it longer",
    expected: WRITE,
  },
  {
    id: "follow-longer-zh",
    language: "zh-CN",
    library: "papers",
    mind: READING_GROUP,
    before: ["把教程的主要观点加到我的笔记里"],
    request: "长一点",
    expected: WRITE,
  },
  // After Add to note, the summary in the Mind is what changes (#1061).
  {
    id: "added-longer",
    language: "en",
    library: "papers",
    mind: READING_GROUP,
    before: ["Summarise the tutorial paper in a short paragraph"],
    addToNote: true,
    request: "Make the summary longer.",
    expected: EDIT,
  },
  {
    id: "added-longer-zh",
    language: "zh-CN",
    library: "papers",
    mind: READING_GROUP,
    before: ["用一小段话总结一下这篇教程论文"],
    addToNote: true,
    request: "把摘要写长一点。",
    expected: EDIT,
  },
  {
    id: "follow-shorter",
    language: "en",
    library: "papers",
    mind: READING_GROUP,
    before: ["Add a short section on GraphST's results to my notes"],
    request: "no, keep it shorter",
    expected: WRITE,
  },
  // Several Documents: each covered, or the main few.
  {
    id: "reports-page",
    language: "en",
    library: "reports",
    mind: { title: "Briefing", blocks: ["Notes for Monday's briefing."] },
    request: "Write a summary page of these documents",
    expected: WRITE_ALL,
    hard: true,
  },
  {
    id: "reports-main",
    language: "en",
    library: "reports",
    mind: { title: "Briefing", blocks: ["Notes for Monday's briefing."] },
    request: "Summarise the two main documents",
    expected: ANSWER,
  },
  {
    id: "reports-all-zh",
    language: "zh-CN",
    library: "reports",
    mind: { title: "简报", blocks: ["周一简报的笔记。"] },
    request: "总结这些文档",
    expected: ANSWER_ALL,
  },
  {
    id: "reports-esg",
    language: "en",
    library: "reports",
    mind: { title: "Briefing", blocks: ["Notes for Monday's briefing."] },
    request: "What does the JP Morgan report say about emissions?",
    expected: ANSWER,
  },
  {
    id: "reports-un-zh",
    language: "zh-CN",
    library: "reports",
    mind: { title: "简报", blocks: ["周一简报的笔记。"] },
    request: "联合国 2022 年度报告的重点是什么？",
    expected: ANSWER,
  },
  {
    id: "reports-write",
    language: "en",
    library: "reports",
    mind: { title: "Briefing", blocks: ["Notes for Monday's briefing."] },
    request: "Add the ESG report's main targets to my notes",
    expected: WRITE,
  },
  // No Mind in focus.
  {
    id: "no-mind-add",
    language: "en",
    library: "papers",
    mind: null,
    request: "Add a risks section",
    expected: NOTHING,
  },
  {
    id: "no-mind-page",
    language: "en",
    library: "papers",
    mind: null,
    request: "Write a summary page of these papers",
    expected: WRITE_ALL,
  },
  // Questions that only sound like requests, and Questions about the User's own text.
  {
    id: "sounds-like",
    language: "en",
    library: "papers",
    mind: Q3,
    request: "what would a shorter version of my notes look like?",
    expected: NOTHING,
  },
  {
    id: "should-i",
    language: "en",
    library: "papers",
    mind: Q3,
    request: "Should I add a risks section?",
    expected: NOTHING,
  },
  {
    id: "polite-zh",
    language: "zh-CN",
    library: "papers",
    mind: WEEKLY,
    request: "可以把第一段改短吗？",
    expected: { ...EDIT, only: 0 },
    hard: true,
  },
  {
    id: "own-question",
    language: "en",
    library: "papers",
    mind: Q3,
    request: "How was the quarter?",
    expected: NOTHING,
  },
  {
    id: "own-explain",
    language: "en",
    library: "papers",
    mind: Q3,
    request: "Explain the second paragraph of my notes",
    expected: NOTHING,
  },
];
