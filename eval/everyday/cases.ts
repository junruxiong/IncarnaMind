/**
 * The everyday requests (README.md): what Users ask the assistant every day,
 * in English and Chinese, beside a Mind or not, about their Documents or
 * their own text. The owner's own conversation comes first. Each says what
 * its Answer must do; `checks.ts` holds every Answer to it.
 */
import type { Expected } from "./checks";

/**
 * The Documents a request is asked over: two arXiv papers, the repository's seven sample PDFs,
 * or contracts and minutes whose files' dates are none of their text's (./dated).
 */
export type LibraryName = "papers" | "reports" | "dated";

export interface EverydayCase {
  id: string;
  language: "en" | "zh-CN";
  library: LibraryName;
  /** The Mind it is asked from, its Blocks in Markdown-ish lines ("## " a heading); null: none in focus. */
  mind: { title: string; blocks: readonly string[] } | null;
  /** The whole Mind is selected as it is asked. */
  selectAll?: boolean;
  /** A Word file of these paragraphs (the first its title) is imported as the Mind it is asked from, in place of `mind`. */
  word?: { name: string; paragraphs: readonly string[] };
  /** The example's Answer, written in advance, is the conversation's first turn, in the case's language. */
  afterExample?: true;
  /** With `afterExample`: asked from the example's own Mind, where the User stays, instead of from a Mind of its own. */
  inExample?: true;
  /** Asked first, in the same conversation, and not checked. */
  before?: readonly string[];
  /** The last of `before` is put into the Mind with Add to note, as a whole Answer, before the request. */
  addToNote?: true;
  request: string;
  expected: Expected;
  /** Hard enough to ask Claude Sonnet 5.5 as well (INCARNAMIND_EVAL_EVERYDAY=hard). */
  hard?: boolean;
  /** An inference is tempting: the app's own check judges each of its cited sentences (./support). */
  research?: true;
  /** A summary: the report counts its sentences outside headings that carry a Citation (./dates). */
  summary?: true;
  /** What the model's first reply is, in place of its own, so the app must ask again (./reasks). */
  firstReply?: string;
}

/** The share of its runs a request must pass: two in three, one in three for a hard one. */
export const floorOf = (each: EverydayCase): number => (each.hard ? 1 / 3 : 2 / 3);

const ANSWER = { write: false, cite: true, coverAll: false } as const;
const ANSWER_ALL = { write: false, cite: true, coverAll: true } as const;
const WRITE = { write: true, cite: true, coverAll: false } as const;
const WRITE_ALL = { write: true, cite: true, coverAll: true } as const;
const EDIT = { write: true, cite: false, coverAll: false } as const;
const NOTHING = { write: false, cite: false, coverAll: false } as const;
const ABOUT_THE_QUESTION = { write: false, cite: false, coverAll: false, noEarlier: true } as const;
const TEXT_DATES = { write: false, cite: true, coverAll: false, textDates: true } as const;

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
const READING_GROUP_ZH = {
  title: "读书会",
  blocks: [
    "## 笔记",
    "魏想要一份周四读书会前可以发给大家的简短材料。",
    "还需要：每篇论文到底主张什么。",
  ],
};
const SUPPLY_AGREEMENT = {
  name: "Supply agreement.docx",
  paragraphs: [
    "Supply agreement",
    "Each party's total liability is capped at the fees paid in the twelve months before the claim.",
    "The Buyer pays each invoice within 30 days of receiving it.",
  ],
};
const SUPPLY_AGREEMENT_ZH = {
  name: "供货协议.docx",
  paragraphs: [
    "供货协议",
    "每一方的赔偿责任以索赔前十二个月内已付的费用为限。",
    "买方应在收到每张发票后 30 天内付款。",
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
  // The app asks the model again inside the turn, its first reply being one it would ask about: nothing the User reads is about that try (#1226).
  {
    id: "reask-write",
    language: "en",
    library: "papers",
    mind: READING_GROUP,
    request: "Write a summary page of these documents",
    firstReply:
      "Here is a summary of the two documents.\n\n## Summary\n\nThe first document is the paper 2306.10683v1. The second is “A Tutorial about Random Neural Networks in Supervised Learning”.",
    expected: WRITE_ALL,
  },
  {
    id: "reask-write-zh",
    language: "zh-CN",
    library: "papers",
    mind: READING_GROUP_ZH,
    request: "写一个总结页面",
    firstReply:
      "下面是这两份文档的总结。\n\n## 总结\n\n第一份文档是论文 2306.10683v1，第二份是《A Tutorial about Random Neural Networks in Supervised Learning》。",
    expected: WRITE_ALL,
  },
  {
    id: "reask-claim",
    language: "en",
    library: "papers",
    mind: READING_GROUP,
    request: "Add a short section on GraphST's results to my notes",
    firstReply: "I've added a short section on GraphST's results to “Reading group”.",
    expected: WRITE,
  },
  {
    id: "reask-claim-zh",
    language: "zh-CN",
    library: "papers",
    mind: READING_GROUP_ZH,
    request: "把教程的主要观点加到我的笔记里",
    firstReply: "我已经在“读书会”里添加了教程的主要观点。",
    expected: WRITE,
  },
  // In a new, empty Mind with no name yet, a piece asked for goes into it (#1212).
  {
    id: "write-empty",
    language: "en",
    library: "papers",
    mind: { title: "", blocks: [] },
    request: "Write a short summary of the GraphST paper.",
    expected: WRITE,
  },
  {
    id: "write-empty-zh",
    language: "zh-CN",
    library: "papers",
    mind: { title: "", blocks: [] },
    request: "帮我写一份 GraphST 论文的简短摘要。",
    expected: WRITE,
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
    expected: { ...EDIT, inPlace: true },
  },
  {
    id: "added-longer-zh",
    language: "zh-CN",
    library: "papers",
    mind: READING_GROUP,
    before: ["用一小段话总结一下这篇教程论文"],
    addToNote: true,
    request: "把摘要写长一点。",
    expected: { ...EDIT, inPlace: true },
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
  {
    id: "example-own",
    language: "en",
    library: "papers",
    mind: READING_GROUP,
    afterExample: true,
    request: "What does the GraphST paper claim?",
    expected: ABOUT_THE_QUESTION,
  },
  {
    id: "example-own-zh",
    language: "zh-CN",
    library: "papers",
    mind: READING_GROUP_ZH,
    afterExample: true,
    request: "GraphST 这篇论文的主要主张是什么？",
    expected: ABOUT_THE_QUESTION,
  },
  {
    id: "example-main",
    language: "en",
    library: "papers",
    mind: READING_GROUP,
    afterExample: true,
    before: ["What does the GraphST paper claim?"],
    request: "Summarise the two main documents.",
    expected: ABOUT_THE_QUESTION,
  },
  {
    id: "example-main-zh",
    language: "zh-CN",
    library: "papers",
    mind: READING_GROUP_ZH,
    afterExample: true,
    before: ["GraphST 这篇论文的主要主张是什么？"],
    request: "总结一下两份主要文档。",
    expected: ABOUT_THE_QUESTION,
  },
  {
    id: "example-word",
    language: "en",
    library: "papers",
    mind: null,
    word: SUPPLY_AGREEMENT,
    afterExample: true,
    request: "What is the liability cap?",
    expected: ABOUT_THE_QUESTION,
  },
  {
    id: "example-word-zh",
    language: "zh-CN",
    library: "papers",
    mind: null,
    word: SUPPLY_AGREEMENT_ZH,
    afterExample: true,
    request: "赔偿责任的上限是多少？",
    expected: ABOUT_THE_QUESTION,
  },
  {
    id: "example-stay",
    language: "en",
    library: "papers",
    mind: null,
    afterExample: true,
    inExample: true,
    request: "What does the GraphST paper claim?",
    expected: ABOUT_THE_QUESTION,
  },
  {
    id: "example-stay-zh",
    language: "zh-CN",
    library: "papers",
    mind: null,
    afterExample: true,
    inExample: true,
    request: "GraphST 这篇论文的主要主张是什么？",
    expected: ABOUT_THE_QUESTION,
  },
  // Research questions where an inference is tempting: a cited sentence says only what its quote supports (#1073).
  {
    id: "research-attention-long",
    language: "en",
    library: "reports",
    mind: null,
    request: "Does self-attention handle very long sequences better than recurrent layers?",
    expected: ANSWER,
    research: true,
  },
  {
    id: "research-attention-heads",
    language: "en",
    library: "reports",
    mind: null,
    request: "Why did the Transformer authors use eight attention heads?",
    expected: ANSWER,
    research: true,
  },
  {
    id: "research-gpt2-size",
    language: "en",
    library: "reports",
    mind: null,
    request: "Does the GPT-2 paper show that bigger language models are always better?",
    expected: ANSWER,
    research: true,
  },
  {
    id: "research-gpt3-few-shot",
    language: "en",
    library: "reports",
    mind: null,
    request: "Is GPT-3's few-shot learning as good as fine-tuning?",
    expected: ANSWER,
    research: true,
  },
  {
    id: "research-gpt3-energy",
    language: "en",
    library: "reports",
    mind: null,
    request: "How much energy does it take to train GPT-3, and is that a problem?",
    expected: ANSWER,
    research: true,
  },
  {
    id: "research-gpt3-contamination",
    language: "en",
    library: "reports",
    mind: null,
    request: "Can we trust GPT-3's benchmark results, given the data contamination?",
    expected: ANSWER,
    research: true,
  },
  {
    id: "research-optimizer",
    language: "en",
    library: "reports",
    mind: null,
    request:
      "Does Gradient Descent: The Ultimate Optimizer remove the need to tune hyperparameters?",
    expected: ANSWER,
    research: true,
  },
  {
    id: "research-jpm-target",
    language: "en",
    library: "reports",
    mind: null,
    request: "Is JPMorgan on track to meet its $2.5 trillion sustainable development target?",
    expected: ANSWER,
    research: true,
  },
  {
    id: "research-un-hunger",
    language: "en",
    library: "reports",
    mind: null,
    request:
      "What does the UN annual report suggest about the war in Ukraine's effect on global hunger?",
    expected: ANSWER,
    research: true,
  },
  {
    id: "research-abpi-gifts",
    language: "en",
    library: "reports",
    mind: null,
    request: "Does the ABPI Code mean pharma companies can never give doctors anything?",
    expected: ANSWER,
    research: true,
  },
  {
    id: "research-attention-long-zh",
    language: "zh-CN",
    library: "reports",
    mind: null,
    request: "Transformer 的自注意力在处理很长的序列时，是不是比循环网络更有优势？",
    expected: ANSWER,
    research: true,
  },
  {
    id: "research-attention-heads-zh",
    language: "zh-CN",
    library: "reports",
    mind: null,
    request: "Transformer 论文为什么选用 8 个注意力头？",
    expected: ANSWER,
    research: true,
  },
  {
    id: "research-gpt2-size-zh",
    language: "zh-CN",
    library: "reports",
    mind: null,
    request: "GPT-2 论文是否说明模型越大效果就一定越好？",
    expected: ANSWER,
    research: true,
  },
  {
    id: "research-gpt3-few-shot-zh",
    language: "zh-CN",
    library: "reports",
    mind: null,
    request: "GPT-3 的小样本学习能达到微调的水平吗？",
    expected: ANSWER,
    research: true,
  },
  {
    id: "research-gpt3-energy-zh",
    language: "zh-CN",
    library: "reports",
    mind: null,
    request: "训练 GPT-3 要消耗多少能源？这会是个问题吗？",
    expected: ANSWER,
    research: true,
  },
  {
    id: "research-gpt3-contamination-zh",
    language: "zh-CN",
    library: "reports",
    mind: null,
    request: "考虑到数据污染，GPT-3 的基准测试结果可信吗？",
    expected: ANSWER,
    research: true,
  },
  {
    id: "research-optimizer-zh",
    language: "zh-CN",
    library: "reports",
    mind: null,
    request: "《Gradient Descent: The Ultimate Optimizer》是不是说明以后不用再调超参数了？",
    expected: ANSWER,
    research: true,
  },
  {
    id: "research-jpm-target-zh",
    language: "zh-CN",
    library: "reports",
    mind: null,
    request: "摩根大通有望实现 2.5 万亿美元的可持续发展目标吗？",
    expected: ANSWER,
    research: true,
  },
  {
    id: "research-un-hunger-zh",
    language: "zh-CN",
    library: "reports",
    mind: null,
    request: "联合国 2022 年度报告认为乌克兰战争对全球饥饿有什么影响？",
    expected: ANSWER,
    research: true,
  },
  {
    id: "research-abpi-gifts-zh",
    language: "zh-CN",
    library: "reports",
    mind: null,
    request: "ABPI 守则是不是意味着药企绝对不能给医生任何东西？",
    expected: ANSWER,
    research: true,
  },
  // Dated-looking Documents: every date an Answer gives is one their text writes, never their file's (#1173).
  {
    id: "dates-contract-summary",
    language: "en",
    library: "dated",
    mind: null,
    request: "Summarise the employment contract.",
    expected: TEXT_DATES,
    summary: true,
  },
  {
    id: "dates-contract-signed",
    language: "en",
    library: "dated",
    mind: null,
    request: "When was the employment contract signed?",
    expected: TEXT_DATES,
  },
  {
    id: "dates-contract-key-dates",
    language: "en",
    library: "dated",
    mind: null,
    request: "What are the key dates in the employment contract?",
    expected: TEXT_DATES,
  },
  {
    id: "dates-nda-effective",
    language: "en",
    library: "dated",
    mind: null,
    request: "When does the mutual NDA take effect?",
    expected: TEXT_DATES,
  },
  {
    id: "dates-minutes",
    language: "en",
    library: "dated",
    mind: null,
    request: "When did the board meet, and what did it decide?",
    expected: TEXT_DATES,
  },
  {
    id: "dates-contract-summary-zh",
    language: "zh-CN",
    library: "dated",
    mind: null,
    request: "总结一下这份劳动合同。",
    expected: TEXT_DATES,
    summary: true,
  },
  {
    id: "dates-contract-signed-zh",
    language: "zh-CN",
    library: "dated",
    mind: null,
    request: "这份劳动合同是什么时候签订的？",
    expected: TEXT_DATES,
  },
  {
    id: "dates-contract-key-dates-zh",
    language: "zh-CN",
    library: "dated",
    mind: null,
    request: "劳动合同里有哪些关键日期？",
    expected: TEXT_DATES,
  },
  {
    id: "dates-nda-effective-zh",
    language: "zh-CN",
    library: "dated",
    mind: null,
    request: "保密协议什么时候生效？",
    expected: TEXT_DATES,
  },
  {
    id: "dates-minutes-zh",
    language: "zh-CN",
    library: "dated",
    mind: null,
    request: "董事会是哪天开的会？做了什么决定？",
    expected: TEXT_DATES,
  },
];
