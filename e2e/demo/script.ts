/**
 * What the README recording shows and says, in each language: the workspace's names, the draft,
 * the Questions, the captions and the call. Everything the app answers is the app's own: nothing
 * here is an answer.
 */
import type { DemoLanguage } from "./documents";

export interface Block {
  type: "heading1" | "heading2" | "paragraph";
  text: string;
}

export interface DemoScript {
  language: DemoLanguage;
  /** The README this recording is for. */
  readme: string;
  /** The shipped examples' Documents, by file name, in this language's folder of them. */
  credit: { source: "credit/en" | "credit/zh-CN" };
  folders: { credit: string; contracts: string };
  /** The Folder the exported file is saved in, which the Saved line names. */
  outputFolder: string;
  /** The Mind the term sheets are compared in, in the Contracts Folder. */
  options: { title: string; intro: string };
  paper: { title: string; blocks: Block[]; covenantHeading: string };
  /** Asked before the recording, to give the draft a cited paragraph to export. */
  seedQuestion: string;
  /** The Question of scene 1, typed at its first frame. */
  question: string;
  compareQuestions: string[];
  /** The line the User types in the notes during the call. */
  callNote: string;
  /** What each side says: the other side asks, the User answers in one line. */
  call: { them: string; me: string; voices: { them: string; me: string } };
  captions: {
    ask: string;
    click: string;
    check: string;
    compare: string;
    call: string;
    export: string;
  };
  endCard: { lines: string[]; audience: string };
  /** What the interface calls things the recording points at. */
  words: {
    contradicted: RegExp;
    saved: string;
    /** A footnote of the exported file: the last source it cites. */
    footnote: RegExp;
    /** What the transcript of the call must have heard: the two figures. */
    heard: RegExp;
    me: string;
    filled: string;
    answerTheirQuestion: string;
    checkCommand: string;
    compareCommand: string;
    recordCommand: string;
    /** What is typed in ⌘K to find each command: the start of its name. */
    typed: { check: string; compare: string; record: string };
  };
}

export const SCRIPTS: Record<DemoLanguage, DemoScript> = {
  en: {
    language: "en",
    readme: "README.md",
    credit: { source: "credit/en" },
    folders: { credit: "Northwind credit review", contracts: "Contracts" },
    outputFolder: "Committee papers",
    options: { title: "Refinancing options", intro: "Three banks sent term sheets." },
    paper: {
      title: "Credit committee paper",
      covenantHeading: "Covenant",
      blocks: [
        { type: "heading1", text: "Credit committee paper" },
        { type: "paragraph", text: "Draft for the credit committee." },
        { type: "heading2", text: "Recommendation" },
        {
          type: "paragraph",
          text: "Hold the facility at its current limit and ask Northwind for a monthly margin report.",
        },
        { type: "heading2", text: "Why" },
        {
          type: "paragraph",
          text: "Group revenue was £1,284 million in the third quarter, up 4.2% on a year earlier. Gross margin fell to 36.8% in Q3, mostly in online. Net debt was £1,190 million on 26 September 2026.",
        },
        { type: "heading2", text: "Covenant" },
      ],
    },
    seedQuestion:
      "In two sentences: what leverage limit does the facilities agreement set, and what was Northwind’s leverage at the end of Q3?",
    question: "Is Northwind inside its leverage covenant, and by how much?",
    compareQuestions: ["Margin", "Tenor", "Leverage covenant"],
    callNote: "Covenant headroom",
    call: {
      them: "Before we go on, how much headroom is there under the leverage covenant at the moment?",
      me: "Net debt to EBITDA is three point one times, against a limit of three point five.",
      voices: { them: "Daniel", me: "Samantha" },
    },
    captions: {
      ask: "Ask your documents.",
      click: "Click it. The quote lights up on the page.",
      check: "Wrote it yourself? It checks that too.",
      compare: "Three term sheets side by side. Every cell has its source.",
      call: "Record a call. Your notes cite what was said.",
      export: "Export to Word, citations included.",
    },
    endCard: {
      lines: [
        "Don’t trust the AI. Click the source.",
        "Every answer shows the page it came from.",
        "Free, open source, runs on your computer.",
        "github.com/IncarnaMind/IncarnaMind",
      ],
      audience:
        "For analysts, lawyers and engineers who write from documents and stand behind every figure.",
    },
    words: {
      contradicted: /Contradicted/,
      saved: "Saved",
      footnote: /Q3 2026 results, p\./,
      heard: /(3\.1|three point one)[\s\S]*(3\.5|three point five)/i,
      me: "Me",
      filled: "Notes filled in",
      answerTheirQuestion: "Answer their question",
      checkCommand: "Check this Mind",
      compareCommand: "Compare Documents…",
      recordCommand: "Record a meeting",
      typed: { check: "Check this", compare: "Compare", record: "Record a" },
    },
  },
  "zh-CN": {
    language: "zh-CN",
    readme: "README.zh-CN.md",
    credit: { source: "credit/zh-CN" },
    folders: { credit: "Northwind 授信审阅", contracts: "合同" },
    outputFolder: "授信委员会文件",
    options: { title: "再融资方案", intro: "三家银行发来了条款清单。" },
    paper: {
      title: "授信委员会报告",
      covenantHeading: "杠杆率约定",
      blocks: [
        { type: "heading1", text: "授信委员会报告" },
        { type: "paragraph", text: "供授信委员会审阅的草稿。" },
        { type: "heading2", text: "建议" },
        { type: "paragraph", text: "维持现有授信额度，并要求 Northwind 每月提供毛利率报告。" },
        { type: "heading2", text: "理由" },
        {
          type: "paragraph",
          text: "第三季度集团收入为 12.84 亿英镑，同比增长 4.2%。第三季度毛利率降至 36.8%，主要拖累来自线上业务。2026 年 9 月 26 日的净债务为 11.90 亿英镑。",
        },
        { type: "heading2", text: "杠杆率约定" },
      ],
    },
    seedQuestion:
      "用两句话回答：贷款协议规定的杠杆率上限是多少，第三季度末 Northwind 的杠杆率是多少？",
    question: "Northwind 有没有超出杠杆率条款的限制，还剩多少空间？",
    compareQuestions: ["利差", "期限", "杠杆率约定"],
    callNote: "杠杆率还有多少余地",
    call: {
      them: "我们先确认一个问题：现在的杠杆率，离贷款协议的上限还有多少空间？",
      me: "现在的杠杆率是三点一倍，上限是三点五倍。",
      voices: { them: "Eddy (Chinese (China mainland))", me: "Tingting" },
    },
    captions: {
      ask: "向你的文档提问。",
      click: "点一下，引文在原页上亮起来。",
      check: "自己写的稿子也能核对。",
      compare: "三份条款清单并排，每个格子都有出处。",
      call: "录一通电话，笔记引用通话里说的话。",
      export: "导出为 Word，引用都在。",
    },
    endCard: {
      lines: [
        "别信 AI，点开出处看。",
        "每个回答都标出它来自哪一页。",
        "免费开源，在你自己的电脑上运行。",
        "github.com/IncarnaMind/IncarnaMind",
      ],
      audience: "适合靠文件写报告、要对每个数字负责的分析师、律师和工程师。",
    },
    words: {
      contradicted: /与文档矛盾/,
      saved: "已保存",
      footnote: /2026 年第三季度业绩/,
      heard: /(3\.1|三点一)[\s\S]*(3\.5|三点五)/,
      me: "我",
      filled: "已补全笔记",
      answerTheirQuestion: "回答对方的问题",
      checkCommand: "核对此 Mind",
      compareCommand: "对比文档…",
      recordCommand: "录制会议",
      typed: { check: "核对此", compare: "对比文档", record: "录制会议" },
    },
  },
};
