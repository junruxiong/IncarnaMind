/**
 * The preset Tags, created on first run so automatic tagging is useful
 * straight away. They are data, not interface text: each is created once, in
 * the interface language of the time, and the User may edit or delete it
 * afterwards. Adding an interface language without these fails the type-check.
 */
import type { Language } from "../language";

export interface PresetTag {
  /** Stored with the Tag (`tags.preset`) and never shown. */
  key: string;
  text: { readonly [L in Language]: { name: string; description: string } };
}

export const PRESET_TAGS: readonly PresetTag[] = [
  {
    key: "paper",
    text: {
      en: {
        name: "Paper",
        description:
          "A research or academic paper, such as a journal article, conference paper, preprint or thesis.",
      },
      "zh-CN": {
        name: "论文",
        description: "研究或学术论文，例如期刊文章、会议论文、预印本或学位论文。",
      },
    },
  },
  {
    key: "report",
    text: {
      en: {
        name: "Report",
        description:
          "A report of findings, analysis or progress, such as a business, technical, financial or government report.",
      },
      "zh-CN": {
        name: "报告",
        description: "呈现调查结果、分析或进展的报告，例如商业、技术、财务或政府报告。",
      },
    },
  },
  {
    key: "book",
    text: {
      en: {
        name: "Book",
        description: "A book or a chapter of one, including e-books, textbooks and long manuals.",
      },
      "zh-CN": {
        name: "书籍",
        description: "一本书或其中的章节，包括电子书、教科书和篇幅较长的手册。",
      },
    },
  },
  {
    key: "contract",
    text: {
      en: {
        name: "Contract",
        description:
          "A legal agreement between parties, such as a contract, terms of service, lease or NDA.",
      },
      "zh-CN": {
        name: "合同",
        description: "双方或多方之间的法律协议，例如合同、服务条款、租约或保密协议。",
      },
    },
  },
  {
    key: "invoice",
    text: {
      en: {
        name: "Invoice",
        description:
          "A bill or receipt that asks for or confirms a payment, with items, amounts and dates.",
      },
      "zh-CN": {
        name: "发票",
        description: "要求或确认付款的账单或收据，列有项目、金额和日期。",
      },
    },
  },
  {
    key: "slides",
    text: {
      en: {
        name: "Slides",
        description:
          "A presentation deck saved as a document, with a title and a few short points on each slide.",
      },
      "zh-CN": {
        name: "幻灯片",
        description: "保存为文档的演示文稿，每页有一个标题和几条简短要点。",
      },
    },
  },
  {
    key: "notes",
    text: {
      en: {
        name: "Notes",
        description:
          "Informal notes, such as meeting notes, lecture notes, reading notes or a personal draft.",
      },
      "zh-CN": {
        name: "笔记",
        description: "非正式的笔记，例如会议记录、课堂笔记、读书笔记或个人草稿。",
      },
    },
  },
];
