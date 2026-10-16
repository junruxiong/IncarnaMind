import { describe, expect, test } from "vitest";
import { routeRequest } from "../../src/core/answers/routeRequest";

/**
 * Everyday requests in the assistant, in English and Chinese, and how each
 * is read. A request read as an edit applies at once, so no Question about a
 * Document may read as one; what can't be told reads as "unclear".
 */

const ROUTES = {
  answer: [
    "What does the tutorial claim?",
    "no citations?",
    "Summarise these papers",
    "Summarise the 10-K",
    "Clarify what the authors mean by product form",
    "Elaborate on the second point",
    "Expand on point 2",
    "Translate the abstract of the paper into Chinese",
    "Put simply, what does it mean?",
    "Explain the method in plain words",
    "what would a shorter version look like?",
    "How do I add a citation?",
    "Compare the two papers",
    "Which paper has more datasets?",
    "The authors make a strong claim about leverage.",
    "Is the GraphST result reliable?",
    "Explain what the paper says about stability",
    "Tell me about the second paper",
    "List the datasets used in the study",
    "Clarify the abstract of the paper",
    "Elaborate on what the study found",
    "what does this mean?",
    "Give me the key numbers",
    "翻译一下这篇论文的摘要",
    "这篇论文讲了什么？",
    "为什么要缩短摘要？",
    "作者说的“乘积形式”是什么意思？",
    "解释一下这篇论文的方法",
    "两篇论文有什么区别？",
    "总结一下这两篇论文",
    "Rewrite the paper's conclusion in plain English",
    "Make the paper's argument simpler",
    "what's in my notes?",
    "Can you summarise the documents?",
    "Delete the paper",
    "Update the numbers from the report",
  ],
  edit: [
    "make the summary section longer",
    "Shorten the summary",
    "make this paragraph more formal",
    "make my summary shorter",
    "make it 200 words",
    "make it longer",
    "longer please",
    "more formal",
    "remove the part about neap tides",
    "Could you translate it into Chinese?",
    "please fix the typos",
    "Rewrite the intro to be clearer",
    "Tighten the conclusion",
    "Delete the last paragraph",
    "把它写长一点",
    "把第二段写得更正式",
    "把第二段改得更正式，其他段落保持不变",
    "把这段缩短一点",
    "能不能帮我把总结改长一些？",
    "删掉关于小潮的部分",
    "长一点",
    "更正式一些",
    "Fix it up a bit",
    "Translate the summary into Chinese",
    "Remove it",
    "translate this",
  ],
  write: [
    "write a summary page",
    "I mean write a summary about these 2 papers",
    "Summarise these papers into a new note",
    "Add a risks section",
    "Can you add a risks section to Project plan?",
    "Make a new note called Fieldwork plan",
    "make notes on these papers",
    "Draft an email to the board",
    "Add a note about the Thursday meeting",
    "What do the papers claim? Add it to my notes",
    "Summarise the paper and add it to my notes",
    "Write a summary page of these documents",
    "Put the key numbers in a table in my notes",
    "写一份这两篇论文的总结",
    "写一个总结页面",
    "新建一篇叫“田野计划”的笔记",
    "帮我在笔记里添加一个风险部分",
    "这篇论文说了什么？把这个加到我的笔记里",
    "把要点记到我的笔记里",
    "Summarise these documents on a new page",
    "Add the abstract of the paper to my notes",
  ],
  unclear: [
    "Should I add a risks section?",
    "Could I get a shorter version",
    "Translate into Chinese",
    "Clarify this paragraph",
    "Simplify",
    "Explain this paragraph",
  ],
} as const;

describe("Routing everyday requests", () => {
  // protects=each everyday request reads as it should: a Question about a Document is answered, a change to the User's own text is an edit (never a second copy), text asked for in a Mind is written, and what can't be told is unclear; fails_when=a Question reads as an edit or a write, which would change the Mind unasked; no_claim=every wording a User may use.
  for (const [kind, requests] of Object.entries(ROUTES)) {
    test.each(requests)(`%s reads as ${kind}`, (request) => {
      expect(routeRequest(request).kind).toBe(kind);
    });
  }

  // protects=a request for a new page or note says so, and a request to write elsewhere doesn't; fails_when=a new page is written into the Mind in focus, or the reverse; no_claim=none beyond protects.
  test.each([
    ["write a summary page", true],
    ["写一个总结页面", true],
    ["Add a note about the Thursday meeting", true],
    ["Summarise these papers into a new note", true],
    ["make notes on these papers", true],
    ["新建一篇叫“田野计划”的笔记", true],
    ["I mean write a summary about these 2 papers", false],
    ["Add a risks section", false],
    ["帮我在笔记里添加一个风险部分", false],
    ["写一份这两篇论文的总结", false],
  ])("%s asks for a new Mind: %s", (request, newMind) => {
    expect(routeRequest(request)).toEqual({ kind: "write", newMind });
  });
});
