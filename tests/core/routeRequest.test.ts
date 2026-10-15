import { describe, expect, test } from "vitest";
import { routeRequest } from "../../src/core/answers/routeRequest";

/** Whether a request in the assistant asks for an answer or for text in a Mind. */
describe("Routing a request", () => {
  // protects=requests for new text route to write, a new page or note marked as one; fails_when=one is taken for a Question, so the text lands only in the panel; no_claim=every wording a User may use.
  test.each([
    ["write a summary page", true],
    ["I mean write a summary about these 2 papers", false],
    ["Summarise these papers into a new note", true],
    ["Add a risks section", false],
    ["Can you add a risks section to Project plan?", false],
    ["Make a new note called Fieldwork plan", true],
    ["make notes on these papers", true],
    ["Draft an email to the board", false],
    ["写一份这两篇论文的总结", false],
    ["新建一篇叫“田野计划”的笔记", true],
    ["帮我在笔记里添加一个风险部分", false],
  ])("%s", (question, newMind) => {
    expect(routeRequest(question)).toEqual({ kind: "write", newMind });
  });

  // protects=changes to text already there route to edit; fails_when=one routes to write or an answer; no_claim=none beyond protects.
  test.each([
    "make the summary section longer",
    "Shorten the summary",
    "remove the part about neap tides",
    "Could you translate it into Chinese?",
    "please fix the typos",
    "把这段缩短一点",
    "能不能帮我把总结改长一些？",
    "删掉关于小潮的部分",
  ])("%s", (question) => {
    expect(routeRequest(question)).toEqual({ kind: "edit", newMind: false });
  });

  // protects=Questions, and Questions that only sound like requests, change nothing; fails_when=one routes to a change; no_claim=none beyond protects.
  test.each([
    "no citations?",
    "What does the tutorial claim?",
    "what would a shorter version look like?",
    "How do I add a citation?",
    "Summarise these papers",
    "The authors make a strong claim about leverage.",
    "这篇论文讲了什么？",
    "为什么要缩短摘要？",
  ])("%s", (question) => {
    expect(routeRequest(question)).toEqual({ kind: "answer", newMind: false });
  });

  // protects=a request it can't tell routes to unclear, so its changes wait; fails_when=it applies at once, or reads as a plain Question; no_claim=none beyond protects.
  test.each(["Should I add a risks section?", "Could I get a shorter version"])(
    "%s",
    (question) => {
      expect(routeRequest(question).kind).toBe("unclear");
    },
  );
});
