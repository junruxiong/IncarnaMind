import type { Node as ProseMirrorNode } from "@tiptap/pm/model";
import { describe, expect, test } from "vitest";
import { asksAboutEarlier, withoutPointersBack } from "../../src/core/answers/pointersBack";
import { pageBlocks } from "../../src/shared/answerBlocks";
import { askAssistant, lastResult } from "../helpers/assistant";
import { setUpWithDocuments, shownPassages } from "../helpers/citations";
import { createTempDataFolder, startCore } from "../helpers/core";
import { connectToMind } from "../helpers/mindClient";
import { readMind } from "../helpers/minds";
import {
  type ModelCall,
  promptOf,
  type ScriptedReply,
  scriptedModel,
  scriptedModels,
} from "../helpers/models";

/**
 * Text that goes into a page stands on its own (#1172): what points to an
 * earlier Answer is cut from what the assistant writes into a Mind and from an
 * Answer as Add to note puts it in a page, never from the Answer in the panel.
 * Only the pointer goes, a sentence with a Citation keeps its claim, and the
 * instructions ask for it too.
 */

const page = (markdown: string) => withoutPointersBack(markdown);

describe("Text without pointers to an earlier Answer", () => {
  // protects=a sentence that is nothing but a pointer to an earlier Answer, or talk of one, goes, in English and Chinese, from a paragraph or a list; fails_when=it stays in the text; no_claim=every wording a model may use.
  test.each([
    [
      "## Summary\n\nRole: software engineer [^1]. Notice, holiday and home working are covered in my earlier answer.",
      "## Summary\n\nRole: software engineer [^1].",
    ],
    [
      "- Pension: 5% of salary [^2].\n- See my earlier answer for notice and holiday.",
      "- Pension: 5% of salary [^2].",
    ],
    [
      "Salary is paid monthly [^1]. I covered holiday in my earlier answer.",
      "Salary is paid monthly [^1].",
    ],
    [
      "The pay is 18,000 yuan [^1].\n\nThe other terms are listed above.",
      "The pay is 18,000 yuan [^1].",
    ],
    ["合同期三年[^1]。居家办公的规定见上一条回答。", "合同期三年[^1]。"],
    ["月薪 18,000 元[^1]。这些内容在上一条回答中已经说明。", "月薪 18,000 元[^1]。"],
    [
      "GraphST 在三项任务中都优于基线[^4]。我这次没有重新核对前面回答中的犯罪记录条数（321,876 和 108,575）。",
      "GraphST 在三项任务中都优于基线[^4]。",
    ],
    [
      "It outperforms the baselines [^2]. This summary leaves out what my earlier answer said about the datasets.",
      "It outperforms the baselines [^2].",
    ],
  ])("%s", (text, kept) => {
    expect(page(text)).toEqual({ markdown: kept, left: 1 });
  });

  // protects=only the clause that points back is cut, wherever it stands, and its sentence keeps what it says, a correction included, in English and Chinese; fails_when=the sentence is lost or the pointer stays; no_claim=none beyond protects.
  test.each([
    [
      "As I said in my earlier answer, notice is three months [^1].",
      "Notice is three months [^1].",
    ],
    [
      "Notice is three months [^1], as set out in my earlier answer.",
      "Notice is three months [^1].",
    ],
    ["Notice is three months [^1] (see above).", "Notice is three months [^1]."],
    ["如上一条回答所述，试用期为三个月[^1]。", "试用期为三个月[^1]。"],
    ["试用期为三个月[^1]，详见上文。", "试用期为三个月[^1]。"],
    ["年假为 15 天[^2]（见上一条回答）。", "年假为 15 天[^2]。"],
    [
      "I made an error in my previous answer: the total is 45, not 54. The difference is in row 2 [^1].",
      "The total is 45, not 54. The difference is in row 2 [^1].",
    ],
    [
      "我上一条回答有误：总额是 45，不是 54。差别在第二行[^1]。",
      "总额是 45，不是 54。差别在第二行[^1]。",
    ],
    [
      "Pension is 5% [^1], as covered in my earlier answer, and holiday is 25 days [^2].",
      "Pension is 5% [^1], and holiday is 25 days [^2].",
    ],
    [
      "养老金为 5%[^1]，如上一条回答所述，年假为 25 天[^2]。",
      "养老金为 5%[^1]，年假为 25 天[^2]。",
    ],
    [
      "The pay is 18,000 [^1], and the other terms are covered in my earlier answer.",
      "The pay is 18,000 [^1].",
    ],
    ["月薪 18,000 元[^1]，居家办公的规定见上一条回答。", "月薪 18,000 元[^1]。"],
    [
      "Holiday is covered in my earlier answer, but notice is three months [^1].",
      "Notice is three months [^1].",
    ],
    ["居家办公的规定见上一条回答，年假为 15 天[^1]。", "年假为 15 天[^1]。"],
  ])("%s", (text, kept) => {
    expect(page(text)).toEqual({ markdown: kept, left: 1 });
  });

  // protects=a cut leaves the rest as written: "eBay" keeps its case, "正如前所述" and "如上述" are cut too, and a table under a cut line keeps its rows; fails_when=the rest is changed or a row becomes "|"; no_claim=none beyond protects.
  test.each([
    ["As I said earlier, eBay takes a 13% fee [^1].", "eBay takes a 13% fee [^1]."],
    ["正如前所述，年假为 15 天[^1]。", "年假为 15 天[^1]。"],
    ["如上述，试用期为三个月[^1]。", "试用期为三个月[^1]。"],
    [
      "The other terms are listed above.\n| Term | Where |\n|---|---|\n| Notice | See my earlier answer. |",
      "| Term | Where |\n|---|---|\n| Notice | See my earlier answer. |",
    ],
    [
      "其余条款见上文。\n| 条款 | 位置 |\n|---|---|\n| 通知期 | 见上一条回答。 |",
      "| 条款 | 位置 |\n|---|---|\n| 通知期 | 见上一条回答。 |",
    ],
  ])("%s", (text, kept) => {
    expect(page(text)).toEqual({ markdown: kept, left: 1 });
  });

  // protects=a heading or lead-in goes when all that was under it was a pointer, and stays when anything else is, or when nothing was, in English and Chinese; fails_when=it is left with nothing under it, or goes with content still under it; no_claim=none beyond protects.
  test.each([
    [
      "## Terms\n\nNotice is covered in my earlier answer.\n\n## Pay\n\nPay is 5k a month [^1].",
      "## Pay\n\nPay is 5k a month [^1].",
    ],
    [
      "## 条款\n\n通知期见上一条回答。\n\n## 薪酬\n\n月薪 18,000 元[^1]。",
      "## 薪酬\n\n月薪 18,000 元[^1]。",
    ],
    [
      "Pay is 5k a month [^1].\n\nKey points:\n\n- See my earlier answer for notice.\n- Holiday is covered above.",
      "Pay is 5k a month [^1].",
    ],
    [
      "月薪 18,000 元[^1]。\n\n要点：\n\n- 通知期见上一条回答。\n- 年假见上文。",
      "月薪 18,000 元[^1]。",
    ],
    [
      "## Notes\n\n## Pay\n\nPay is 5k a month [^1]. Holiday is covered in my earlier answer.",
      "## Notes\n\n## Pay\n\nPay is 5k a month [^1].",
    ],
    [
      "Key points:\n\n- Pay is 5k a month [^1].\n- See my earlier answer for notice.",
      "Key points:\n\n- Pay is 5k a month [^1].",
    ],
    [
      "## 条款\n\n### 通知期\n\n通知期见上一条回答。\n\n### 薪酬\n\n月薪 18,000 元[^1]。",
      "## 条款\n\n### 薪酬\n\n月薪 18,000 元[^1]。",
    ],
    [
      "要点：\n\n- 月薪 18,000 元[^1]。\n- 通知期见上一条回答。",
      "要点：\n\n- 月薪 18,000 元[^1]。",
    ],
  ])("%s", (text, kept) => {
    expect(page(text).markdown).toBe(kept);
  });

  // protects=a marker written after the full stop is its own sentence's: a pointer after it isn't taken for cited, and a cited pointer before it keeps it rather than handing it to the next claim, in English and Chinese; fails_when=the pointer stays, or the marker moves to the next sentence; no_claim=none beyond protects.
  test.each([
    [
      "Role: software engineer.[^1] Notice, holiday and home working are covered in my earlier answer.",
      "Role: software engineer.[^1]",
    ],
    ["合同期三年。[^1]居家办公的规定见上一条回答。", "合同期三年。[^1]"],
    [
      "Holiday is covered in my earlier answer. [^1] Notice is three months [^2].",
      "Holiday is covered in my earlier answer. [^1] Notice is three months [^2].",
    ],
    [
      "居家办公的规定见上一条回答。[^1]年假为 15 天[^2]。",
      "居家办公的规定见上一条回答。[^1]年假为 15 天[^2]。",
    ],
  ])("%s", (text, kept) => {
    expect(page(text).markdown).toBe(kept);
  });

  // protects=the Documents' words, cited claims, code, tables and text that is only a pointer stay as written, in English and Chinese; fails_when=content is cut; no_claim=none beyond protects.
  test.each([
    "An earlier study found the same effect [^1].",
    "Prices are listed above market rates.",
    "The previous answer on the form was left blank.",
    "Our first response is due in 24 hours.",
    "公司之前的回复期限是 24 小时。",
    "Notice is covered in my earlier answer [^1].",
    "合同写道“详见上文第三条”[^1]。",
    "上述条款适用于全体员工。",
    "```text\nSee my earlier answer.\n```",
    "| Term | Where |\n|---|---|\n| Notice | covered in my earlier answer. |",
  ])("%s", (text) => {
    const written = `Tides follow the moon [^9].\n\n${text}`;
    expect(page(written)).toEqual({ markdown: written, left: 0 });
  });

  // protects=text that is nothing but a pointer is given back as it was, rather than nothing; fails_when=it comes back empty; no_claim=none beyond protects.
  test("text that is only a pointer stays as it was", () => {
    const text = "## Summary\n\nThis is covered in my earlier answer.";
    expect(page(text)).toEqual({ markdown: text, left: 0 });
  });

  // protects=a Question about an earlier Answer is told apart from one that isn't, in English and Chinese; fails_when=an ordinary request reads as one about an earlier Answer, or the other way round; no_claim=none beyond protects.
  test("a Question about an earlier Answer", () => {
    expect(asksAboutEarlier("Where did your earlier answer cover the notice period?")).toBe(true);
    expect(asksAboutEarlier("What did you say about the pension?")).toBe(true);
    expect(asksAboutEarlier("You said 27 days. Where is that?")).toBe(true);
    expect(asksAboutEarlier("你刚才说的年假是多少天？")).toBe(true);
    expect(asksAboutEarlier("上一条回答里的年假是多少天？")).toBe(true);
    expect(asksAboutEarlier("Write a short summary of this agreement.")).toBe(false);
    expect(asksAboutEarlier("请为这份合同写一个简短的摘要。")).toBe(false);
    expect(asksAboutEarlier("你说明一下这份合同的通知期。")).toBe(false);
  });
});

async function setUp(script: (call: ModelCall) => ScriptedReply) {
  const model = scriptedModel(script);
  const core = startCore(await createTempDataFolder(), {
    createChatModel: scriptedModels(model).createChatModel,
  });
  await core.saveChatProvider({ kind: "ollama", modelId: "local-model" });
  const mind = await core.createMind({ title: "" });
  return { core, model, mind };
}

const blocksText = (blocks: unknown) => JSON.stringify(blocks);

const SPRING = "Spring tides happen at new moon and at full moon";

const TIDES = {
  name: "Tides.md",
  contents: `# Tides\n\n${SPRING}, when the Sun and the Moon pull together.\n`,
};

function citationChecks(doc: ProseMirrorNode): string[] {
  const checks: string[] = [];
  doc.descendants((node) => {
    if (node.type.name === "citation") checks.push(String(node.attrs.check));
    return true;
  });
  return checks;
}

const PANEL = [
  {
    language: "English",
    question: "Summarise this agreement for my manager.",
    reply:
      "The agreement employs D. Reyes as a software engineer. Notice, holiday and home working are covered in my earlier answer.",
    inPage: "The agreement employs D. Reyes as a software engineer.",
    pointer: "covered in my earlier answer",
  },
  {
    language: "Chinese",
    question: "请为这份合同写一个简短的摘要。",
    reply: "这份合同聘用李明为物流专员。居家办公的规定见上一条回答。",
    inPage: "这份合同聘用李明为物流专员。",
    pointer: "见上一条回答",
  },
  {
    language: "English, a correction",
    question: "Check the total again.",
    reply: "I made an error in my previous answer: the total is 45, not 54.",
    inPage: "The total is 45, not 54.",
    pointer: "my previous answer",
  },
  {
    language: "Chinese, a correction",
    question: "再核对一下总额。",
    reply: "我上一条回答有误：总额是 45，不是 54。",
    inPage: "总额是 45，不是 54。",
    pointer: "上一条回答",
  },
];

describe("Through the assistant", { timeout: 30_000 }, () => {
  for (const each of PANEL) {
    // protects=an Answer whose model points to an earlier Answer stays whole in the panel, and Add to note puts it in a page without the pointer, its substance kept; the instructions ask for none; fails_when=the panel's Answer is cut, or the pointer or less than the substance goes into the page; no_claim=how often a real model writes one (eval/everyday).
    test(`the panel keeps the model's words, and the page gets them without the pointer (${each.language})`, async () => {
      const { core, model, mind } = await setUp(() => ({ text: each.reply }));

      const answer = await askAssistant(core, { text: each.question, mindId: mind.id });

      expect(answer.text).toBe(each.reply);
      const added = blocksText(pageBlocks(answer, each.question));
      expect(added).toContain(each.inPage);
      expect(added).not.toContain(each.pointer);
      const system = promptOf(model)[0]?.text ?? "";
      expect(system).toContain("Never send the reader to an earlier Answer");
      expect(system).toContain("见上一条回答");
    });
  }

  // protects=asked about an earlier Answer, the page may say where it was too; fails_when=its pointer is taken out; no_claim=none beyond protects.
  test("asked about an earlier Answer, the page keeps what points to it", async () => {
    const reply = "Yes. Notice is covered in my earlier answer, under Terms.";
    const question = "Did your earlier answer cover the notice period?";
    const { core, mind } = await setUp(() => ({ text: reply }));

    const answer = await askAssistant(core, { text: question, mindId: mind.id });

    expect(answer.text).toBe(reply);
    expect(blocksText(pageBlocks(answer, question))).toContain("covered in my earlier answer");
  });

  // protects=text the assistant writes into a Mind has no pointer to an earlier Answer, and the model is told one was cut, so it can write what it pointed to; its reply stays as written; fails_when=the pointer lands in the page, the model isn't told, or the reply is cut; no_claim=none beyond protects.
  test("a pointer written into a Mind is cut, the model is told, and its reply stays", async () => {
    const reply = "Added a summary. I left out what my earlier answer covered.";
    const content =
      "## Summary\n\nThe agreement employs D. Reyes as a software engineer. Notice and holiday are covered in my earlier answer.";
    const { core, model, mind } = await setUp(({ results }) =>
      results.length === 0
        ? { calls: [{ tool: "add_to_mind", input: { content } }] }
        : { text: reply },
    );

    const answer = await askAssistant(core, { text: "Add a summary to my notes", mindId: mind.id });

    const client = await connectToMind(core, mind.id);
    await client.settled();
    const text = readMind(client).textContent;
    expect(text).toContain("The agreement employs D. Reyes as a software engineer.");
    expect(text).not.toContain("earlier answer");
    expect(lastResult(model)).toContain(
      "What pointed to an earlier Answer was cut from 1 sentence",
    );
    expect(answer.text).toBe(reply);
  });

  // protects=a sentence whose source is written as its name in brackets counts as cited once the source is linked, so it is never taken out and its Citation lands; fails_when=it is read before its source is linked and taken out; no_claim=none beyond protects.
  test("a sentence citing a source written as text keeps its claim", async () => {
    const content =
      "Tides follow the moon. Spring tides are described in my earlier answer [Tides, p. 1].";
    const model = scriptedModel((call): ScriptedReply => {
      const search = call.results.find((result) => result.tool === "search_documents");
      if (!search) {
        return { calls: [{ tool: "search_documents", input: { query: "spring tides" } }] };
      }
      if (!call.results.some((result) => result.tool === "cite")) {
        const [passage] = shownPassages(search.text);
        const record = { marker: 1, passage: passage?.id, quote: SPRING };
        return { calls: [{ tool: "cite", input: { citations: [record] } }] };
      }
      return call.results.filter((result) => result.tool === "add_to_mind").length > 1
        ? { text: "Done." }
        : { calls: [{ tool: "add_to_mind", input: { content } }] };
    });
    const { core, mind, client } = await setUpWithDocuments(model, [TIDES]);

    await askAssistant(core, { text: "Add what the tides file says", mindId: mind.id });
    await client.settled();

    const written = readMind(client);
    expect(written.textContent).toContain("Spring tides are described in my earlier answer");
    expect(citationChecks(written)).toEqual(["found"]);
  });
});
