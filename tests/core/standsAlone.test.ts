import { describe, expect, test } from "vitest";
import { asksAboutEarlier, withoutPointersBack } from "../../src/core/answers/pointersBack";
import { answerBlocks } from "../../src/shared/answerBlocks";
import { askAssistant, lastResult } from "../helpers/assistant";
import { createTempDataFolder, startCore } from "../helpers/core";
import { connectToMind } from "../helpers/mindClient";
import { readMind } from "../helpers/minds";
import { promptOf, type ScriptedReply, scriptedModel, scriptedModels } from "../helpers/models";

/**
 * Text written for a page stands on its own (#1172): what points to an earlier
 * Answer is taken out of an Answer and of what the assistant writes into a
 * Mind, and a sentence with a Citation keeps its claim. The instructions ask
 * for it too.
 */

describe("Text without pointers to an earlier Answer", () => {
  // protects=a sentence that sends the reader to an earlier Answer, or speaks of one, goes, in English and Chinese, from a paragraph or a list; fails_when=it stays in the text; no_claim=every wording a model may use.
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
    expect(withoutPointersBack(text)).toEqual({ markdown: kept, left: 1 });
  });

  // protects=a clause or bracket that points back is cut and its sentence keeps its words and Citation; fails_when=the sentence is lost or the pointer stays; no_claim=none beyond protects.
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
  ])("%s", (text, kept) => {
    expect(withoutPointersBack(text)).toEqual({ markdown: kept, left: 1 });
  });

  // protects=the Documents' words, cited claims, code, tables and text that is only a pointer stay as written; fails_when=content is cut; no_claim=none beyond protects.
  test.each([
    "An earlier study found the same effect [^1].",
    "Prices are listed above market rates.",
    "The previous answer on the form was left blank.",
    "Notice is covered in my earlier answer [^1].",
    "合同写道“详见上文第三条”[^1]。",
    "```text\nSee my earlier answer.\n```",
    "| Term | Where |\n|---|---|\n| Notice | covered in my earlier answer. |",
    "This is covered in my earlier answer.",
  ])("%s", (text) => {
    expect(withoutPointersBack(text)).toEqual({ markdown: text, left: 0 });
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

async function setUp(script: (call: { results: unknown[] }) => ScriptedReply) {
  const model = scriptedModel(script);
  const core = startCore(await createTempDataFolder(), {
    createChatModel: scriptedModels(model).createChatModel,
  });
  await core.saveChatProvider({ kind: "ollama", modelId: "local-model" });
  const mind = await core.createMind({ title: "" });
  return { core, model, mind };
}

const SUMMARIES = [
  {
    language: "English",
    question: "Summarise this agreement for my manager.",
    kept: "The agreement employs D. Reyes as a software engineer.",
    pointer: "Notice, holiday and home working are covered in my earlier answer.",
  },
  {
    language: "Chinese",
    question: "请为这份合同写一个简短的摘要。",
    kept: "这份合同聘用李明为物流专员。",
    pointer: "居家办公的规定见上一条回答。",
  },
];

describe("Through the assistant", { timeout: 30_000 }, () => {
  for (const each of SUMMARIES) {
    // protects=an Answer whose model ends with a pointer to an earlier Answer keeps none, in its text and in the Blocks Add to note puts in a page; fails_when=the pointer is stored or added; no_claim=how often a real model writes one (eval/everyday).
    test(`a summary's pointer to an earlier Answer is not kept, nor added to a page (${each.language})`, async () => {
      const { core, model, mind } = await setUp(() => ({ text: `${each.kept} ${each.pointer}` }));

      const answer = await askAssistant(core, { text: each.question, mindId: mind.id });

      expect(answer.text).toBe(each.kept);
      const system = promptOf(model)[0]?.text ?? "";
      expect(system).toContain("so it stands on its own");
      expect(system).toContain("见上一条回答");
      const added = JSON.stringify(answerBlocks(answer));
      expect(added).toContain(each.kept);
      expect(added).not.toContain(each.pointer);
    });
  }

  // protects=asked about an earlier Answer, the Answer may say where it was; fails_when=its pointer is taken out; no_claim=none beyond protects.
  test("asked about an earlier Answer, the Answer keeps what points to it", async () => {
    const reply = "Yes. Notice is covered in my earlier answer, under Terms.";
    const { core, mind } = await setUp(() => ({ text: reply }));

    const answer = await askAssistant(core, {
      text: "Did your earlier answer cover the notice period?",
      mindId: mind.id,
    });

    expect(answer.text).toBe(reply);
  });

  // protects=text the assistant writes into a Mind has no pointer to an earlier Answer, and the model is told one was cut, so it can write what it pointed to; its reply, which only reports on the change, is kept as written; fails_when=the pointer lands in the page, the model isn't told, or the reply is cut; no_claim=none beyond protects.
  test("a pointer written into a Mind is cut, the model is told, and its reply on the change stays", async () => {
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
    expect(lastResult(model)).toContain("pointing to an earlier Answer");
    expect(answer.text).toBe(reply);
  });
});
