import type { Node as ProseMirrorNode } from "@tiptap/pm/model";
import { describe, expect, test } from "vitest";
import { asksAboutEarlier, withoutPointersBack } from "../../src/core/answers/pointersBack";
import { pageBlocks } from "../../src/shared/answerBlocks";
import type { NodeJSON } from "../../src/shared/blockJson";
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
 * Text that goes into a page stands on its own (#1172): a sentence that is
 * only a pointer to an earlier Answer goes from what the assistant writes
 * into a Mind and from an Answer as Add to note puts it in a page, never from
 * the Answer in the panel. A sentence that says more stays as written and is
 * counted, and quotes, tables and code are never touched.
 */

const OWN = "Pay is 5k a month [^1].";
const OWN_ZH = "月薪 18,000 元[^1]。";

describe("Sentences that are only a pointer", () => {
  // protects=a sentence that is a pointer and nothing else goes, in English and Chinese, from a paragraph or a list; fails_when=it stays; no_claim=every wording a model may use.
  test.each([
    `${OWN} Notice, holiday and home working are covered in my earlier answer.`,
    `${OWN} See my earlier answer for notice and holiday.`,
    `${OWN} Please refer to my previous response for the full breakdown.`,
    `${OWN} For more detail, see my earlier answer.`,
    `${OWN} (See my earlier answer.)`,
    `${OWN} Details are in my earlier answer.`,
    `${OWN} I covered holiday in my earlier answer.`,
    `${OWN} As mentioned above.`,
    `${OWN} Covered above.`,
    `${OWN}\n\nThe other terms are listed above.`,
    `- ${OWN}\n- See my earlier answer for notice.`,
    `${OWN_ZH}居家办公的规定见上一条回答。`,
    `${OWN_ZH}具体数据集和消融结果可参见前一条回答。`,
    `${OWN_ZH}这些内容在上一条回答中已经说明。`,
    `${OWN_ZH}其余内容我已在之前的回答中说明。`,
    `${OWN_ZH}请参考前面的回答。`,
    `${OWN_ZH}更多信息，请参见上一条回答。`,
    `${OWN_ZH}如上所述。`,
    `- ${OWN_ZH}\n- 通知期见上一条回答。`,
  ])("%s", (text) => {
    const kept = text.startsWith("- ") ? text.split("\n")[0] : text.startsWith(OWN) ? OWN : OWN_ZH;
    expect(withoutPointersBack(text)).toEqual({
      markdown: kept,
      removed: 1,
      left: 0,
      framing: 0,
    });
  });

  // protects=a sentence that points back but says anything more stays exactly as written, and is counted, a correction included, in English and Chinese; fails_when=it is cut, changed or not counted; no_claim=none beyond protects.
  test.each([
    "I made an error in my previous answer and the correct total excludes VAT.",
    "I made an error in my previous answer: the total is 45, not 54.",
    "My earlier answer wrongly said the lease ends in March and it actually ends in May.",
    "This note supersedes my earlier answer and now covers the deposit.",
    "I didn't recheck the figures in my earlier answer.",
    "The notice period, as covered in my earlier answer, is three months.",
    "Pension is 5%, as covered in my earlier answer, and holiday is 25 days.",
    "Pay is monthly, and the other terms are covered in my earlier answer.",
    "For more detail on the 2024 figures, see my earlier answer.",
    "As I said earlier, eBay takes a 13% fee.",
    "我在上一条回答中弄错了租期实际应为三年",
    "这次摘要更正了我上一条回答里把租期写成两年的错误并补充了押金条款",
    "我这次没有重新核对前面回答中的犯罪记录条数（321,876 和 108,575）。",
    "养老金为 5%，如上一条回答所述，年假为 25 天。",
    "如上所述，试用期是三个月。",
    "试用期为三个月的规定见上文。",
  ])("%s", (sentence) => {
    const text = `${OWN}\n\n${sentence}`;
    expect(withoutPointersBack(text)).toEqual({ markdown: text, removed: 0, left: 1, framing: 0 });
  });

  // protects=quoted words, in either full-stop position, table cells, code, list items with other text, cited pointers and what a Document could say are never touched, in English and Chinese; fails_when=any of it changes; no_claim=none beyond protects.
  test.each([
    'The lease says: "Rent is due monthly. See above." [^1]',
    "The lease says “See my earlier answer.” [^1]",
    "合同写道：“详见上文。”[^1]",
    "合同写道：“如上所述”[^1]。",
    "合同写道：「见上一条回答。」[^1]",
    "The letter says 'See above.' [^1]",
    "| Term | Where |\n|---|---|\n| Notice | See my earlier answer. |",
    "```text\nSee my earlier answer.\n```",
    "- Notice: see my earlier answer.\n- Pay: 5k a month.",
    "- Pay is 5k a month. See my earlier answer.",
    "Covered in my earlier answer.[^2]",
    "Our first response is due in 24 hours.",
    "公司之前的回复期限是 24 小时。",
    "The previous answer on the form was left blank.",
    "Prices are listed above market rates.",
  ])("%s", (written) => {
    const text = `${OWN}\n\n${written}`;
    expect(withoutPointersBack(text)).toMatchObject({ markdown: text, removed: 0 });
  });

  // protects=a marker written after the full stop is its own sentence's: a pointer after it isn't taken for cited, and a cited pointer before it keeps it, in English and Chinese; fails_when=the pointer stays, or the marker moves to the next sentence; no_claim=none beyond protects.
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
    expect(withoutPointersBack(text).markdown).toBe(kept);
  });

  // protects=on the whole text, a heading or lead-in goes when all under it was a pointer, and stays when anything else is, or when nothing was, in English and Chinese; fails_when=it is left with nothing under it, or goes with content under it; no_claim=none beyond protects.
  test.each([
    [
      "## Terms\n\nSee my earlier answer.\n\n## Pay\n\nPay is 5k a month.",
      "## Pay\n\nPay is 5k a month.",
    ],
    [
      "## 条款\n\n通知期见上一条回答。\n\n## 薪酬\n\n月薪 18,000 元。",
      "## 薪酬\n\n月薪 18,000 元。",
    ],
    [
      "Pay is 5k a month.\n\nKey points:\n\n- See my earlier answer for notice.\n- Holiday is covered above.",
      "Pay is 5k a month.",
    ],
    ["月薪 18,000 元。\n\n要点：\n\n- 通知期见上一条回答。\n- 年假见上文。", "月薪 18,000 元。"],
    ["**Terms**\n\nSee my earlier answer.\n\nPay is 5k a month.", "Pay is 5k a month."],
    ["**Key points:**\n\n- See my earlier answer.\n\nPay is 5k a month.", "Pay is 5k a month."],
    [
      "## Notes\n\n## Pay\n\nPay is 5k a month. Holiday is covered in my earlier answer.",
      "## Notes\n\n## Pay\n\nPay is 5k a month.",
    ],
    [
      "Key points:\n\n- Pay is 5k a month.\n- See my earlier answer for notice.",
      "Key points:\n\n- Pay is 5k a month.",
    ],
  ])("%s", (text, kept) => {
    expect(withoutPointersBack(text).markdown).toBe(kept);
  });

  // protects=on part of a text still coming in, no heading or lead-in goes, as what goes under it hasn't all come; fails_when=one goes before the text is whole; no_claim=none beyond protects.
  test("part of a text keeps its headings", () => {
    const text = "## Terms\n\nSee my earlier answer.\n\n## Pay\n\nPay is 5k a month.";
    expect(withoutPointersBack(text, { final: false })).toEqual({
      markdown: "## Terms\n\n## Pay\n\nPay is 5k a month.",
      removed: 1,
      left: 0,
      framing: 0,
    });
  });

  // protects=text that is nothing but a pointer is given back as it was, and counted; fails_when=it comes back empty or uncounted; no_claim=none beyond protects.
  test("text that is only a pointer stays as it was", () => {
    const text = "## Summary\n\nThis is covered in my earlier answer.";
    expect(withoutPointersBack(text)).toEqual({ markdown: text, removed: 0, left: 1, framing: 0 });
  });

  // protects=a Question about an earlier Answer is told apart from one that isn't, in English and Chinese; fails_when=an ordinary request reads as one about an earlier Answer, or the other way round; no_claim=none beyond protects.
  test("a Question about an earlier Answer", () => {
    expect(asksAboutEarlier("Where did your earlier answer cover the notice period?")).toBe(true);
    expect(asksAboutEarlier("What did you say about the pension?")).toBe(true);
    expect(asksAboutEarlier("你刚才说的年假是多少天？")).toBe(true);
    expect(asksAboutEarlier("上一条回答里的年假是多少天？")).toBe(true);
    expect(asksAboutEarlier("Write a short summary of this agreement.")).toBe(false);
    expect(asksAboutEarlier("When is the first response due under the agreement?")).toBe(false);
    expect(asksAboutEarlier("请为这份合同写一个简短的摘要。")).toBe(false);
  });
});

async function setUp(script: (call: ModelCall) => ScriptedReply, chunkSize?: number) {
  const model = scriptedModel(script, { chunkSize });
  const core = startCore(await createTempDataFolder(), {
    createChatModel: scriptedModels(model).createChatModel,
  });
  await core.saveChatProvider({ kind: "ollama", modelId: "local-model" });
  const mind = await core.createMind({ title: "" });
  return { core, model, mind };
}

const textOf = (block: NodeJSON): string =>
  block.text ?? (block.content ?? []).map(textOf).join("");

const pageText = (blocks: readonly NodeJSON[]) => blocks.map(textOf).join("\n");

const shapeOf = (doc: ProseMirrorNode) => {
  const shape: string[] = [];
  doc.forEach((block) => {
    if (block.textContent.trim() !== "") shape.push(`${block.type.name}: ${block.textContent}`);
  });
  return shape;
};

const PANEL = [
  {
    language: "English",
    question: "Summarise this agreement for my manager.",
    reply:
      "The agreement employs D. Reyes as a software engineer. Notice, holiday and home working are covered in my earlier answer.",
    inPage: "The agreement employs D. Reyes as a software engineer.",
  },
  {
    language: "Chinese",
    question: "请为这份合同写一个简短的摘要。",
    reply: "这份合同聘用李明为物流专员。居家办公的规定见上一条回答。",
    inPage: "这份合同聘用李明为物流专员。",
  },
  {
    language: "English, a correction",
    question: "Check the total again.",
    reply: "I made an error in my previous answer and the correct total excludes VAT.",
    inPage: "I made an error in my previous answer and the correct total excludes VAT.",
  },
  {
    language: "Chinese, a correction",
    question: "再核对一下租期。",
    reply: "我在上一条回答中弄错了租期，实际应为三年。",
    inPage: "我在上一条回答中弄错了租期，实际应为三年。",
  },
];

const STREAMED = [
  {
    name: "a heading over a pointer",
    content: "## Terms\n\nSee my earlier answer.\n\n## Pay\n\nPay is 5k a month.",
    shape: ["heading: Pay", "paragraph: Pay is 5k a month."],
  },
  {
    name: "a last heading over a pointer",
    content:
      "## Summary\n\nThe agreement employs D. Reyes.\n\n## Terms\n\nNotice is covered in my earlier answer.",
    shape: ["heading: Summary", "paragraph: The agreement employs D. Reyes."],
  },
  {
    name: "a lead-in over pointers",
    content:
      "The agreement employs D. Reyes.\n\nKey points:\n\n- See my earlier answer for notice.\n- Holiday is covered above.",
    shape: ["paragraph: The agreement employs D. Reyes."],
  },
  {
    name: "a Chinese heading over a pointer",
    content: "## 条款\n\n通知期见上一条回答。\n\n## 薪酬\n\n月薪 18,000 元。",
    shape: ["heading: 薪酬", "paragraph: 月薪 18,000 元。"],
  },
  {
    name: "a quoted pointer and a correction",
    content:
      "合同写道：“租金按月支付，如上所述。”\n\nI made an error in my previous answer and the correct total excludes VAT.",
    shape: [
      "paragraph: 合同写道：“租金按月支付，如上所述。”",
      "paragraph: I made an error in my previous answer and the correct total excludes VAT.",
    ],
  },
];

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

describe("Through the assistant", { timeout: 30_000 }, () => {
  for (const each of PANEL) {
    // protects=the panel keeps the model's words, and Add to note puts them in a page without a sentence that is only a pointer, but with a correction whole; the instructions ask for no pointers; fails_when=the panel is changed, a pure pointer goes into the page, or a correction loses words; no_claim=how often a real model writes one (eval/everyday).
    test(`the panel keeps the model's words, and the page loses only a pure pointer (${each.language})`, async () => {
      const { core, model, mind } = await setUp(() => ({ text: each.reply }));

      const answer = await askAssistant(core, { text: each.question, mindId: mind.id });

      expect(answer.text).toBe(each.reply);
      expect(pageText(pageBlocks(answer, each.question))).toBe(each.inPage);
      const system = promptOf(model)[0]?.text ?? "";
      expect(system).toContain("Never send the reader to an earlier Answer");
      expect(system).toContain("见上一条回答");
    });
  }

  // protects=asked about an earlier Answer, the page may say where it was too; fails_when=its pointer is taken out; no_claim=none beyond protects.
  test("asked about an earlier Answer, the page keeps what points to it", async () => {
    const reply = "Yes. See my earlier answer.";
    const question = "Did your earlier answer cover the notice period?";
    const { core, mind } = await setUp(() => ({ text: reply }));

    const answer = await askAssistant(core, { text: question, mindId: mind.id });

    expect(pageText(pageBlocks(answer, question))).toBe(reply);
  });

  for (const streamInput of [false, true]) {
    for (const each of STREAMED) {
      // protects=written into a Mind, as the input streams in or all at once, only pure pointers go, and a heading or lead-in goes only once nothing under it stayed, never landing over other text; quotes and corrections stay; fails_when=a heading or lead-in is left over the wrong text or with nothing under it, or anything else is lost; no_claim=none beyond protects.
      test(`${each.name}, ${streamInput ? "streamed" : "all at once"}`, async () => {
        const { core, mind } = await setUp(
          ({ results }) =>
            results.length === 0
              ? { calls: [{ tool: "add_to_mind", input: { content: each.content }, streamInput }] }
              : { text: "Done." },
          4,
        );

        await askAssistant(core, { text: "Add this to my notes", mindId: mind.id });

        const client = await connectToMind(core, mind.id);
        await client.settled();
        expect(shapeOf(readMind(client))).toEqual(each.shape);
      });
    }
  }

  // protects=the model is told when a pure pointer was left out of what it wrote, so it can write what it pointed to, and its reply stays; fails_when=the pointer lands in the page, the model isn't told, or the reply is cut; no_claim=none beyond protects.
  test("a pure pointer written into a Mind is left out, and the model is told", async () => {
    const reply = "Added a summary. Notice and holiday are covered in my earlier answer.";
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
    expect(shapeOf(readMind(client))).toEqual([
      "heading: Summary",
      "paragraph: The agreement employs D. Reyes as a software engineer.",
    ]);
    expect(lastResult(model)).toContain(
      "1 sentence that only pointed to an earlier Answer was left out",
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
