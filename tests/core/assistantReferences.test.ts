import type { JSONContent } from "@tiptap/core";
import { describe, expect, test } from "vitest";
import type { Core } from "../../src/core";
import { namesReferenceList } from "../../src/shared/referenceList";
import { askAssistant, changesIn } from "../helpers/assistant";
import { isCheckRun } from "../helpers/checks";
import { setUpWithDocuments, shownPassages } from "../helpers/citations";
import { connectToMind, type MindClient } from "../helpers/mindClient";
import { heading, note, readMind, resolveChanges, writeMind } from "../helpers/minds";
import { type ModelCall, type ScriptedReply, scriptedModel } from "../helpers/models";

/**
 * A Mind's References hold only its reference list, and what the assistant writes into it stays
 * in its text, above them (#1483), through the core's public interface as the assistant asks, in
 * English and Chinese. The model is told IncarnaMind lists the References itself; a reference list
 * a model writes anyway as footnote definitions goes with them; and what is added at the Mind's end,
 * after its last paragraph or into a section, and what a rewrite of a section or of the last
 * paragraph changes, stays above a reference list the Mind keeps. The model's choices are
 * scripted; real models' are checked by hand.
 */

interface Language {
  name: "English" | "Chinese";
  file: { name: string; contents: string };
  /** The Document's sentences an Answer cites, each a quote. */
  facts: { lead: string; done: string; budget: string };
  /** A sentence of the Mind's text citing `fact` with marker 1. */
  cited(fact: string): string;
  /** What the scripted rewrite adds to the end of the text it is given. */
  rewritten: string;
  /** A first write, titled, that ends with its own reference list as models write one: footnote definitions under a heading. */
  summary: { title: string; content(lead: string): string };
  /** A Mind the User wrote, ending with their reference list. */
  draft: { title: string; blocks: JSONContent[]; entries: string[] };
  /** A reference list's heading, as the title of a Mind that holds nothing else. */
  references: string;
  style: string;
  asks: {
    write: string;
    append: string;
    afterLast: string;
    intoSection: string;
    rewriteSection: string;
    rewriteLast: string;
  };
}

const ENGLISH: Language = {
  name: "English",
  file: {
    name: "Project report.md",
    contents:
      "# Project report\n\nThe project lead is Li Ming.\n\nThe project was completed on 15 December 2025.\n\nThe budget was two million euros.\n",
  },
  facts: {
    lead: "The project lead is Li Ming",
    done: "The project was completed on 15 December 2025",
    budget: "The budget was two million euros",
  },
  cited: (fact) => `${fact} [^1].`,
  rewritten: " Put formally.",
  summary: {
    title: "Project summary",
    content: (lead) => `# Project summary\n\n${lead}\n\n## References\n\n[^1]: Project report.md`,
  },
  draft: {
    title: "Project report draft",
    blocks: [
      heading(1, "Project report draft"),
      heading(2, "Background"),
      note("The project started in 2024."),
      heading(2, "References"),
      note("Smith, J. (2020). Project methods. Journal of Projects, 3(2), 1–10."),
    ],
    entries: ["Smith, J. (2020). Project methods. Journal of Projects, 3(2), 1–10."],
  },
  references: "References",
  style: "apa",
  asks: {
    write: "Write a summary of the project",
    append: "Add when the project was completed",
    afterLast: "Add the budget after the last paragraph",
    intoSection: "Add who leads the project to the report",
    rewriteSection: "Rewrite the report more formally",
    rewriteLast: "Rewrite the last paragraph more formally",
  },
};

const CHINESE: Language = {
  name: "Chinese",
  file: {
    name: "项目报告.md",
    contents:
      "# 项目报告\n\n项目负责人是李明。\n\n项目于2025年12月15日完成。\n\n项目预算为200万欧元。\n",
  },
  facts: {
    lead: "项目负责人是李明",
    done: "项目于2025年12月15日完成",
    budget: "项目预算为200万欧元",
  },
  cited: (fact) => `${fact}[^1]。`,
  rewritten: "（已改写）",
  summary: {
    title: "项目总结",
    // As Chinese is written: the heading a line of its own, the definitions with a full-width colon.
    content: (lead) => `# 项目总结\n\n${lead}\n\n**参考文献：**\n\n[^1]：项目报告.md`,
  },
  draft: {
    title: "项目报告草稿",
    blocks: [
      heading(1, "项目报告草稿"),
      heading(2, "背景"),
      note("项目始于2024年。"),
      note("参考文献："),
      note("[1] 张三. 项目方法[J]. 项目学报, 2020, 3(2): 1-10."),
    ],
    entries: ["[1] 张三. 项目方法[J]. 项目学报, 2020, 3(2): 1-10."],
  },
  references: "参考文献",
  style: "gb-t-7714-2015-numeric",
  asks: {
    write: "根据文件写一份项目总结",
    append: "加上项目完成的时间",
    afterLast: "在最后一段后面加上项目预算",
    intoSection: "在报告里加上项目负责人",
    rewriteSection: "把报告改写得更正式",
    rewriteLast: "把最后一段改写得更正式",
  },
};

/** What the scripted model does for the Question being asked: cite `quote` first, if any, then call `tool`. */
interface Step {
  quote?: string;
  tool: "create_mind" | "add_to_mind" | "edit_mind";
  input: Record<string, unknown>;
}

const called = (call: ModelCall, tool: string) => call.results.some((each) => each.tool === tool);

const PASSAGE = /<passage>\n([\s\S]*?)\n<\/passage>/;

/**
 * Does each Step as it is asked; checks each sentence it wrote as supported; and rewrites a passage
 * as edit_mind asks by adding `rewritten` at its end.
 */
function stepModel(language: Language, step: () => Step, calls: ModelCall[]) {
  return scriptedModel((call): ScriptedReply => {
    if (isCheckRun(call)) {
      return called(call, "search_documents")
        ? { text: "Verdict: supported" }
        : { calls: [{ tool: "search_documents", input: { query: "project" } }] };
    }
    const passage = PASSAGE.exec(call.prompt)?.[1];
    if (passage !== undefined) return { text: `${passage}${language.rewritten}` };
    calls.push(call);
    const { quote, tool, input } = step();
    if (quote && !called(call, "search_documents")) {
      return { calls: [{ tool: "search_documents", input: { query: quote } }] };
    }
    if (quote && !called(call, "cite")) {
      const shown = call.results
        .filter((each) => each.tool === "search_documents")
        .flatMap((each) => shownPassages(each.text));
      const passage = shown.find((each) => each.text.includes(quote))?.id ?? "";
      return { calls: [{ tool: "cite", input: { citations: [{ marker: 1, passage, quote }] } }] };
    }
    if (!called(call, tool)) return { calls: [{ tool, input }] };
    return { text: "Done." };
  });
}

/** The Mind's paragraphs and headings, each with whether it holds a Citation. */
function linesOf(client: MindClient): { text: string; cited: boolean }[] {
  const lines: { text: string; cited: boolean }[] = [];
  readMind(client).descendants((node) => {
    if (!node.isTextblock) return true;
    let cited = false;
    node.descendants((child) => {
      if (child.type.name === "citation") cited = true;
      return true;
    });
    lines.push({ text: node.textContent, cited });
    return false;
  });
  return lines;
}

async function setUp(language: Language) {
  let step: Step | null = null;
  const calls: ModelCall[] = [];
  const model = stepModel(
    language,
    () => {
      if (!step) throw new Error("No step is scripted for this Question.");
      return step;
    },
    calls,
  );
  const { core, mind, client } = await setUpWithDocuments(model, [language.file]);
  /** Asks in the assistant with the Mind in focus, and applies what it changed there, as its window does. */
  const ask = async (text: string, next: Step, mindId: string, page: MindClient) => {
    step = next;
    const answer = await askAssistant(core, { text, mindId });
    await page.settled();
    await resolveChanges(page, "accept");
    await page.settled();
    return answer;
  };
  return { core, mind, client, calls, ask };
}

/** The Mind's reference list holds exactly `entries`, and every line with a Citation is above it. */
function expectReferencesKept(client: MindClient, entries: readonly string[]) {
  const lines = linesOf(client);
  const at = lines.findIndex((line) => namesReferenceList(line.text));
  expect(at).toBeGreaterThan(0);
  expect(lines.slice(at + 1).map((line) => line.text)).toEqual(entries);
  expect(lines.slice(at + 1).some((line) => line.cited)).toBe(false);
}

async function exported(core: Core, mindId: string): Promise<string> {
  return new TextDecoder().decode((await core.exportMind(mindId, { format: "markdown" })).data);
}

describe("What the assistant writes stays above a Mind's References (#1483)", {
  timeout: 90_000,
}, () => {
  // protects=a model that ends a piece with its own References as footnote definitions ("## References", or "**参考文献：**" with "[^1]："), as models do, leaves no such heading in the Mind; what it adds later, at the end and after the last paragraph, stays in the text; and the References IncarnaMind builds (in an export) list the Document alone, below every cited sentence; fails_when=the heading stays and what is added later lands under it, or the References hold a sentence; no_claim=that a real model leaves its own References out, which is checked by hand.
  test.each([ENGLISH, CHINESE])(
    "$name: a reference list the model writes itself goes, and what it adds later stays in the text",
    async (language) => {
      const { core, mind, client, calls, ask } = await setUp(language);
      const { facts } = language;
      const first = await ask(
        language.asks.write,
        {
          quote: facts.lead,
          tool: "create_mind",
          input: {
            title: language.summary.title,
            content: language.summary.content(language.cited(facts.lead)),
          },
        },
        mind.id,
        client,
      );
      const made = changesIn(first).find((change) => change.created);
      if (!made) throw new Error("The Answer made no Mind.");
      const page = await connectToMind(core, made.mindId);
      await resolveChanges(page, "accept");
      await page.settled();

      await ask(
        language.asks.append,
        { quote: facts.done, tool: "add_to_mind", input: { content: language.cited(facts.done) } },
        made.mindId,
        page,
      );
      await ask(
        language.asks.afterLast,
        {
          quote: facts.budget,
          tool: "add_to_mind",
          input: { after: facts.done, content: language.cited(facts.budget) },
        },
        made.mindId,
        page,
      );

      const lines = linesOf(page);
      expect(lines.filter((line) => namesReferenceList(line.text))).toEqual([]);
      expect(lines.filter((line) => line.cited).map((line) => line.text)).toEqual([
        expect.stringMatching(new RegExp(`^${facts.lead}`)),
        expect.stringMatching(new RegExp(`^${facts.done}`)),
        expect.stringMatching(new RegExp(`^${facts.budget}`)),
      ]);
      await core.setMindCitationStyle(made.mindId, language.style);
      const markdown = await exported(core, made.mindId);
      const [text = "", references = ""] = markdown.split(
        language.name === "English" ? "## References" : "## 参考文献",
      );
      for (const fact of Object.values(facts)) {
        expect(text).toContain(fact);
        expect(references).not.toContain(fact);
      }
      expect(references).toContain(language.file.name.replace(/\.md$/, ""));

      // The model is told why: IncarnaMind lists the References itself.
      const asked = calls.find((call) => call.tools.includes("add_to_mind"));
      expect(asked?.system).toContain(
        "IncarnaMind lists the Documents a Mind cites under References (参考文献) at its end by itself",
      );
      const content = asked?.options.tools?.find((tool) => tool.name === "add_to_mind");
      expect(JSON.stringify(content)).toContain(
        "write no References or 参考文献 heading, no list of sources and no footnote definitions",
      );
    },
  );

  // protects=in a Mind that ends with the User's reference list (a "References" heading, or "参考文献：" on a line of its own), what the assistant adds at the end, after the last paragraph and into the report's section goes above the list, and rewriting that section or the last paragraph leaves the list as it was; fails_when=a cited sentence lands under the heading, or a reference entry is rewritten; no_claim=where a real model chooses to put it.
  test.each([ENGLISH, CHINESE])(
    "$name: what the assistant adds and rewrites stays above the reference list the Mind keeps",
    async (language) => {
      const { mind, client, ask } = await setUp(language);
      const { facts, draft } = language;
      writeMind(client, draft.blocks);
      await client.settled();

      await ask(
        language.asks.append,
        { quote: facts.done, tool: "add_to_mind", input: { content: language.cited(facts.done) } },
        mind.id,
        client,
      );
      expectReferencesKept(client, draft.entries);
      await ask(
        language.asks.afterLast,
        {
          quote: facts.budget,
          tool: "add_to_mind",
          input: { after: facts.done, content: language.cited(facts.budget) },
        },
        mind.id,
        client,
      );
      expectReferencesKept(client, draft.entries);
      await ask(
        language.asks.intoSection,
        {
          quote: facts.lead,
          tool: "add_to_mind",
          input: { after: draft.title, content: language.cited(facts.lead) },
        },
        mind.id,
        client,
      );
      expectReferencesKept(client, draft.entries);
      await ask(
        language.asks.rewriteSection,
        {
          tool: "edit_mind",
          input: { part: draft.title, instruction: language.asks.rewriteSection },
        },
        mind.id,
        client,
      );
      expectReferencesKept(client, draft.entries);
      await ask(
        language.asks.rewriteLast,
        { tool: "edit_mind", input: { instruction: language.asks.rewriteLast } },
        mind.id,
        client,
      );
      expectReferencesKept(client, draft.entries);

      const cited = linesOf(client).filter((line) => line.cited);
      expect(cited.map((line) => line.text)).toEqual([
        expect.stringMatching(new RegExp(`^${facts.done}`)),
        expect.stringMatching(new RegExp(`^${facts.budget}`)),
        expect.stringMatching(new RegExp(`^${facts.lead}`)),
      ]);
      // The section's rewrite and the last paragraph's both end in the last cited sentence.
      expect(cited.at(-1)?.text.endsWith(`${language.rewritten}${language.rewritten}`)).toBe(true);
    },
  );

  // protects=a Mind that is only a reference list, its title naming it, has no text above the list to keep apart: what the assistant adds goes at its end, as before; fails_when=it goes above the title; no_claim=none beyond protects.
  test.each([ENGLISH, CHINESE])(
    "$name: in a Mind that is only a reference list, what is added goes at its end",
    async (language) => {
      const { mind, client, ask } = await setUp(language);
      const { facts, draft } = language;
      writeMind(client, [heading(1, language.references), note(draft.entries[0] ?? "")]);
      await client.settled();

      await ask(
        language.asks.append,
        { quote: facts.done, tool: "add_to_mind", input: { content: language.cited(facts.done) } },
        mind.id,
        client,
      );

      expect(linesOf(client)).toEqual([
        { text: language.references, cited: false },
        { text: draft.entries[0], cited: false },
        { text: expect.stringMatching(new RegExp(`^${facts.done}`)), cited: true },
      ]);
    },
  );
});
