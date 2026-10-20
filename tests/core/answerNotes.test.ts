import { describe, expect, test } from "vitest";
import { askAssistant } from "../helpers/assistant";
import { setUpWithDocuments } from "../helpers/citations";
import { type ModelCall, type ScriptedReply, scriptedModel } from "../helpers/models";

const FIRST =
  "Tea drinking began in southwest China, where it was first used as a medicinal drink.";
const TEA = { name: "Tea history.md", contents: `# Tea history\n\n${FIRST}\n` };

const QUESTION = "Which Document is the fastest to read? Answer in one short sentence.";

/** The model thinking aloud in its reply, as a Question about Documents sometimes finds it. */
const NOTES =
  "Wait, there may be a second copy of this file. Let's search again to be safe. ".repeat(60);

/** A model that searches once, then writes `reply` for what it read. */
const searchingThenWriting = (reply: string) =>
  scriptedModel((call: ModelCall): ScriptedReply => {
    if (call.json) return { text: JSON.stringify({ citations: [] }) };
    const search = call.results.find((result) => result.tool === "search_documents");
    if (!search) return { calls: [{ tool: "search_documents", input: { query: "tea" } }] };
    return { text: reply };
  });

describe("A model that writes its notes instead of an Answer (#1344)", { timeout: 30_000 }, () => {
  // protects=pages of notes after a search, with no Citation in them, never stay as the Answer: it has no text and fails with a line to try again; fails_when=the notes are the Answer's text or it is "done"; no_claim=what a real model writes.
  test("fails as no Answer, with none of the notes left in it", async () => {
    const { core } = await setUpWithDocuments(searchingThenWriting(NOTES), [TEA]);

    const answer = await askAssistant(core, { text: QUESTION }, "failed");

    expect(answer.errorKind).toBe("no-answer");
    expect(answer.text).toBe("");
  });

  // protects=a long Answer whose sentences carry Citations is not taken for notes, however long; fails_when=a cited Answer is cut off at the length of notes; no_claim=none beyond protects.
  test("a long Answer with a Citation stays", async () => {
    const long = `Tea was first drunk in southwest China [^1].\n\n${"It spread along the trade routes. ".repeat(80)}`;
    const model = scriptedModel((call: ModelCall): ScriptedReply => {
      if (call.json) return { text: JSON.stringify({ citations: [] }) };
      const search = call.results.find((result) => result.tool === "search_documents");
      if (!search) return { calls: [{ tool: "search_documents", input: { query: "tea" } }] };
      return { text: long };
    });
    const { core } = await setUpWithDocuments(model, [TEA]);

    const answer = await askAssistant(core, { text: "Where was tea first drunk?" });

    expect(answer.status).toBe("done");
    expect(answer.text.length).toBeGreaterThan(2_000);
  });

  // protects=long text before any search is the User's request, an essay or a poem, and stays; fails_when=a long reply with no search is failed; no_claim=none beyond protects.
  test("long text without a search stays", async () => {
    const model = scriptedModel(
      (call: ModelCall): ScriptedReply =>
        call.json ? { text: JSON.stringify({ citations: [] }) } : { text: NOTES },
    );
    const { core } = await setUpWithDocuments(model, [TEA]);

    const answer = await askAssistant(core, { text: "Write me a long poem about tea." });

    expect(answer.status).toBe("done");
    expect(answer.text.length).toBeGreaterThan(3_000);
  });
});
