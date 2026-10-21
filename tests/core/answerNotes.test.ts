import { describe, expect, test } from "vitest";
import { askAssistant } from "../helpers/assistant";
import { setUpWithDocuments } from "../helpers/citations";
import { type ModelCall, type ScriptedReply, scriptedModel } from "../helpers/models";

const FIRST =
  "Tea drinking began in southwest China, where it was first used as a medicinal drink.";
const TEA = { name: "Tea history.md", contents: `# Tea history\n\n${FIRST}\n` };

const QUESTION = "Which Document is the fastest to read? Answer in one short sentence.";

/** A model running on in its reply, with no Citation in it, far past anything it was asked for. */
const NOTES =
  "Wait, there may be a second copy of this file. Let's search again to be safe. ".repeat(200);

/** A model that searches once, then writes `reply` for what it read. */
const searchingThenWriting = (reply: string) =>
  scriptedModel((call: ModelCall): ScriptedReply => {
    if (call.json) return { text: JSON.stringify({ citations: [] }) };
    const search = call.results.find((result) => result.tool === "search_documents");
    if (!search) return { calls: [{ tool: "search_documents", input: { query: "tea" } }] };
    return { text: reply };
  });

describe("A model that writes its notes instead of an Answer (#1344)", { timeout: 30_000 }, () => {
  // protects=a reply that runs on for pages after a search, with no Citation in it, never stays as the Answer: it has no text and fails with a line to try again; fails_when=the notes are the Answer's text or it is "done"; no_claim=what a real model writes.
  test("a reply that runs on with no Citation fails as no Answer, with none of it left", async () => {
    const { core } = await setUpWithDocuments(searchingThenWriting(NOTES), [TEA]);

    const answer = await askAssistant(core, { text: QUESTION }, "failed");

    expect(answer.errorKind).toBe("no-answer");
    expect(answer.text).toBe("");
  });

  // protects=a general-knowledge Answer after a search that found nothing, with no marker in it, is an Answer, not notes; fails_when=a reply of a couple of thousand characters with no marker is failed; no_claim=what the Answer says.
  test("a few thousand characters with no marker after a search stay", async () => {
    const general = `The Documents say nothing about brewing temperatures, so this is general knowledge. ${"Green tea is brewed cooler than black tea. ".repeat(45)}`;
    const { core } = await setUpWithDocuments(searchingThenWriting(general), [TEA]);

    const answer = await askAssistant(core, { text: "How hot should the water be for tea?" });

    expect(answer.status).toBe("done");
    expect(answer.text.length).toBeGreaterThan(1_800);
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

  // protects=when the model uses every step on searches, the last step is told it has no more calls and to say what wasn't checked, may not call a Tool, and is capped, so a model that wanted another call neither writes it out nor thinks for pages; fails_when=the last request has no such line, still allows a call, or has no cap; no_claim=what a real model writes after it (#1344).
  test("the last step is told to write from what was read, may call nothing, and is capped", async () => {
    const calls: ModelCall[] = [];
    const model = scriptedModel((call: ModelCall): ScriptedReply => {
      if (call.json) return { text: JSON.stringify({ citations: [] }) };
      calls.push(call);
      if (call.options.toolChoice?.type === "none") {
        return { text: "I read some of the Documents, and didn't check the rest." };
      }
      return { calls: [{ tool: "search_documents", input: { query: `tea ${calls.length}` } }] };
    });
    const { core } = await setUpWithDocuments(model, [TEA]);

    const answer = await askAssistant(core, { text: QUESTION });

    const last = calls.at(-1);
    expect(answer.status).toBe("done");
    expect(last?.options.toolChoice).toEqual({ type: "none" });
    expect(last?.options.maxOutputTokens).toBe(8_000);
    expect(last?.prompt).toContain(
      "You can make no more calls, and none can record a Citation now",
    );
    expect(last?.prompt).toContain("say plainly what you didn't check");
    for (const earlier of calls.slice(0, -1)) {
      expect(earlier.prompt).not.toContain("You can make no more calls");
      expect(earlier.options.maxOutputTokens).toBeUndefined();
    }
  });
});
