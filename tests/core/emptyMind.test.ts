import { describe, expect, test } from "vitest";
import type { Language } from "../../src/core/language";
import { askAssistant } from "../helpers/assistant";
import { createTempDataFolder, startCore } from "../helpers/core";
import { connectToMind } from "../helpers/mindClient";
import { note, readMind, writeMind } from "../helpers/minds";
import { type ModelCall, scriptedModel, scriptedModels } from "../helpers/models";

/**
 * A new, empty, unnamed Mind is the open Mind (#1212): the model is told the
 * page the User is in is there and empty, so a request to write lands in it.
 */

const SUMMARY = "The agreement runs for three years from 1 November 2026.";
const NO_MIND = "No Mind is open, so I can't write it there.";

const CASES: { language: Language; untitled: string; request: string }[] = [
  { language: "en", untitled: "Untitled", request: "Write a brief summary of this agreement." },
  { language: "zh-CN", untitled: "未命名", request: "写一份简要总结。" },
];

async function setUp(language: Language) {
  const calls: ModelCall[] = [];
  // As models did: an unnamed, empty <mind> read as no Mind open.
  const model = scriptedModel((call) => {
    calls.push(call);
    if (call.prompt.includes('<mind name="">')) return { text: NO_MIND };
    return call.results.length === 0
      ? { calls: [{ tool: "add_to_mind", input: { content: SUMMARY } }] }
      : { text: "Added the summary." };
  });
  const core = startCore(await createTempDataFolder(), {
    createChatModel: scriptedModels(model).createChatModel,
  });
  await core.updateSettings({ user: { language } });
  await core.saveChatProvider({ kind: "ollama", modelId: "local-model" });
  return { core, calls };
}

describe("An empty, unnamed Mind is the open Mind", { timeout: 30_000 }, () => {
  for (const each of CASES) {
    // protects=asked to write in a new, empty, unnamed Mind, the model is told the Mind is there by its shown name and empty, and what it writes lands there; fails_when=the Mind reaches the model with no name or no word that it is open and empty, or the reply says none is open; no_claim=how a real model reads it (checked by hand).
    test(`a request to write lands in it, and the model is told it is open and empty (${each.language})`, async () => {
      const { core, calls } = await setUp(each.language);
      const mind = await core.createMind({ title: "" });

      const answer = await askAssistant(core, { text: each.request, mindId: mind.id });

      expect(answer.text).not.toContain(NO_MIND);
      const [first] = calls;
      expect(first?.prompt).toContain(`<mind name="${each.untitled}">`);
      expect(first?.system).toContain("The Mind in <mind> is open and still empty");
      const client = await connectToMind(core, mind.id);
      await client.settled();
      expect(readMind(client).textContent).toContain(SUMMARY);
    });
  }

  // protects=a Mind with text gets no note that it is empty; fails_when=the note goes with any Mind; no_claim=none beyond protects.
  test("a Mind with text is not called empty", async () => {
    const { core, calls } = await setUp("en");
    const mind = await core.createMind({ title: "" });
    const client = await connectToMind(core, mind.id);
    writeMind(client, [note("Notes for Monday.")]);
    await client.settled();

    await askAssistant(core, { text: "Write a brief summary of this agreement.", mindId: mind.id });

    expect(calls[0]?.prompt).toContain('<mind name="Notes for Monday">');
    expect(calls[0]?.system).not.toContain("open and still empty");
  });
});
