import { describe, expect, test } from "vitest";
import { askAssistant, changesIn } from "../helpers/assistant";
import { setUpWithDocuments } from "../helpers/citations";
import { note, writeMind } from "../helpers/minds";
import { type ModelCall, scriptedModel } from "../helpers/models";

/**
 * A model that can't use Tools, asked in the assistant to write into a Mind
 * (#947): it answers in the reply, is told it can't change the Mind, and the
 * Answer records that it couldn't, for the line under it. A Question records
 * nothing.
 */

const TIDES = {
  name: "Tides.md",
  contents: "# Tides\n\nSpring tides happen at new moon and at full moon.\n",
};

/** A model that refuses Tools and structured output, as a small server model does. */
function noToolsModel(systems: string[]) {
  return scriptedModel((call: ModelCall) => {
    if (call.tools.length > 0) {
      return { error: { status: 400, message: "this model does not support tools" } };
    }
    if (call.json) return { error: { status: 400, message: "response_format is not supported" } };
    systems.push(call.system);
    return { text: "Spring tides happen at new moon and at full moon." };
  });
}

describe("A model that can't write into a Mind", { timeout: 30_000 }, () => {
  // protects=asked to write or change a Mind, a model without Tools answers in the reply, is told it can't change the Mind, records that it couldn't and claims nothing; fails_when=the line's record is missing, the model isn't told, it is asked again to write, or a claim is flagged; no_claim=what a real small model says.
  test.each(["write a summary page", "把这段改长一点"])(
    "“%s” is answered in the reply, and the Answer says it couldn't write",
    async (text) => {
      const systems: string[] = [];
      const model = noToolsModel(systems);
      const { core, mind, client } = await setUpWithDocuments(model, [TIDES]);
      writeMind(client, [note("Tides follow the moon.")]);
      await client.settled();

      const answer = await askAssistant(core, { text, mindId: mind.id });

      expect(answer.text).toBe("Spring tides happen at new moon and at full moon.");
      expect(changesIn(answer)).toEqual([]);
      expect(answer.steps?.cannotWrite).toBe(true);
      expect(answer.steps?.unbackedClaims).toBeUndefined();
      expect(systems.at(-1)).toContain("you can't change a Mind in this Answer");
      expect(systems).toHaveLength(1);
    },
  );

  // protects=a Question asked of the same model records nothing, so no line shows; fails_when=it records that it couldn't write; no_claim=none beyond protects.
  test("a Question records nothing", async () => {
    const systems: string[] = [];
    const { core, mind } = await setUpWithDocuments(noToolsModel(systems), [TIDES]);

    const answer = await askAssistant(core, {
      text: "What does the paper say about spring tides?",
      mindId: mind.id,
    });

    expect(answer.steps?.cannotWrite).toBeUndefined();
    expect(systems.at(-1)).not.toContain("you can't change a Mind");
  });
});
