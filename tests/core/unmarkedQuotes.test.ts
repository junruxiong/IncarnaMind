import { describe, expect, test } from "vitest";
import { matchUnmarked, parseAsked } from "../../src/core/answers/unmarkedQuotes";
import { askAndFinish, citationsIn, setUpWithDocuments } from "../helpers/citations";
import { type ModelCall, type ScriptedReply, scriptedModel } from "../helpers/models";

/*
 * #760: an Answer that quotes the Passages it read without Citation markers
 * gets its Citations, each found on its page; when nothing matches it, the
 * model is asked for them once, and never "no sources".
 */

const FIRST =
  "Tea drinking began in southwest China, where it was first used as a medicinal drink.";
const SECOND = "Traders carried tea along the Tea Horse Road to Tibet.";
const TEA = {
  name: "Tea history.md",
  contents: `# Tea history\n\n${FIRST}\n\n${SECOND}\n`,
};

/** A model that searches once, then answers `answer` with no Citations; `asked` replies to the one request for them. */
function quotingModel(
  answer: string,
  asked: (call: ModelCall) => ScriptedReply = () => ({ text: "{}" }),
) {
  return scriptedModel((call) => {
    if (call.json) return asked(call);
    if (!call.results.some((result) => result.tool === "search_documents")) {
      return {
        calls: [{ tool: "search_documents", input: { query: "where tea was first drunk" } }],
      };
    }
    return { text: answer };
  });
}

/** The ids of the Passages a search result lists ("P1", …). */
const passageIds = (text: string) => [...text.matchAll(/<passage id=\\?"(P\d+)/g)].map((m) => m[1]);

describe("An Answer that quotes without markers (#760)", { timeout: 30_000 }, () => {
  // protects=a quotation the Answer copied from a Passage gets its Citation, found on its page, with no request more; fails_when=it stays uncited, is cited not found, or the model is asked again; no_claim=the line under the Answer (tests/renderer/answerSteps.test.ts).
  test("a quotation copied from a Passage gets its Citation, found, and the model isn't asked again", async () => {
    const model = quotingModel(`As one article puts it, "${FIRST}" So it began in China.`);
    const { core, client, mind } = await setUpWithDocuments(model, [TEA]);

    const { answerId, finished } = await askAndFinish(
      core,
      client,
      mind.id,
      "Where was tea first drunk?",
    );

    expect(citationsIn(client, answerId)).toEqual([
      expect.objectContaining({ check: "found", quote: FIRST }),
    ]);
    expect(finished.citedAfterChecking).toEqual({ count: 1, asked: false });
    expect(model.doStreamCalls).toHaveLength(2);
  });

  // protects=a marker the model wrote but gave no record for takes the Citation found for its sentence, instead of being removed; fails_when=the marker is removed, or a second marker is added; no_claim=none beyond protects.
  test("a marker written with no record behind it takes the Citation its quotation matches", async () => {
    const model = quotingModel(`It is described as "${SECOND}" [^1] That is all.`);
    const { core, client, mind } = await setUpWithDocuments(model, [TEA]);

    const { answerId, finished } = await askAndFinish(
      core,
      client,
      mind.id,
      "How did tea reach Tibet?",
    );

    expect(citationsIn(client, answerId)).toEqual([
      expect.objectContaining({ check: "found", quote: SECOND }),
    ]);
    expect(finished.citedAfterChecking).toEqual({ count: 1, asked: false });
    expect(finished.citations).toHaveLength(1);
  });

  // protects=a sentence worded closely like a Passage's is cited with that sentence as its quote; fails_when=it stays uncited or cites another Passage sentence; no_claim=how close is close enough beyond this pair.
  test("a sentence worded closely like a Passage's is cited with the Passage's sentence", async () => {
    const model = quotingModel(
      "Tea drinking began in southwest China, where people first used it as a medicinal drink.",
    );
    const { core, client, mind } = await setUpWithDocuments(model, [TEA]);

    const { answerId, finished } = await askAndFinish(
      core,
      client,
      mind.id,
      "Where was tea first drunk?",
    );

    expect(citationsIn(client, answerId)).toEqual([
      expect.objectContaining({ check: "found", quote: FIRST }),
    ]);
    expect(finished.citedAfterChecking).toEqual({ count: 1, asked: false });
  });

  // protects=when nothing matches the model is asked once for its Citations, with the Passages it read, and only a quote the check finds is taken; fails_when=no request is made, more than one is, or an unfound quote is cited; no_claim=what a real model replies.
  test("when nothing matches, the model is asked once, with the Passages it read", async () => {
    const asked: ModelCall[] = [];
    const model = quotingModel("Traders took tea to Tibet on horseback.", (call) => {
      asked.push(call);
      const [id] = passageIds(
        call.options.prompt.map((message) => JSON.stringify(message)).join(""),
      );
      return {
        text: JSON.stringify({
          citations: [
            {
              sentence: "Traders took tea to Tibet on horseback.",
              passage: id,
              quote: SECOND,
            },
            {
              sentence: "Traders took tea to Tibet on horseback.",
              passage: id,
              quote: "Not in it.",
            },
          ],
        }),
      };
    });
    const { core, client, mind } = await setUpWithDocuments(model, [TEA]);

    const { answerId, finished } = await askAndFinish(
      core,
      client,
      mind.id,
      "How did tea reach Tibet?",
    );

    expect(asked).toHaveLength(1);
    expect(JSON.stringify(asked[0]?.options.prompt)).toContain(SECOND);
    expect(citationsIn(client, answerId)).toEqual([
      expect.objectContaining({ check: "found", quote: SECOND }),
    ]);
    expect(finished.citedAfterChecking).toEqual({ count: 1, asked: true });
  });

  // protects=the one request is made at most once, and an Answer it can't help stays uncited; an Answer that shares nothing with the Passages isn't asked about; fails_when=it asks twice, asks for a decline, or invents a Citation; no_claim=none beyond protects.
  test("asked and given nothing, the Answer stays uncited after one request; a decline isn't asked about", async () => {
    let requests = 0;
    const replyNothing = () => {
      requests++;
      return { text: JSON.stringify({ citations: [] }) };
    };
    const drawing = quotingModel("Tea reached Tibet with traders, I think.", replyNothing);
    const first = await setUpWithDocuments(drawing, [TEA]);

    const asked = await askAndFinish(first.core, first.client, first.mind.id, "Who carried it?");

    expect(requests).toBe(1);
    expect(citationsIn(first.client, asked.answerId)).toEqual([]);
    expect(asked.finished.citedAfterChecking).toEqual({ count: 0, asked: true });

    const declining = quotingModel("Your Documents don't say who grew it.", replyNothing);
    const second = await setUpWithDocuments(declining, [TEA]);
    const declined = await askAndFinish(second.core, second.client, second.mind.id, "Who grew it?");
    expect(requests).toBe(1);
    expect(declined.finished.citedAfterChecking).toEqual({ count: 0, asked: false });
  });

  // protects=an Answer that cites with its own markers is left as it is, and nothing is asked; fails_when=its markers move or a request is made; no_claim=none beyond protects.
  test("an Answer with its own Citations isn't asked again", async () => {
    let requests = 0;
    const model = scriptedModel((call) => {
      if (call.json) {
        requests++;
        return { text: "{}" };
      }
      const searches = call.results.filter((result) => result.tool === "search_documents");
      if (searches.length === 0) {
        return { calls: [{ tool: "search_documents", input: { query: "tea" } }] };
      }
      if (call.results.some((result) => result.tool === "cite")) return { text: "" };
      const [id] = passageIds(searches[0]?.text ?? "");
      return {
        text: "It began in southwest China [^1]. I can't say more.",
        calls: [{ tool: "cite", input: { citations: [{ marker: 1, passage: id, quote: FIRST }] } }],
      };
    });
    const { core, client, mind } = await setUpWithDocuments(model, [TEA]);

    const { answerId, finished } = await askAndFinish(
      core,
      client,
      mind.id,
      "Where was tea first drunk?",
    );

    expect(requests).toBe(0);
    expect(citationsIn(client, answerId)).toHaveLength(1);
    expect(finished.citedAfterChecking).toEqual({ count: 0, asked: false });
  });
});

describe("Matching sentences to Passages", () => {
  const found = (text: string) => (_passage: string, quote: string) => text.includes(quote);

  // protects=quotations in curly quotes and Chinese corner brackets match, short ones don't, and each sentence gets one Citation; fails_when=a two-word quotation is cited or a sentence gets two; no_claim=the Citation check itself (tests/core/citations.test.ts).
  test("quotations in any quotation marks, long enough to say something", () => {
    const passages = [{ id: "P1", text: `${FIRST}\n茶最早在中国西南地区作为药用饮品。` }];
    const text = `${FIRST}\n茶最早在中国西南地区作为药用饮品。`;
    expect(
      matchUnmarked(
        `It says “${FIRST}” Also “began in”. 文中说「茶最早在中国西南地区作为药用饮品。」`,
        passages,
        found(text),
      ),
    ).toEqual([
      { passage: "P1", quote: FIRST, sentence: `It says “${FIRST}` },
      {
        passage: "P1",
        quote: "茶最早在中国西南地区作为药用饮品。",
        sentence: "文中说「茶最早在中国西南地区作为药用饮品。",
      },
    ]);
  });

  // protects=a sentence that shares only a few words with a Passage isn't cited; fails_when=a loose paraphrase is cited; no_claim=none beyond protects.
  test("a loosely related sentence isn't cited", () => {
    expect(
      matchUnmarked("China is where tea is from.", [{ id: "P1", text: FIRST }], found(FIRST)),
    ).toEqual([]);
  });

  // protects=the reply to the one request is read leniently and only well-formed items are kept; fails_when=a malformed item throws or is kept; no_claim=none beyond protects.
  test("the one request's reply: only whole items", () => {
    expect(
      parseAsked({
        citations: [
          { sentence: " A. ", passage: "P1", quote: " q " },
          { sentence: "B.", passage: 2, quote: "q" },
          "nonsense",
        ],
      }),
    ).toEqual([{ sentence: "A.", passage: "P1", quote: "q" }]);
    expect(parseAsked(undefined)).toEqual([]);
  });
});
