import type { MockLanguageModelV4 } from "ai/test";
import { describe, expect, onTestFinished, test, vi } from "vitest";
import type { Connector, Core } from "../../src/core";
import { translate } from "../../src/shared/i18n";
import { askAndFinish } from "../helpers/citations";
import { testProcesses, tideServer, waitForState } from "../helpers/connectors";
import { createTempDataFolder, startCore } from "../helpers/core";
import { connectToMind } from "../helpers/mindClient";
import { answerText } from "../helpers/minds";
import { type ModelCall, scriptedModel, scriptedModels } from "../helpers/models";
import { type FakeRemote, startFakeRemote } from "../helpers/remoteMcp";

/**
 * "Keep everything on this computer" and Connectors: one on another computer
 * isn't connected to or offered to Answers, and says why; programs on this
 * computer and servers at a loopback address keep working. The server on
 * "another computer" is a fake on this one, reached under a name that is not
 * a loopback address (the test stands in for DNS).
 */

const FAR_HOST = "wiki.example.test";
const SEARCH = "wiki__search_wiki";
const LOOKUP = "tides__lookup_tide";

/** A model that uses `tool` when it is offered, then says what it got; else says it can't. */
function usingModel(tool: string, input: unknown): MockLanguageModelV4 {
  return scriptedModel((call: ModelCall) => {
    if (!call.tools.includes(tool)) return { text: "No such Tool." };
    const got = call.results.find((result) => result.tool === tool);
    if (!got) return { text: "Let me look.", calls: [{ tool, input }] };
    return { text: `Got: ${got.text}` };
  });
}

/** fetch as if `FAR_HOST` were the fake service, and the names it was asked for. */
function standInForDns(fake: FakeRemote) {
  const asked: string[] = [];
  const real = globalThis.fetch;
  vi.stubGlobal("fetch", (input: Request | URL | string, init?: RequestInit) => {
    const url = new URL(input instanceof Request ? input.url : String(input));
    if (url.hostname !== FAR_HOST) return real(input, init);
    asked.push(url.href);
    const here = new URL(fake.url);
    url.protocol = here.protocol;
    url.hostname = here.hostname;
    url.port = here.port;
    return real(url, init);
  });
  onTestFinished(() => {
    vi.unstubAllGlobals();
  });
  return {
    asked,
    url: `https://${FAR_HOST}:${new URL(fake.url).port}${new URL(fake.url).pathname}`,
  };
}

async function setUp(model: MockLanguageModelV4) {
  const core = startCore(await createTempDataFolder(), {
    createChatModel: scriptedModels(model).createChatModel,
    processes: testProcesses(),
  });
  await core.saveChatProvider({ kind: "ollama", modelId: "local-model" });
  const mind = await core.createMind({ title: "Tides" });
  const client = await connectToMind(core, mind.id);
  return { core, mind, client };
}

const stateOf = async (core: Core, id: string) =>
  (await core.listConnectors()).find((each: Connector) => each.id === id)?.state;

describe("A Connector on another computer in Local mode", { timeout: 30_000 }, () => {
  // protects=in Local mode a Connector on another computer is neither connected to nor offered, and no request reaches its server, though it is added then; fails_when=its Tools are offered, or the server hears anything; no_claim=none beyond protects.
  test("added in Local mode, it is paused: its Tools aren't offered and its server hears nothing", async () => {
    const fake = await startFakeRemote();
    onTestFinished(() => fake.close());
    fake.open = true;
    const far = standInForDns(fake);
    const model = usingModel(SEARCH, { query: "tides" });
    const { core, mind, client } = await setUp(model);
    await core.setLocalOnly(true);

    const added = await core.addConnector({ name: "Wiki", url: far.url });
    const { answerId } = await askAndFinish(core, client, mind.id, "What does the wiki say?");

    expect(added.state).toBe("paused");
    expect(await stateOf(core, added.id)).toBe("paused");
    expect(model.doStreamCalls[0]?.tools?.map((each) => each.name) ?? []).not.toContain(SEARCH);
    expect(answerText(client, answerId)).toBe("No such Tool.");
    expect(far.asked).toEqual([]);
    expect(fake.requests).toEqual([]);
  });

  // protects=turning Local mode off brings the Connector back without adding it again, and turning it on again stops it; fails_when=it stays paused, needs adding again, or keeps its connection in Local mode; no_claim=none beyond protects.
  test("Local mode off brings it back, and on pauses it again, without adding it again", async () => {
    const fake = await startFakeRemote();
    onTestFinished(() => fake.close());
    fake.open = true;
    const far = standInForDns(fake);
    const model = usingModel(SEARCH, { query: "tides" });
    const { core, mind, client } = await setUp(model);
    await core.setLocalOnly(true);
    const added = await core.addConnector({ name: "Wiki", url: far.url });
    expect(far.asked).toEqual([]);

    await core.setLocalOnly(false);
    await waitForState(core, added.id, "ready");
    const used = await askAndFinish(core, client, mind.id, "What does the wiki say?");
    expect(answerText(client, used.answerId)).toMatch(/^Got: /);
    expect(far.asked.length).toBeGreaterThan(0);

    await core.setLocalOnly(true);
    expect(await stateOf(core, added.id)).toBe("paused");
    const heard = fake.requests.length;
    const paused = await askAndFinish(core, client, mind.id, "And now?");
    expect(answerText(client, paused.answerId)).toBe("No such Tool.");
    expect(fake.requests).toHaveLength(heard);
    expect((await core.listConnectors()).map((each) => each.name)).toEqual(["Wiki"]);
  });

  // protects=signing in to a paused Connector is refused, since the browser and the server would be reached; fails_when=a sign-in starts, or the server hears anything; no_claim=none beyond protects.
  test("signing in to it is refused", async () => {
    const fake = await startFakeRemote();
    onTestFinished(() => fake.close());
    const far = standInForDns(fake);
    const { core } = await setUp(usingModel(SEARCH, {}));
    await core.setLocalOnly(true);
    const added = await core.addConnector({ name: "Wiki", url: far.url });

    await expect(core.signInToConnector(added.id)).rejects.toThrow(/isn't on this computer/);

    expect(far.asked).toEqual([]);
    expect(fake.requests).toEqual([]);
  });
});

describe("Connectors on this computer in Local mode", { timeout: 30_000 }, () => {
  // protects=a remote Connector at a loopback address still connects and is used in Local mode; fails_when=it is paused or not offered; no_claim=none beyond protects.
  test("a server at a loopback address keeps working", async () => {
    const fake = await startFakeRemote();
    onTestFinished(() => fake.close());
    fake.open = true;
    const model = usingModel(SEARCH, { query: "tides" });
    const { core, mind, client } = await setUp(model);
    await core.setLocalOnly(true);

    const added = await core.addConnector({ name: "Wiki", url: fake.url });
    await waitForState(core, added.id, "ready");
    const { answerId } = await askAndFinish(core, client, mind.id, "What does the wiki say?");

    expect(answerText(client, answerId)).toMatch(/^Got: /);
    expect(fake.toolCalls.map((each) => each.tool)).toEqual(["search_wiki"]);
  });

  // protects=a program on this computer keeps working in Local mode; fails_when=it is paused or not offered; no_claim=none beyond protects.
  test("a program on this computer keeps working", async () => {
    const dataDir = await createTempDataFolder();
    const model = usingModel(LOOKUP, { place: "Dover" });
    const core = startCore(dataDir, {
      createChatModel: scriptedModels(model).createChatModel,
      processes: testProcesses(),
    });
    await core.saveChatProvider({ kind: "ollama", modelId: "local-model" });
    const mind = await core.createMind({ title: "Tides" });
    const client = await connectToMind(core, mind.id);
    await core.setLocalOnly(true);

    const added = await core.addConnector(tideServer(`${dataDir}/tide.log`));
    await waitForState(core, added.id, "ready");
    const { answerId } = await askAndFinish(core, client, mind.id, "When is high tide?");

    expect(answerText(client, answerId)).toMatch(/^Got: High water at Dover/);
  });
});

describe("Why a Connector is paused, in words", () => {
  // protects=the row says why in English and Chinese, as the issue words it; fails_when=either wording changes; no_claim=none beyond protects.
  test("the state reads as the reason", () => {
    expect(translate("en", "connectors.state.paused")).toBe(
      "Not used while everything stays on this computer",
    );
    expect(translate("zh-CN", "connectors.state.paused")).toBe(
      "所有内容都留在这台电脑上时不会使用",
    );
  });
});
