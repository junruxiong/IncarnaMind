import { readdir, readFile } from "node:fs/promises";
import { join } from "node:path";
import { describe, expect, test } from "vitest";
import { ConsentDeclinedError, DATABASE_FILE, InvalidInputError } from "../../src/core";
import {
  askAndFinish,
  citingModel,
  type ShownPassage,
  setUpWithDocuments,
} from "../helpers/citations";
import { createMemoryKeychain, nextEvent } from "../helpers/core";
import { mockRerankingModels } from "../helpers/embeddingProviders";

const COHERE = { id: "https://api.cohere.com", name: "Cohere" };
const VOYAGE = { id: "https://api.voyageai.com", name: "Voyage AI" };
const COHERE_KEY = "cohere-test-key";

/** A Markdown Document of `count` long sections, the word "lighthouse" only in `marked`. */
function sections(count: number, marked: number, topic: string): string {
  return Array.from({ length: count }, (_, index) => {
    const number = index + 1;
    const special = number === marked ? ` The ${topic} lighthouse stands here.` : "";
    const filler = Array.from(
      { length: 18 },
      (_, line) => `Section ${number} sentence ${line} talks about ordinary coastal weather.`,
    ).join(" ");
    return `## Section ${number}\n\n${filler}${special}`;
  }).join("\n\n");
}

/** A reranking model that prefers the Weather Document's Passages, whatever search found. */
const prefersWeather = (text: string) => (text.startsWith("Weather\n") ? 0.9 : 0.1);

/**
 * Two Documents, a local chat model that searches for "lighthouse" and shows
 * what it was given, and mock reranking models.
 */
async function setUp() {
  let shown: ShownPassage[] = [];
  const rerankers = mockRerankingModels(prefersWeather);
  const keychain = createMemoryKeychain();
  const model = citingModel({
    query: "lighthouse",
    records: (passages) => {
      shown = passages;
      return [];
    },
    answer: "There is a lighthouse.",
  });
  const setup = await setUpWithDocuments(
    model,
    [
      { name: "Coast.md", contents: sections(24, 12, "red") },
      { name: "Weather.md", contents: sections(6, 0, "") },
    ],
    { createRerankingModel: rerankers.createRerankingModel, keychain },
  );
  const ask = async (text = "Where is the lighthouse?") => {
    await askAndFinish(setup.core, setup.client, setup.mind.id, text);
    return shown;
  };
  return { ...setup, rerankers, keychain, ask };
}

/** Sets Cohere up, accepting its flow. */
async function useCohere(core: Awaited<ReturnType<typeof setUp>>["core"]) {
  const accept = core.on(
    "consent.requested",
    (request) => void core.respondToConsent(request.requestId, true),
  );
  const saved = await core.saveRerankSettings({ kind: "cohere", apiKey: COHERE_KEY });
  accept();
  return saved;
}

describe("Rerank", { timeout: 30_000 }, () => {
  test("without a key nothing changes: search keeps its order, and nothing is sent", async () => {
    const { core, rerankers, ask } = await setUp();

    expect(await core.getRerankSettings()).toEqual({
      enabled: false,
      kind: null,
      modelId: null,
      hasApiKey: false,
      service: null,
      paused: false,
    });
    const shown = await ask();

    expect(shown[0]?.document).toBe("Coast");
    expect(rerankers.calls).toEqual([]);
    expect((await core.listDataFlows()).some((flow) => flow.flow.id === "rerank")).toBe(false);
  });

  test("with a Cohere key, document search reranks its candidates: the order changes", async () => {
    const { core, rerankers, ask } = await setUp();
    const changed = nextEvent(core, "rerank.changed");

    const saved = await useCohere(core);

    const expected = {
      enabled: true,
      kind: "cohere",
      modelId: "rerank-v3.5",
      hasApiKey: true,
      service: COHERE,
      paused: false,
    };
    expect(saved).toEqual(expected);
    expect(await changed).toEqual(expected);
    const shown = await ask();

    // The reranking model prefers Weather, so its Passages now come first.
    expect(shown[0]?.document).toBe("Weather");
    expect(rerankers.calls).toHaveLength(1);
    const call = rerankers.calls[0];
    expect(call?.spec).toEqual({ kind: "cohere", apiKey: COHERE_KEY, modelId: "rerank-v3.5" });
    expect(call?.query).toBe("lighthouse");
    // The candidates, each with its Document's name.
    expect(call?.documents.length).toBeGreaterThan(1);
    expect(call?.documents.some((text) => text.startsWith("Coast\n"))).toBe(true);
    expect(call?.documents.some((text) => text.startsWith("Weather\n"))).toBe(true);
  });

  test("Voyage AI works the same way, with its own default model, and the model can be named", async () => {
    const { core, rerankers, ask } = await setUp();
    core.on("consent.requested", (request) => void core.respondToConsent(request.requestId, true));

    expect(await core.saveRerankSettings({ kind: "voyage", apiKey: "voyage-key" })).toMatchObject({
      kind: "voyage",
      modelId: "rerank-2.5",
      service: VOYAGE,
    });
    await ask();
    expect(rerankers.calls[0]?.spec).toEqual({
      kind: "voyage",
      apiKey: "voyage-key",
      modelId: "rerank-2.5",
    });

    expect(
      await core.saveRerankSettings({ kind: "voyage", modelId: "rerank-2.5-lite" }),
    ).toMatchObject({
      modelId: "rerank-2.5-lite",
      hasApiKey: true,
    });
    // Another provider needs its own key.
    await expect(core.saveRerankSettings({ kind: "cohere" })).rejects.toThrow(InvalidInputError);
  });

  test("a failing reranking request keeps search's order", async () => {
    const { core, rerankers, ask } = await setUp();
    await useCohere(core);
    rerankers.fail({ status: 500, message: "Internal server error." });

    const shown = await ask();

    expect(rerankers.calls.length).toBeGreaterThan(0);
    expect(shown[0]?.document).toBe("Coast");
  });

  test("removing the key goes back to search's own order", async () => {
    const { core, rerankers, keychain, ask } = await setUp();
    await useCohere(core);

    expect(await core.removeRerankSettings()).toMatchObject({ enabled: false, hasApiKey: false });
    expect(keychain.secrets.has("rerank:api-key")).toBe(false);
    const shown = await ask();

    expect(shown[0]?.document).toBe("Coast");
    expect(rerankers.calls).toEqual([]);
  });

  test("local mode pauses it: nothing is sent, and it can't be set up meanwhile", async () => {
    const { core, rerankers, ask } = await setUp();
    await useCohere(core);
    const paused = nextEvent(core, "rerank.changed");

    await core.setLocalOnly(true);

    expect(await paused).toMatchObject({ enabled: true, paused: true });
    const shown = await ask();
    expect(shown[0]?.document).toBe("Coast");
    expect(rerankers.calls).toEqual([]);
    await expect(core.saveRerankSettings({ kind: "voyage", apiKey: "voyage-key" })).rejects.toThrow(
      /Local mode/,
    );
    await expect(core.testRerankConnection()).rejects.toThrow(/Local mode/);

    await core.setLocalOnly(false);
    expect((await ask())[0]?.document).toBe("Weather");
  });

  test("its key goes to the keychain, never into SQLite", async () => {
    const { core, keychain, dataDir } = await setUp();
    await useCohere(core);

    expect(keychain.secrets.get("rerank:api-key")).toBe(COHERE_KEY);
    const databaseFiles = (await readdir(dataDir)).filter((name) => name.startsWith(DATABASE_FILE));
    expect(databaseFiles).toContain(DATABASE_FILE);
    for (const name of databaseFiles) {
      expect((await readFile(join(dataDir, name))).includes(COHERE_KEY)).toBe(false);
    }
  });
});

describe("Consent for rerank", { timeout: 30_000 }, () => {
  test("nothing is sent before the User accepts; declining sets nothing up", async () => {
    const { core, rerankers, keychain, ask } = await setUp();
    const requested = nextEvent(core, "consent.requested");

    const saving = core.saveRerankSettings({ kind: "cohere", apiKey: COHERE_KEY });
    const request = await requested;
    expect(request.flow).toEqual({ id: "rerank", service: COHERE, sends: ["queries", "passages"] });
    await core.respondToConsent(request.requestId, false);

    await expect(saving).rejects.toThrow(ConsentDeclinedError);
    expect((await core.getRerankSettings()).enabled).toBe(false);
    expect(keychain.secrets.has("rerank:api-key")).toBe(false);
    expect((await ask())[0]?.document).toBe("Coast");
    expect(rerankers.calls).toEqual([]);
  });

  test("revoked later, a search asks again; declining keeps search's order and sends nothing", async () => {
    const { core, rerankers, ask } = await setUp();
    await useCohere(core);
    await core.revokeConsent("rerank", COHERE.id);
    const requests: string[] = [];
    core.on("consent.requested", (request) => {
      requests.push(request.flow.id);
      void core.respondToConsent(request.requestId, false);
    });

    const shown = await ask();

    expect(requests).toEqual(["rerank"]);
    expect(shown[0]?.document).toBe("Coast");
    expect(rerankers.calls).toEqual([]);
  });
});

describe("Testing the rerank connection", { timeout: 30_000 }, () => {
  test("asks first, sends nothing of the User's, and saves nothing", async () => {
    const { core, rerankers } = await setUp();
    const requested = nextEvent(core, "consent.requested");
    const testing = core.testRerankConnection({ kind: "cohere", apiKey: COHERE_KEY });
    const request = await requested;
    expect(rerankers.calls).toEqual([]);
    await core.respondToConsent(request.requestId, true);

    expect(await testing).toEqual({ ok: true });
    expect(rerankers.calls).toHaveLength(1);
    expect(rerankers.calls[0]?.query).toContain("capital of France");
    expect((await core.getRerankSettings()).enabled).toBe(false);
  });

  test("errors are classified: a bad key, a rate limit, no connection", async () => {
    const { core, rerankers } = await setUp();
    await useCohere(core);

    rerankers.fail({ status: 401, message: "invalid api token" });
    expect(await core.testRerankConnection()).toEqual({
      ok: false,
      error: { kind: "auth", message: "invalid api token" },
    });
    rerankers.fail({ status: 429, message: "Too many requests." });
    expect(await core.testRerankConnection()).toMatchObject({
      ok: false,
      error: { kind: "rate-limit" },
    });
    rerankers.fail({ message: "Cannot connect to API." });
    expect(await core.testRerankConnection()).toMatchObject({
      ok: false,
      error: { kind: "network" },
    });
    // With no key given or saved for the provider, it refuses.
    await expect(core.testRerankConnection({ kind: "voyage" })).rejects.toThrow(InvalidInputError);
  });
});
