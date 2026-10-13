import { describe, expect, onTestFinished, test } from "vitest";
import type { Core } from "../../src/core";
import { createTempDataFolder, startCore } from "../helpers/core";
import { startFakeRegistry } from "../helpers/fakeRegistry";

/** A core whose Connectors search asks a local stand-in for the MCP Registry. */
async function setUp() {
  const registry = await startFakeRegistry();
  onTestFinished(() => registry.close());
  const core: Core = startCore(await createTempDataFolder(), {
    connectorRegistryUrl: registry.url,
  });
  return { core, registry };
}

/** Answers every consent request as `accept`, counting them. */
function answerConsent(core: Core, accept: boolean) {
  const asked: string[] = [];
  core.on("consent.requested", (request) => {
    asked.push(`${request.flow.id} → ${request.flow.service.name}: ${request.newKinds.join(", ")}`);
    void core.respondToConsent(request.requestId, accept);
  });
  return asked;
}

describe("Searching the MCP Registry", () => {
  test("sends nothing until the User searches, and nothing for one letter", async () => {
    const { core, registry } = await setUp();
    const asked = answerConsent(core, true);

    await core.listConnectorDirectory();
    expect(await core.searchConnectorRegistry("z")).toEqual({ outcome: "found", servers: [] });
    expect(await core.searchConnectorRegistry("  ")).toEqual({ outcome: "found", servers: [] });

    expect(registry.requests).toEqual([]);
    expect(asked).toEqual([]);
  });

  test("asks the User once, then sends only the words, and reads each community server with how it runs", async () => {
    const { core, registry } = await setUp();
    const asked = answerConsent(core, true);

    const search = await core.searchConnectorRegistry("  zotero ");

    expect(asked).toEqual(["connector-registry → MCP Registry: connector-search"]);
    expect(registry.requests).toEqual(["/v0.1/servers?search=zotero&version=latest&limit=20"]);
    expect(search).toEqual({
      outcome: "found",
      servers: [
        {
          id: "io.github.example/zotero-mcp",
          name: "Zotero library (local)",
          publisher: "io.github.example",
          description: "Reads a local Zotero database; runs a command on this computer",
          url: null,
          command: {
            command: "uvx",
            args: ["zotero-mcp-server@0.11.0", "serve"],
            env: ["ZOTERO_API_KEY"],
          },
          link: "https://github.com/example/zotero-mcp",
        },
        {
          id: "org.zotero.example/web",
          name: "Zotero web API",
          publisher: "org.zotero.example",
          description: "Searches zotero.org with your API key",
          url: "https://mcp.zotero-web.example.org/mcp",
          command: null,
          link: "https://zotero-web.example.org",
        },
        {
          id: "io.github.example/bibtex",
          name: "bibtex",
          publisher: "io.github.example",
          description: "Citation keys from Better BibTeX for Zotero",
          url: null,
          command: { command: "npx", args: ["-y", "@example/bibtex-mcp@2.0.0"], env: [] },
          link: null,
        },
      ],
    });
    // Asked once: the next search goes straight to the Registry.
    await core.searchConnectorRegistry("bibtex");
    expect(asked).toHaveLength(1);
    expect(registry.requests).toHaveLength(2);
  });

  test("declined, nothing is sent, and the search says why", async () => {
    const { core, registry } = await setUp();
    answerConsent(core, false);

    expect(await core.searchConnectorRegistry("zotero")).toEqual({
      outcome: "failed",
      reason: "declined",
    });
    expect(registry.requests).toEqual([]);
  });

  test("in Local mode, nothing is sent or asked", async () => {
    const { core, registry } = await setUp();
    const asked = answerConsent(core, true);
    await core.setLocalOnly(true);

    expect(await core.searchConnectorRegistry("zotero")).toEqual({
      outcome: "failed",
      reason: "local-mode",
    });
    expect(asked).toEqual([]);
    expect(registry.requests).toEqual([]);
  });

  test("a Registry that fails or can't be reached says so", async () => {
    const { core, registry } = await setUp();
    answerConsent(core, true);

    registry.failWith = 503;
    expect(await core.searchConnectorRegistry("zotero")).toEqual({
      outcome: "failed",
      reason: "service-error",
    });
    await registry.close();
    expect(await core.searchConnectorRegistry("zotero")).toEqual({
      outcome: "failed",
      reason: "unreachable",
    });
  });
});
