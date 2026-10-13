/**
 * A local stand-in for the official MCP Registry's search
 * (GET /v0.1/servers?search=…), answering in its format as of 2026-10-13 and
 * recording each request, so a test can check what was sent and when. No
 * test reaches the real Registry. Free of the test runner, so the smoke
 * tests use it too.
 */
import { createServer } from "node:http";
import type { AddressInfo } from "node:net";

/** The Registry's servers in its format, the ones whose name or description has the words found. */
const SERVERS = [
  {
    server: {
      name: "io.github.example/zotero-mcp",
      title: "Zotero library (local)",
      description: "Reads a local Zotero database; runs a command on this computer",
      version: "0.11.0",
      repository: { url: "https://github.com/example/zotero-mcp", source: "github" },
      packages: [
        {
          registryType: "pypi",
          identifier: "zotero-mcp-server",
          version: "0.11.0",
          runtimeHint: "uvx",
          transport: { type: "stdio" },
          packageArguments: [{ type: "positional", value: "serve" }],
          environmentVariables: [{ name: "ZOTERO_API_KEY", isSecret: true }],
        },
      ],
    },
    _meta: { "io.modelcontextprotocol.registry/official": { status: "active", isLatest: true } },
  },
  {
    server: {
      name: "org.zotero.example/web",
      title: "Zotero web API",
      description: "Searches zotero.org with your API key",
      version: "1.2.0",
      websiteUrl: "https://zotero-web.example.org",
      remotes: [{ type: "streamable-http", url: "https://mcp.zotero-web.example.org/mcp" }],
    },
    _meta: { "io.modelcontextprotocol.registry/official": { status: "active", isLatest: true } },
  },
  {
    server: {
      name: "io.github.example/bibtex",
      description: "Citation keys from Better BibTeX for Zotero",
      version: "2.0.0",
      packages: [
        {
          registryType: "npm",
          identifier: "@example/bibtex-mcp",
          version: "2.0.0",
          transport: { type: "stdio" },
        },
      ],
    },
    _meta: { "io.modelcontextprotocol.registry/official": { status: "active", isLatest: true } },
  },
  {
    server: {
      name: "io.github.example/old-zotero",
      description: "An old Zotero server, deleted from the Registry",
      version: "0.1.0",
    },
    _meta: { "io.modelcontextprotocol.registry/official": { status: "deleted" } },
  },
];

export interface FakeRegistry {
  /** e.g. "http://127.0.0.1:51234": `CoreAdapters.connectorRegistryUrl`. */
  readonly url: string;
  /** Each request's path and query, as sent. */
  readonly requests: string[];
  /** Answer each search with this HTTP status instead, e.g. 500. */
  failWith: number | null;
  close(): Promise<void>;
}

export async function startFakeRegistry(): Promise<FakeRegistry> {
  const requests: string[] = [];
  const state: { failWith: number | null } = { failWith: null };
  const server = createServer((request, response) => {
    const url = new URL(request.url ?? "/", "http://registry.test");
    requests.push(`${url.pathname}${url.search}`);
    if (state.failWith !== null) {
      response.writeHead(state.failWith).end();
      return;
    }
    if (url.pathname !== "/v0.1/servers") {
      response.writeHead(404).end();
      return;
    }
    const words = (url.searchParams.get("search") ?? "").toLowerCase();
    const found = SERVERS.filter(({ server }) =>
      `${server.name} ${server.description}`.toLowerCase().includes(words),
    );
    response.writeHead(200, { "content-type": "application/json" });
    response.end(JSON.stringify({ servers: found, metadata: { count: found.length } }));
  });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  const { port } = server.address() as AddressInfo;
  return {
    url: `http://127.0.0.1:${port}`,
    requests,
    get failWith() {
      return state.failWith;
    },
    set failWith(status: number | null) {
      state.failWith = status;
    },
    close: () =>
      new Promise<void>((resolve) => {
        server.closeAllConnections();
        server.close(() => resolve());
      }),
  };
}
