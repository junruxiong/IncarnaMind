import { createServer, type IncomingMessage } from "node:http";
import type { AddressInfo } from "node:net";
import { onTestFinished } from "vitest";

export interface OllamaStub {
  baseUrl: string;
  /** Bodies of every POST /api/pull, parsed. */
  pulls: unknown[];
}

const readBody = (request: IncomingMessage) =>
  new Promise<string>((resolve, reject) => {
    let body = "";
    request.on("data", (chunk) => {
      body += chunk;
    });
    request.on("end", () => resolve(body));
    request.on("error", reject);
  });

/**
 * A tiny local HTTP server that speaks just enough of Ollama's API:
 * `/api/tags` lists `models`, and `/api/pull` streams `pullLines` as
 * newline-delimited JSON, a few milliseconds apart. Stopped when the test finishes.
 */
export async function startOllamaStub(
  options: { models?: string[]; pullLines?: object[] } = {},
): Promise<OllamaStub> {
  const { models = [], pullLines = [] } = options;
  const pulls: unknown[] = [];
  const server = createServer(async (request, response) => {
    if (request.method === "GET" && request.url === "/api/tags") {
      response.writeHead(200, { "content-type": "application/json" });
      response.end(JSON.stringify({ models: models.map((name) => ({ name, model: name })) }));
      return;
    }
    if (request.method === "POST" && request.url === "/api/pull") {
      pulls.push(JSON.parse(await readBody(request)));
      response.writeHead(200, { "content-type": "application/x-ndjson" });
      for (const line of pullLines) {
        response.write(`${JSON.stringify(line)}\n`);
        await new Promise((resolve) => setTimeout(resolve, 2));
      }
      response.end();
      return;
    }
    response.writeHead(404).end();
  });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  onTestFinished(() => new Promise<void>((resolve) => server.close(() => resolve())));
  const { port } = server.address() as AddressInfo;
  return { baseUrl: `http://127.0.0.1:${port}`, pulls };
}

/** A URL where nothing is listening. */
export async function unusedLocalUrl(): Promise<string> {
  const server = createServer();
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  const { port } = server.address() as AddressInfo;
  await new Promise<void>((resolve) => server.close(() => resolve()));
  return `http://127.0.0.1:${port}`;
}
