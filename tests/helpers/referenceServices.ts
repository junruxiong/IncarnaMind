/**
 * A local stand-in for Look up's two services (#286): Crossref's
 * `GET /works/{doi}` and arXiv's `GET /api/query?id_list=…`, answering from
 * recorded answers (tests/fixtures/references/). It records every request, so
 * a test can check that nothing was sent, or only the identifier. No test
 * reaches the real services. Free of the test runner, so the smoke tests use
 * it too.
 */
import { readFileSync } from "node:fs";
import { createServer } from "node:http";
import type { AddressInfo } from "node:net";
import { join } from "node:path";

/** One request the stand-in received. */
export interface ServiceRequest {
  method: string;
  /** The path and query, as sent. */
  path: string;
  /** How many bytes of body came with it. */
  bodyLength: number;
}

export interface ReferenceServices {
  /** The base URL of both services, e.g. "http://127.0.0.1:51234". */
  readonly url: string;
  /** Every request, in order. */
  readonly requests: ServiceRequest[];
  /** Answers every request with this status from now on (null: answer again). */
  fail(status: number | null): void;
  close(): Promise<void>;
}

/** A recorded answer, from tests/fixtures/references. */
export const recordedAnswer = (name: string) =>
  readFileSync(join(__dirname, "../fixtures/references", name), "utf8");

/**
 * Starts the stand-in. `crossref` maps DOIs to recorded Crossref answers (as
 * JSON text); `arxiv` maps arXiv ids to recorded Atom feeds. Anything else
 * is a 404, as the real services answer an unknown DOI or id.
 */
export async function startReferenceServices(answers: {
  crossref?: Readonly<Record<string, string>>;
  arxiv?: Readonly<Record<string, string>>;
}): Promise<ReferenceServices> {
  const requests: ServiceRequest[] = [];
  let failure: number | null = null;
  const server = createServer((request, response) => {
    let bodyLength = 0;
    request.on("data", (chunk: Buffer) => {
      bodyLength += chunk.length;
    });
    request.on("end", () => {
      const path = request.url ?? "/";
      requests.push({ method: request.method ?? "", path, bodyLength });
      if (failure !== null) {
        response.writeHead(failure).end();
        return;
      }
      const url = new URL(path, "http://localhost");
      const doi = url.pathname.startsWith("/works/")
        ? decodeURIComponent(url.pathname.slice("/works/".length))
        : null;
      const id = url.pathname === "/api/query" ? url.searchParams.get("id_list") : null;
      const crossref = doi === null ? undefined : answers.crossref?.[doi];
      const arxiv = id === null ? undefined : answers.arxiv?.[id];
      if (crossref !== undefined) {
        response.writeHead(200, { "content-type": "application/json" }).end(crossref);
      } else if (arxiv !== undefined) {
        response.writeHead(200, { "content-type": "application/atom+xml" }).end(arxiv);
      } else {
        response.writeHead(404, { "content-type": "text/plain" }).end("Resource not found.");
      }
    });
  });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  const { port } = server.address() as AddressInfo;
  return {
    url: `http://127.0.0.1:${port}`,
    requests,
    fail(status) {
      failure = status;
    },
    close: () => new Promise<void>((resolve) => server.close(() => resolve())),
  };
}
