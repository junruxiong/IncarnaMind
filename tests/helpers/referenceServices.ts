/**
 * A local stand-in for Look up's two services (#286): Crossref's
 * `GET /works/{doi}` and arXiv's `GET /api/query?id_list=…`, answering from
 * recorded answers (tests/fixtures/references/), and, for checking reference
 * lists (#711), Crossref's `GET /works?query.bibliographic=…` over a few
 * works, best match first by the words they share. It records every request, so
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

/** A work as Crossref's JSON gives it, for the stand-in's searches and DOIs. */
export interface CrossrefWork {
  DOI: string;
  title: string[];
  author: { family: string; given: string }[];
  issued: { "date-parts": number[][] };
  "container-title"?: string[];
  type?: string;
}

const wordsOf = (text: string) => new Set(text.toLowerCase().match(/[\p{L}\p{N}]+/gu) ?? []);

/** The works whose titles share the most words with `query`, at most five, as Crossref answers a search. */
function searched(works: readonly CrossrefWork[], query: string): string {
  const asked = wordsOf(query);
  const items = works
    .map((work) => {
      let shared = 0;
      for (const word of wordsOf(work.title[0] ?? "")) if (asked.has(word)) shared++;
      return { work, shared };
    })
    .filter(({ shared }) => shared > 0)
    .sort((a, b) => b.shared - a.shared)
    .slice(0, 5)
    .map(({ work, shared }) => ({ ...work, score: shared }));
  return JSON.stringify({ status: "ok", message: { items } });
}

/**
 * Starts the stand-in. `crossref` maps DOIs to recorded Crossref answers (as
 * JSON text); `arxiv` maps arXiv ids to recorded Atom feeds; `works` are
 * searched, and answer their DOIs too. Anything else is a 404, as the real
 * services answer an unknown DOI or id.
 */
export async function startReferenceServices(answers: {
  crossref?: Readonly<Record<string, string>>;
  arxiv?: Readonly<Record<string, string>>;
  works?: readonly CrossrefWork[];
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
      const work =
        doi === null
          ? undefined
          : answers.works?.find((each) => each.DOI.toLowerCase() === doi.toLowerCase());
      const crossref =
        doi === null
          ? undefined
          : (answers.crossref?.[doi] ?? (work && JSON.stringify({ status: "ok", message: work })));
      const arxiv = id === null ? undefined : answers.arxiv?.[id];
      const query = url.pathname === "/works" ? url.searchParams.get("query.bibliographic") : null;
      if (query !== null) {
        response
          .writeHead(200, { "content-type": "application/json" })
          .end(searched(answers.works ?? [], query));
      } else if (crossref !== undefined) {
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
