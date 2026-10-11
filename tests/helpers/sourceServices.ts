/**
 * A local stand-in for Find sources' providers (#296), answering from
 * answers recorded from the real services on 2026-10-11
 * (tests/fixtures/sources/), trimmed: OpenAlex at /openalex, arXiv (its API
 * and its PDFs) at /arxiv, NCBI's E-utilities at /pubmed and Crossref at
 * /crossref. Open copies a recorded answer points to (a publisher's PDF or
 * web page) are served at /files; "paywall.pdf" answers with a sign-in page,
 * as a paywall does. It records every request, so a test can check what was sent, and to
 * whom. No test reaches the real services. Free of the test runner, so the
 * smoke tests use it too.
 */
import { readFileSync } from "node:fs";
import { createServer } from "node:http";
import type { AddressInfo } from "node:net";
import { join } from "node:path";
import { buildPdf } from "./pdf";

export interface SourceRequest {
  method: string;
  /** The path and query, as sent. */
  path: string;
  userAgent: string;
  bodyLength: number;
}

export interface SourceServices {
  /** e.g. "http://127.0.0.1:51234". */
  readonly url: string;
  /** Where each provider is, for `CoreAdapters.sourceProviders`. */
  readonly providers: { openalex: string; arxiv: string; pubmed: string; crossref: string };
  /** Every request, in order. */
  readonly requests: SourceRequest[];
  /** Answers every request to this provider ("openalex", "arxiv", "pubmed", "crossref", "files") with this status from now on; null answers again. */
  fail(provider: string, status: number | null): void;
  close(): Promise<void>;
}

const fixture = (name: string) =>
  readFileSync(join(__dirname, "../fixtures/sources", name), "utf8");

/** A paper's PDF, with words a Question can find and an Answer can quote. */
export function paperPdf(title: string, lines: readonly string[]): Uint8Array {
  return buildPdf([{ lines: [title, "", ...lines] }], { info: { Title: title } });
}

/** The open copies the stand-in serves, by file name. */
export const OPEN_COPIES: Readonly<Record<string, Uint8Array>> = {
  "nature-methods.pdf": paperPdf(
    "Improving the sensitivity of in vivo CRISPR off-target detection",
    [
      "DISCOVER-Seq+ combines a DNA-PKcs inhibitor with DISCOVER-Seq.",
      "It finds up to fivefold more off-target sites in primary cells than DISCOVER-Seq alone.",
    ],
  ),
  "biorxiv-preprint.pdf": paperPdf(
    "Improving the sensitivity of in vivo CRISPR off-target detection",
    ["This is the preprint of the paper."],
  ),
};

/** The open copies that are a publisher's web page, by file name. */
const OPEN_PAGES: Readonly<Record<string, string>> = {
  "chopoff.html": `<!doctype html>
<html lang="en">
<head>
<title>CHOPOFF: symbolic alignments enable fast and sensitive CRISPR off-target detection | bioRxiv</title>
<meta property="og:site_name" content="bioRxiv">
</head>
<body>
<nav><a href="/">Home</a> <a href="/about">About</a> <a href="/submit">Submit a manuscript</a></nav>
<article>
<h1>CHOPOFF: symbolic alignments enable fast and sensitive CRISPR off-target detection</h1>
<p>CHOPOFF searches a genome for the off-target sites of a CRISPR guide with symbolic alignments, which allow mismatches, insertions and deletions at once.</p>
<p>It builds an index of the genome once, then answers each guide from that index, so a search over a whole genome with many guides takes minutes rather than days.</p>
<p>Against other tools, CHOPOFF finds the sites they find and more of those with several mismatches and a bulge, which matter when editing cells from patients.</p>
</article>
</body>
</html>
`,
};

/** arXiv's PDFs, by id with its version. */
const ARXIV_PDFS: Readonly<Record<string, Uint8Array>> = {
  "2602.16327v1": paperPdf("Guide-Guard: Off-Target Predicting in CRISPR Applications", [
    "Guide-Guard predicts the behaviour of the system given a gRNA with 84% accuracy.",
    "It can be trained on several genes at the same time and keep its accuracy.",
  ]),
};

/** What each PMC id's EFetch gives: the full text, or (a publisher that doesn't allow it) none. */
const PMC: Readonly<Record<string, string>> = {
  PMC10034092: "pmc-efetch.xml",
  PMC10257351: "pmc-no-body.xml",
};

const EMPTY_FEED =
  '<?xml version="1.0" encoding="UTF-8"?><feed xmlns="http://www.w3.org/2005/Atom"></feed>';

export async function startSourceServices(): Promise<SourceServices> {
  const requests: SourceRequest[] = [];
  const failures = new Map<string, number>();
  let base = "";
  const send = (
    response: import("node:http").ServerResponse,
    status: number,
    type: string,
    body: string | Uint8Array,
  ) => {
    response
      .writeHead(status, { "content-type": type })
      .end(typeof body === "string" ? body.replaceAll("http://STANDIN", base) : body);
  };
  const server = createServer((request, response) => {
    let bodyLength = 0;
    request.on("data", (chunk: Buffer) => {
      bodyLength += chunk.length;
    });
    request.on("end", () => {
      const path = request.url ?? "/";
      requests.push({
        method: request.method ?? "",
        path,
        userAgent: request.headers["user-agent"] ?? "",
        bodyLength,
      });
      const url = new URL(path, "http://localhost");
      const [, provider = "", ...rest] = url.pathname.split("/");
      const failure = failures.get(provider);
      if (failure !== undefined) {
        response.writeHead(failure, { "retry-after": "120" }).end();
        return;
      }
      const tail = decodeURIComponent(rest.join("/"));
      const notFound = () => send(response, 404, "text/plain", "Not found.");
      switch (provider) {
        case "openalex": {
          if (tail === "works")
            return send(response, 200, "application/json", fixture("openalex-search.json"));
          if (tail === "works/doi:10.3389/fbioe.2023.1143157") {
            return send(response, 200, "application/json", fixture("openalex-work.json"));
          }
          return notFound();
        }
        case "arxiv": {
          if (tail === "api/query") {
            const id = url.searchParams.get("id_list");
            if (!id)
              return send(response, 200, "application/atom+xml", fixture("arxiv-search.xml"));
            const feed = fixture("arxiv-search.xml");
            return send(
              response,
              200,
              "application/atom+xml",
              feed.includes(`abs/${id}`) ? feed : EMPTY_FEED,
            );
          }
          const pdf = ARXIV_PDFS[tail.replace(/^pdf\//, "")];
          return pdf ? send(response, 200, "application/pdf", pdf) : notFound();
        }
        case "pubmed": {
          if (tail === "entrez/eutils/esearch.fcgi") {
            return send(response, 200, "application/json", fixture("pubmed-esearch.json"));
          }
          if (tail === "entrez/eutils/esummary.fcgi") {
            return send(response, 200, "application/json", fixture("pubmed-esummary.json"));
          }
          if (tail === "entrez/eutils/efetch.fcgi") {
            const file = PMC[url.searchParams.get("id") ?? ""];
            return file ? send(response, 200, "text/xml", fixture(file)) : notFound();
          }
          return notFound();
        }
        case "crossref": {
          return tail === "works/10.3389/fbioe.2023.1143157"
            ? send(response, 200, "application/json", fixture("crossref-frontiers.json"))
            : notFound();
        }
        case "files": {
          if (tail === "paywall.pdf") {
            return send(
              response,
              200,
              "text/html",
              "<html><body><form>Sign in to read</form></body></html>",
            );
          }
          const page = OPEN_PAGES[tail];
          if (page) return send(response, 200, "text/html; charset=utf-8", page);
          const copy = OPEN_COPIES[tail];
          return copy ? send(response, 200, "application/pdf", copy) : notFound();
        }
        default:
          return notFound();
      }
    });
  });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  const { port } = server.address() as AddressInfo;
  base = `http://127.0.0.1:${port}`;
  return {
    url: base,
    providers: {
      openalex: `${base}/openalex`,
      arxiv: `${base}/arxiv`,
      pubmed: `${base}/pubmed`,
      crossref: `${base}/crossref`,
    },
    requests,
    fail(provider, status) {
      if (status === null) failures.delete(provider);
      else failures.set(provider, status);
    },
    close: () =>
      new Promise<void>((resolve, reject) => {
        server.closeAllConnections();
        server.close((error) => (error ? reject(error) : resolve()));
      }),
  };
}
