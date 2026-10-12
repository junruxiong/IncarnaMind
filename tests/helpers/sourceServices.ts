/**
 * A local stand-in for Find sources' providers (#296), answering from
 * answers recorded from the real services on 2026-10-11
 * (tests/fixtures/sources/), trimmed: OpenAlex at /openalex, arXiv (its API
 * and its PDFs) at /arxiv, NCBI's E-utilities at /pubmed, Crossref at
 * /crossref, openFDA's drug labels at /openfda, DailyMed at /dailymed,
 * ClinicalTrials.gov at /clinicaltrials, and (recorded on 2026-10-12) the
 * eCFR at /ecfr and GovInfo's API and link service at /govinfo. Open copies a recorded answer points to (a publisher's PDF or
 * web page) are served at /files; "paywall.pdf" answers with a sign-in page,
 * as a paywall does. It records every request, so a test can check what was sent, and to
 * whom. No test reaches the real services. Free of the test runner, so the
 * smoke tests use it too.
 */
import { readFileSync } from "node:fs";
import { createServer } from "node:http";
import type { AddressInfo } from "node:net";
import { join } from "node:path";
import { gzipSync } from "node:zlib";
import { buildPdf } from "./pdf";

export interface SourceRequest {
  method: string;
  /** The path and query, as sent. */
  path: string;
  userAgent: string;
  /** As sent: a test checks that a key goes in a header and never in the address. */
  headers: Record<string, string | string[] | undefined>;
  bodyLength: number;
}

export interface SourceServices {
  /** e.g. "http://127.0.0.1:51234". */
  readonly url: string;
  /** Where each provider is, for `CoreAdapters.sourceProviders`. */
  readonly providers: {
    openalex: string;
    arxiv: string;
    pubmed: string;
    crossref: string;
    openfda: string;
    dailymed: string;
    clinicaltrials: string;
    ecfr: string;
    govinfo: string;
  };
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

/** DailyMed's labels, by set id. */
const DAILYMED_LABELS: Readonly<Record<string, string>> = {
  "e087ce3b-8d7f-43a3-8d20-54513da7617d": "dailymed-label-metformin.xml",
  "0058175f-3474-40c3-a046-6cfaec86d84b": "dailymed-label-aspirin.xml",
};

/** ClinicalTrials.gov's study records, by NCT number. */
const CLINICAL_TRIALS: Readonly<Record<string, string>> = {
  NCT04368728: "clinicaltrials-study-results.json",
  NCT04625985: "clinicaltrials-study-no-results.json",
};

/** The eCFR's versioner, by what is asked of title 17: where each section or part is, its texts by date ("*": any). */
const ECFR_ANCESTRY: Readonly<Record<string, string>> = {
  "240.10b-5": "ecfr-ancestry-240.10b-5.json",
  "part 243": "ecfr-ancestry-part-243.json",
};
const ECFR_TEXTS: Readonly<Record<string, string>> = {
  "240.10b-5@*": "ecfr-240.10b-5.xml",
  "part 243@*": "ecfr-part-243.xml",
  "240.15Aj-1@2025-03-23": "ecfr-240.15Aj-1-2025-03-23.xml",
};
const ECFR_VERSIONS: Readonly<Record<string, string>> = {
  "240.15Aj-1": "ecfr-versions-240.15Aj-1.json",
};

/** GovInfo's link service: where each citation it knows redirects (the CFR's by part and section). */
const GOVINFO_LINKS: Readonly<Record<string, string>> = {
  "uscode/42/1983":
    "https://www.govinfo.gov/content/pkg/USCODE-2024-title42/html/USCODE-2024-title42-chap21-subchapI-sec1983.htm",
  "uscode/15/78j":
    "https://www.govinfo.gov/content/pkg/USCODE-2024-title15/html/USCODE-2024-title15-chap2B-sec78j.htm",
  "cfr/17/240/10b-5":
    "https://www.govinfo.gov/content/pkg/CFR-2025-title17-vol4/xml/CFR-2025-title17-vol4-sec240-10b-5.xml",
};

/** GovInfo's granules: their summary and their text. */
const GOVINFO_GRANULES: Readonly<Record<string, { summary: string; text: string }>> = {
  "USCODE-2024-title42-chap21-subchapI-sec1983": {
    summary: "govinfo-usc-1983-summary.json",
    text: "govinfo-usc-1983.htm",
  },
  "USCODE-2024-title15-chap2B-sec78j": {
    summary: "govinfo-usc-78j-summary.json",
    text: "govinfo-usc-78j.htm",
  },
  "CFR-2025-title17-vol4-sec240-10b-5": {
    summary: "govinfo-cfr-240.10b-5-summary.json",
    text: "govinfo-cfr-240.10b-5.xml",
  },
};

/** A GovInfo key the stand-in turns away, as GovInfo turns away one it doesn't know. */
export const REJECTED_GOVINFO_KEY = "rejected-govinfo-key";

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
        headers: request.headers,
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
        case "openfda": {
          if (tail !== "drug/label.json") return notFound();
          const search = url.searchParams.get("search") ?? "";
          const answers = ["openfda-label-metformin.json", "openfda-label-aspirin.json"].map(
            (file) =>
              JSON.parse(fixture(file)) as {
                meta: unknown;
                results: { set_id: string; openfda: { generic_name?: string[] } }[];
              },
          );
          const setId = /set_id:"([^"]+)"/.exec(search)?.[1];
          const word = /aspirin/i.test(search)
            ? "aspirin"
            : /metformin/i.test(search)
              ? "metformin"
              : "";
          const results = answers.flatMap((answer) =>
            answer.results.filter((label) =>
              setId
                ? label.set_id === setId
                : word !== "" &&
                  label.openfda.generic_name?.some((name) => name.toLowerCase().includes(word)),
            ),
          );
          if (results.length === 0) {
            return send(
              response,
              404,
              "application/json",
              '{"error":{"code":"NOT_FOUND","message":"No matches found!"}}',
            );
          }
          return send(
            response,
            200,
            "application/json",
            JSON.stringify({ meta: answers[0]?.meta, results }),
          );
        }
        case "dailymed": {
          if (tail === "spls.json") {
            const setId = url.searchParams.get("setid");
            const name = (url.searchParams.get("drug_name") ?? "").toLowerCase();
            const data = ["dailymed-spls-metformin.json", "dailymed-spls-aspirin.json"]
              .flatMap(
                (file) =>
                  (JSON.parse(fixture(file)) as { data: { setid: string; title: string }[] }).data,
              )
              .filter((entry) =>
                setId ? entry.setid === setId : entry.title.toLowerCase().includes(name),
              );
            return send(response, 200, "application/json", JSON.stringify({ data, metadata: {} }));
          }
          // As the real service does: it turns away a request that accepts nothing but XML.
          if (request.headers.accept === "application/xml") {
            return send(response, 406, "text/plain", "Not acceptable.");
          }
          const label = /^spls\/([^/]+)\.xml$/.exec(tail)?.[1];
          const file = label ? DAILYMED_LABELS[label] : undefined;
          return file ? send(response, 200, "application/xml", fixture(file)) : notFound();
        }
        case "clinicaltrials": {
          if (tail === "studies") {
            return send(response, 200, "application/json", fixture("clinicaltrials-search.json"));
          }
          const file = CLINICAL_TRIALS[tail.replace(/^studies\//, "")];
          return file ? send(response, 200, "application/json", fixture(file)) : notFound();
        }
        case "ecfr": {
          // As the real versioner does: it answers only a request that accepts a compressed answer.
          if (!/gzip|deflate|br/.test(String(request.headers["accept-encoding"] ?? ""))) {
            return send(
              response,
              406,
              "text/plain",
              "<error><message>This endpoint requires response compression. Send an Accept-Encoding header that permits compression.</message></error>",
            );
          }
          const gzipped = (type: string, body: string) =>
            response
              .writeHead(200, { "content-type": type, "content-encoding": "gzip" })
              .end(gzipSync(body));
          const section = url.searchParams.get("section");
          const asked = section ?? `part ${url.searchParams.get("part") ?? ""}`;
          if (tail === "api/versioner/v1/titles.json") {
            return gzipped("application/json", fixture("ecfr-titles.json"));
          }
          if (tail === "api/search/v1/results") {
            const words = url.searchParams.get("query") ?? "";
            return gzipped(
              "application/json",
              /manipulative|deceptive/i.test(words)
                ? fixture("ecfr-search.json")
                : '{"results":[],"meta":{"current_page":1,"total_pages":0,"total_count":0}}',
            );
          }
          const ancestry = /^api\/versioner\/v1\/ancestry\/[\d-]+\/title-17\.json$/.test(tail)
            ? ECFR_ANCESTRY[asked]
            : undefined;
          if (ancestry) return gzipped("application/json", fixture(ancestry));
          if (tail === "api/versioner/v1/versions/title-17.json") {
            const versions = ECFR_VERSIONS[asked];
            return gzipped(
              "application/json",
              versions ? fixture(versions) : '{"content_versions":[],"meta":{}}',
            );
          }
          const date = /^api\/versioner\/v1\/full\/([\d-]+)\/title-17\.xml$/.exec(tail)?.[1];
          const text = date ? (ECFR_TEXTS[`${asked}@${date}`] ?? ECFR_TEXTS[`${asked}@*`]) : null;
          if (text) return gzipped("application/xml", fixture(text));
          return send(response, 404, "application/json", '{"error":"No matching content found."}');
        }
        case "govinfo": {
          if (tail.startsWith("link/")) {
            const sectionNumber = url.searchParams.get("sectionnum");
            const cited = `${tail.slice("link/".length)}${sectionNumber ? `/${sectionNumber}` : ""}`;
            const target = GOVINFO_LINKS[cited];
            if (!target) return send(response, 400, "text/html", "<html>Bad request</html>");
            return response.writeHead(302, { location: target }).end();
          }
          // As the real API does: no key, or one it doesn't know, is 401.
          const key = request.headers["x-api-key"];
          if (typeof key !== "string" || key === REJECTED_GOVINFO_KEY) {
            const code = typeof key === "string" ? "API_KEY_INVALID" : "API_KEY_MISSING";
            return send(response, 401, "application/json", `{"error":{"code":"${code}"}}`);
          }
          const granule = /^packages\/[^/]+\/granules\/([^/]+)\/(summary|htm|xml)$/.exec(tail);
          const files = granule?.[1] ? GOVINFO_GRANULES[granule[1]] : undefined;
          if (!files || !granule) return notFound();
          if (granule[2] === "summary") {
            return send(response, 200, "application/json", fixture(files.summary));
          }
          return send(
            response,
            200,
            granule[2] === "htm" ? "text/html" : "application/xml",
            fixture(files.text),
          );
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
      openfda: `${base}/openfda`,
      dailymed: `${base}/dailymed`,
      clinicaltrials: `${base}/clinicaltrials`,
      ecfr: `${base}/ecfr`,
      govinfo: `${base}/govinfo`,
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
