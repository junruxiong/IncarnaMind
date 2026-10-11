# Find sources: one Source provider interface, asked with consent, adding only open copies

Users need to find and add the sources they cite without leaving IncarnaMind (#296). The first are academic papers. Laws, regulations, filings, drug labels and clinical trials come next (#303–#307), and each must be a small provider added to one framework, not a change to it.

Find sources searches public services from the core (the main process, never the window), merges what they find into one ranked list, and adds a work the User chooses to the Library: an open-access copy as a Document with its reference record (#286) filled, or, with no open copy, the reference record alone. The same search is a Tool an Answer can call (ADR-0013). The code is in `src/core/sources/`.

## The provider interface

A Source provider (CONTEXT.md) is one module implementing `SourceProvider` (`src/core/sources/types.ts`), registered in `src/core/sources/providers/index.ts`, with its strings (`sources.provider.<id>.about`, English and Chinese):

- `id`, `name` and `service`: the service is what consent is asked for, once per provider.
- `terms`: where its operator states its terms and limits, and `intervalMs`, the least time between two requests to it.
- `search(query, context)`: words and filters (`fromYear`, `toYear`, `openAccessOnly`, `limit`) to ranked works. Absent for a provider that only looks works up.
- `resolve(identifier, context)`: one work by an identifier it holds (DOI, arXiv ID, PubMed ID; later a CELEX number, a CFR section, an NCT number).
- `fetchFullText(location, context)`: an open copy it offered with itself as `provider`.

A provider's work (`ProviderWork`) carries:

- its identifiers, so works two providers share merge into one;
- its reference record's fields, in the record's own shape (CSL names);
- a citation count, when the provider counts them;
- the **date of the version** described (`versionDate`). Laws and labels change, and preprints have versions, so the version a Document holds is recorded with it and a Citation is checked against that stored text (ADR-0010);
- its page at the publisher (`landingUrl`), for "Get the PDF…";
- its **open copies**, best first (`FullTextLocation`). Each has a `format`, its `version` (published, accepted or submitted), its `licence` as written, and the `provider` that fetches it, or null for an open-access address on the web.

The framework around the providers is the same for every one. It does consent, pacing and backoff, merging (by any shared identifier, ranked by reciprocal rank fusion with k = 60, as hybrid search fuses), downloading, saving, recording where a Document came from, the Answer's Tool and the Library's results.

**Full text, by its shape.** `FullText` is one of these:

- **"pdf"**: kept as it came. Its Locations are pages.
- **"sections"**: headings and paragraphs, kept as Markdown, one heading per section. Its Locations are sections ("§ 3.1 Cell-free methods", ADR-0011), and the quote check reads the text of the section the quote is in. PubMed Central's articles use it now, from their JATS XML. A law provider gives each provision as a section, headed as it is cited ("s. 3(2)", "Art. 6(1)(f)", "Item 1A"), and a provision becomes a Passage with no change to the core. Whether a provision's Location drops the "§" before its own label is the first such provider's decision (#303), as an amendment to ADR-0011.
- **"html"**: a publisher's web page. It is to be saved as a web page snapshot (#261, `saveWebPage`) once that is in IncarnaMind. Until then such a work is added as its reference record, with the page to open. The seam is `downloadable` in `src/core/sources/index.ts`.

**What isn't in the interface yet, and how it comes:**

- **A per-User key** (GovInfo's api.data.gov key, CourtListener's token, an optional OpenAlex key). It is stored only through `src/core/secrets.ts`, set in Settings → Search, and sent in a header, never in the address. The flow's consent then also says the key is sent. It is never shared across Users, which those services penalise. The first provider that needs one adds it (#304).
- **A paid provider** (a commercial paper search, a legal research service) is an optional Connector on the User's own key or subscription, not a provider shipped with the app. Results it may not let IncarnaMind store are reference-only and carry no checked Citation.
- **A User-Agent with a contact address** (SEC EDGAR asks for one). The address is the maintainer's, read from the build configuration, never invented and never the User's (#304).
- **Licence and terms flags beyond a licence's commercial use** (may the full text be stored, does it need a licence application). These come with the first provider whose terms need them. A provider whose terms forbid storing its text offers no open copies, so its works are added as reference records.
- **A local copy searched offline** (Taiwan's law database, downloaded whole, #307). It is a provider whose `search` reads its own file, which it refreshes at most once a day after consent.

## Privacy and terms

- **Consent.** A new data flow, "find-sources", sends two kinds of data: "search-terms" (the words and filters searched for) and "work-ids" (the DOI, arXiv ID or PubMed ID of a work added). Nothing from the User's Documents or Minds is sent. Consent is asked once per provider, through `src/core/consent.ts`, before its first request. A declined provider is skipped, and the results say so. Like Look up's services, the flow lists a provider on the Privacy page once the User has decided on it, to revoke.
- **Local mode** turns Find sources off: nothing is searched or downloaded, and the Answer isn't offered the Tool.
- **Downloading an open copy** from a publisher's or a repository's address sends no User data. It is the address of a work the User chose, fetched once, without cookies, within 60 seconds and 100 MB, as adding a web page fetches it (#261), so it isn't a consent flow; `docs/privacy.md` says so. A copy through a provider (arXiv's PDFs, PubMed Central's text) goes to that provider's service, under its consent.
- **Only open copies, never from behind a paywall.** IncarnaMind fetches only addresses a provider lists as open access, and keeps only what is the file asked for: a "PDF" that turns out to be a sign-in page is dropped, and the next copy is tried. A work with no open copy is added as its reference record, with "Get the PDF…" opening its page in the User's browser.
- **Licences** show on each result (CC BY, CC BY-NC-ND …). A licence that rules out commercial use says "non-commercial use only". The licence and version of a Document's text are kept with it (`Document.found`, migration 46).
- **Pacing.** Requests to one service go one at a time, `intervalMs` apart, measured from the end of the last. The pacing is shared app-wide, so Look up (#286) and Find sources share arXiv's. A 429 or 503 with a Retry-After of at most 30 seconds waits and tries once more; otherwise the provider's part of the search fails with "rate-limited", in plain words. Searches are cached for 30 minutes, so the same search, from the Library or an Answer's "Show results", isn't sent twice. Nothing is bulk-downloaded: one copy, when the User adds a work.

## The academic providers, as their operators state their terms (checked 2026-10-11)

- **OpenAlex**: search, metadata, open-copy locations and citation counts.
  - "All data is CC0" (https://help.openalex.org/how-to-use-the-api/api-overview).
  - "You can make basic queries with no key at all." The keyless budget is $0.10 a day, and a search costs $0.001, so about 100 searches a day. Looking up one work is free. More than 100 requests a second gets a 429, which asks for exponential backoff (https://help.openalex.org/how-to-use-the-api/rate-limits-and-authentication, https://help.openalex.org/access/example-costs/).
  - Its cached PDFs cost $0.01 each, so IncarnaMind downloads open copies from where OpenAlex says they are instead.
  - Its current pages describe no polite pool or `mailto`, so none is sent.
  - IncarnaMind waits 100 ms between requests.
- **arXiv**: preprints, every one open.
  - "No more than one request every three seconds, and limit requests to a single connection at a time", across all of one's machines. Its metadata is CC0. E-print content may be retrieved and stored "for your own personal use, or for research purposes", fetched from arXiv.org itself (https://info.arxiv.org/help/api/tou.html).
  - IncarnaMind keeps arXiv's requests 3 seconds apart on one lane, PDFs included, and fetches a PDF only when the User adds a paper.
- **PubMed and PubMed Central**, through NCBI's E-utilities (https://www.ncbi.nlm.nih.gov/books/NBK25497/).
  - No more than three requests a second without a key, and a `tool` name. The `email` must be a developer's, never a user's, so none is sent. NCBI's page sat behind a browser check on 2026-10-11, so these rules were read from its search snippet and the libraries that quote it.
  - PMC permits automated retrieval only through its own services, E-utilities among them (https://pmc.ncbi.nlm.nih.gov/tools/amdataset). So PMC's full text is fetched with EFetch (`db=pmc`) as JATS XML, never from PMC's pages, and an article whose publisher withholds its text gives no body, so it stays a reference.
  - Each article's licence is read from its JATS.
  - IncarnaMind waits 400 ms between requests.
- **Crossref**: a DOI's record, when the User types a DOI.
  - Its metadata is facts, mostly not subject to copyright (https://www.crossref.org/documentation/retrieve-metadata/rest-api/).
  - From 2025-12-01 its public pool allows 5 single-record requests a second, one at a time. The polite pool needs a `mailto` address (https://www.crossref.org/blog/announcing-changes-to-rest-api-rate-limits/), which IncarnaMind never invents for a User, so it stays in the public pool, one request a second.
  - It knows no open copies itself: OpenAlex, asked at the same time, does.

## The Answer's Tool

`find_sources` (`src/core/sources/tool.ts`) runs the same search, under the same consent. Its results are public records, not the User's data, so the chat flow doesn't count them as "tool-results". Its Effects are network calls to each provider's service, which consent covers, so it runs without an approval card.

Only a Question that asks for papers or other sources is offered it (`src/core/sources/intent.ts`, words in English and Chinese). Offered to every Answer, a model could send searches out, and put a consent request in front of the User, for a Question about their own Documents, and a small local model's window would carry the Tool for nothing. The Library's Find sources is there for anything the words miss. The assistant (#253) may widen this, for example within a conversation that has already searched.

What it finds is never cited with a marker. The Answer names the works, and its Tool-call row's "Show results" opens them in the Library to add.

## Considered options

- **Searching from the window (the renderer).** Rejected: the window reaches nothing on the network, and SEC data has no CORS.
- **A cached PDF from OpenAlex's content API.** Rejected: it costs $0.01 a PDF, 10% of the keyless daily budget.
- **PMC's pages or its OA Web Service.** Rejected: automated retrieval from PMC's pages isn't allowed, and the OA Web Service is retired in 2026. E-utilities is the way PMC allows.
- **Unpaywall for open copies.** Not needed: OpenAlex carries Unpaywall's data, and Unpaywall needs an email on every request.
- **Offering the Tool to every Answer.** Rejected for now, as above.
- **Keeping found PDFs in the Output location (#124).** Rejected for now. Saved text and web pages (#261) are kept in the data folder, so found copies are too, in its "Sources" folder. The first save would otherwise ask where new files go, in the middle of adding a paper. A copy deleted permanently is removed from it.

## Consequences

- Migration 46 adds `found_sources`: one row per Document, or reference record with no Document, that Find sources added. It holds the provider, the work's key, the address the text came from, its licence, version and version date, and when it was fetched. The record's fields are written as Look up's ("lookup") before the file is read, so a reading of the file fills only what the providers left empty, and the User's edits stay theirs.
- `Document.found` says where a found Document's text came from. `reference_records.document_id` stays NULL for a reference added alone. Adding the work later with an open copy links that record to the new Document.
- Tests never reach the services. A local stand-in (`tests/helpers/sourceServices.ts`) answers with responses recorded from each provider on 2026-10-11, trimmed (`tests/fixtures/sources/`). The core's tests also point every provider at an address nothing answers at, unless a test gives its stand-in.
