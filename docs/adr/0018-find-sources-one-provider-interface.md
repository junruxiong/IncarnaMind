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
- **"sections"**: headings and paragraphs, kept as Markdown, one heading per section. Its Locations are sections ("§ 3.1 Cell-free methods", ADR-0011), and the quote check reads the text of the section the quote is in. PubMed Central's articles use it now, from their JATS XML. A law provider gives each provision as a section, headed as it is cited ("s. 3(2)", "Art. 6(1)(f)", "Item 1A"), and a provision becomes a Passage with no change to the core. The first such providers (the eCFR and GovInfo, #304) write a heading without its own "§", so the Location reads "§ 240.10b-5(b)" once (ADR-0011's amendment).
- **"html"**: a publisher's or a repository's web page, open to read. It is saved as Add a web page saves one (#261): a snapshot in the data folder's "Web pages", a Document of kind "web", with its address and its versions. Its reference record is the providers', filled before the page is read, so what the page says of itself fills only what they left empty. A page that can't be saved (a sign-in, a page with too little text) leaves the reference record, as a PDF that can't be had does.

**What isn't in the interface yet, and how it comes:**

- **A per-User key** (GovInfo's api.data.gov key, CourtListener's token, an optional OpenAlex key). It is stored only through `src/core/secrets.ts` (`sources.<id>.key`), set from a line under Find sources' search, and sent in a header, never in the address. It is never shared across Users, which those services penalise. openFDA's is optional (`takesKey`, #305). GovInfo's is needed (`needsKey`, #304): without it GovInfo is off, nothing is sent and no consent is asked, and the result says why with "Add key…" and where to get one. For a provider that needs a key, the flow's consent also says the key is sent ("provider-key", asked for that service alone through `consent.ts`'s `sendsTo`).
- **A paid provider** (a commercial paper search, a legal research service) is an optional Connector on the User's own key or subscription, not a provider shipped with the app. Results it may not let IncarnaMind store are reference-only and carry no checked Citation.
- **A User-Agent with a contact address** (SEC EDGAR asks for one, and on 2026-10-12 answered 403 "Undeclared Automated Tool" to `SOURCE_USER_AGENT`, which names only the project). The address is a project mailbox the maintainer supplies, read from the build configuration, never invented and never the User's; EDGAR waits for it (#558).
- **Licence and terms flags beyond a licence's commercial use** (may the full text be stored, does it need a licence application). These come with the first provider whose terms need them. A provider whose terms forbid storing its text offers no open copies, so its works are added as reference records.
- **A local copy searched offline** (Taiwan's law database, downloaded whole, #307). It is a provider whose `search` reads its own file, which it refreshes at most once a day after consent.

## Privacy and terms

- **Consent.** A new data flow, "find-sources", sends two kinds of data: "search-terms" (the words and filters searched for) and "work-ids" (the DOI, arXiv ID or PubMed ID of a work added). Nothing from the User's Documents or Minds is sent. Consent is asked once per provider, through `src/core/consent.ts`, before its first request. A declined provider is skipped, and the results say so. Like Look up's services, the flow lists a provider on the Privacy page once the User has decided on it, to revoke.
- **Local mode** turns Find sources off: nothing is searched or downloaded, and the Answer isn't offered the Tool.
- **Downloading an open copy** from a publisher's or a repository's address sends no User data. It is the address of a work the User chose, fetched once, without cookies, within 60 seconds and 100 MB, as adding a web page fetches it (#261), so it isn't a consent flow; `docs/privacy.md` says so. A copy through a provider (arXiv's PDFs, PubMed Central's text) goes to that provider's service, under its consent.
- **Only open copies, never from behind a paywall.** IncarnaMind fetches only addresses a provider lists as open access, and keeps only what is the file asked for: a "PDF" that turns out to be a sign-in page is dropped, and the next copy is tried, as a web page that asks for a sign-in isn't saved. A work with no open copy is added as its reference record, with "Get the PDF…" opening its page in the User's browser.
- **Licences** show on each result (CC BY, CC BY-NC-ND …). A licence that rules out commercial use says "non-commercial use only". The licence and version of a Document's text are kept with it (`Document.found`, migration 49).
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

## Drug labels and clinical trials (#305), as their operators state their terms (checked 2026-10-11)

These three providers are asked only when chosen, never with a search for papers (`searchedByDefault: false`): a drug name sent to the FDA for every search about CRISPR would be a data flow nobody asked for. The Library's panel starts with them off; the Answer's Tool takes a `kind` ("drug_labels", "clinical_trials") to name them. Typing an NCT number or a label's set id looks it up in its own provider whichever are chosen. Each is a service of its own for consent. The User-Agent is `SOURCE_USER_AGENT`, which names the project and no one.

- **openFDA** (https://open.fda.gov/apis/drug/label/): a label with each section in a field of its own.
  - Content is CC0 unless marked (https://open.fda.gov/terms/). Without a key: 240 requests a minute and 1,000 a day from one address; with a key, 120,000 a day (https://open.fda.gov/apis/authentication/). IncarnaMind waits 600 ms between requests.
  - The optional key is the User's own, kept only through `src/core/secrets.ts` (`sources.openfda.key`) and sent in an `X-Api-Key` header, never in the address. A key is never given back, and no error carries it. A request with an invalid key in that header answers 403 `API_KEY_INVALID`, so the header is the one openFDA reads; one with no match answers 404 `NOT_FOUND`, which is no results, not an error.
  - Every answer's `meta.disclaimer` ("Do not rely on openFDA to make decisions regarding medical care…") is shown with the result, and kept in the Document's first section. IncarnaMind adds no advice of its own.
  - The section fields, from real answers: `boxed_warning`, `indications_and_usage`, `dosage_and_administration`, `dosage_forms_and_strengths`, `contraindications`, `warnings_and_cautions`, `adverse_reactions`, `drug_interactions`, `use_in_specific_populations`, `pregnancy`, `pediatric_use`, `geriatric_use`, `overdosage`, `description`, `clinical_pharmacology`, `mechanism_of_action`, `pharmacokinetics`, `nonclinical_toxicology`, `clinical_studies`, `how_supplied`, `information_for_patients`, `spl_medguide`; an older or over-the-counter label has `warnings`, `precautions`, `purpose`, `do_not_use`, `ask_doctor` and others instead. Each is a list of one text. A section's number is not a field: it is the first word of the text ("5 WARNINGS AND PRECAUTIONS …"). A subsection such as 5.1 is inside its section's text, and tables are in separate `*_table` fields, which are left out.
  - openFDA lags DailyMed and doesn't name every label. One label was version 8 at openFDA and 9 at DailyMed, and its `openfda` block, which holds the drug and company names, was empty, so a search by name can't find it there. A label is found by name at DailyMed, and by set id at either.
- **DailyMed** (https://dailymed.nlm.nih.gov/dailymed/app-support-web-services.cfm): `spls.json?drug_name=` and `?setid=` search and give the set id, version and date; `spls/{setid}.xml` is the whole label.
  - Its page gives no limit and no terms. NLM's general API terms, from a search result and not read, ask for no more than 20 requests a second, so IncarnaMind waits half a second. A search lists labels newest first, not by relevance.
  - A single label as `.json` answers 415: only its XML is read. Its answers set cookies, which IncarnaMind doesn't keep.
  - The SPL's section codes (LOINC) name only a section's kind (a top-level section such as 43685-7 "Warnings and precautions", but every subsection is 42229-5 "unclassified"). The number is not in a code: it is in the section's title ("5.1 Lactic Acidosis"), as the label's author wrote it, in capitals or not. So the number is read from the title, and a label with none (an over-the-counter label's "Uses", "Warnings") has none. A boxed warning is headed "Boxed warning". The highlights, which repeat the sections, and the product listing and package label are left out.
- **ClinicalTrials.gov** (API v2, https://clinicaltrials.gov/data-api/api): search by condition, drug or words (`query.term`), and one record by NCT number, with no key.
  - Its rate limit and terms could not be read: its pages are drawn by script, and the OpenAPI description (https://clinicaltrials.gov/api/oas/v2) gives neither. Third-party guides say about 50 requests a minute (some, three a second). IncarnaMind waits one second between requests until the operator's own figure is read.
  - A record is large (2 MB for NCT04368728, with its results), and is fetched only when the User adds it.
- **Which label provider reads the label.** Both search. DailyMed's copy is tried first, because its sections keep their subsections and numbers ("5.1 Lactic Acidosis" is a Passage of its own); openFDA's fields are the fallback when DailyMed isn't allowed or doesn't answer, and they are what a label DailyMed doesn't carry is read from. A work both found is one work, by its set id.
- **Locations.** A Document made from a label has "About this label" (its set id, version and effective date, and openFDA's disclaimer when it was read there), then a section for each of the label's own, as `§ 5.1 Lactic Acidosis` cites it (ADR-0011's Markdown Locations). A trial's Document has "Overview" (with its NCT number and the record's last-update date), "Conditions and design", "Arms and interventions", "Primary outcome", "Secondary outcomes", "Eligibility criteria" and, when results are posted, "Results: participant flow", "Results: outcome measures" and "Results: adverse events". A study with no posted results has no Results sections. The Document's version date is the label's effective date or the record's last update.
- No migration: `found_sources.work_key` holds `setid:…` and `nct:…` as it holds the other keys.

## US federal regulations and the US Code (#304), as their operators state their terms (checked 2026-10-12)

The eCFR and GovInfo are asked only when chosen, as drug labels are (`searchedByDefault: false`); a citation the User types ("17 CFR 240.10b-5", "17 CFR Part 243", "42 U.S.C. § 1983", `./citations.ts`) looks itself up whichever are chosen; the Answer's Tool names them with `kind: "us_law"`. Each is a service of its own for consent. SEC EDGAR, the third provider #304 asked for, waits for a declared contact address (#558).

- **The eCFR** (https://www.ecfr.gov), the CFR as amended to date, by the Office of the Federal Register and GPO: its search API (`/api/search/v1/results`, with `date=current`) and its versioner (`titles`, `ancestry`, `versions`, `full`) answer without a key.
  - Its developer pages redirect an automated reader to an "unblock" page, so its terms and limits couldn't be read. IncarnaMind waits a second between requests.
  - The versioner answers 406 "This endpoint requires response compression" to a request that doesn't accept a compressed answer, so every request sends `Accept-Encoding: gzip, deflate, br`.
  - Its operators call it "an unofficial editorial compilation of CFR material and Federal Register amendments … not an official legal edition of the CFR" (https://www.archives.gov/federal-register/cfr/about-ecfr; GovInfo's https://www.govinfo.gov/help/cfr says the same). So each result says "Unofficial compilation, not the official edition", the Answer's Tool tells the model so, and the Document's first section, "About this text", says it.
  - A citation is looked up by its ancestry on the day (today: the title's "up to date as of" day, from `titles.json`, kept for an hour). A date ("… as of 2019-06-30") gives the version in force that day; the eCFR's history starts in January 2017. A section the eCFR doesn't have on the day is looked up in its versions: one removed since (17 CFR 240.15Aj-1, removed 2025-03-24) is found as removed, with its text as in force the day before; one it never had finds nothing.
  - A section is one request for its text, a part (Part 240 is 3.8 MB of XML) one more; each is fetched only when the User adds it.
- **GovInfo** (https://api.govinfo.gov/docs/), GPO's official publications: a section of the United States Code, or of the CFR's annual edition, by its citation.
  - Its API needs an api.data.gov key, the User's own (`needsKey`), sent as `X-Api-Key`. A key it doesn't know answers 401 `API_KEY_INVALID` (not 403), which the result says plainly with "Change key…". Its own errors name https://www.govinfo.gov/api-signup, which the key line links to.
  - The API has no search by citation, and its search is a POST. GovInfo's link service (`/link/uscode/{title}/{section}`, `/link/cfr/{title}/{part}?sectionnum=`) redirects a citation to its granule ("USCODE-2024-title42-chap21-subchapI-sec1983"), and answers 400 for one it doesn't have. IncarnaMind reads the redirect without following it, and without the key; the granule's `summary`, and its text when the User adds it (`htm` for the US Code, `xml` for the CFR), come from the API. GovInfo only looks citations up: words go to the eCFR.
  - A key allows 1,000 requests an hour; past that the API answers 429 for up to an hour. A 429 stops the lookup ("rate-limited"), and an add that meets one keeps nothing (failed, "rate-limited"), so adding again later adds the text. IncarnaMind waits 500 ms between requests.
  - Its works have no copyright (17 U.S.C. § 105, https://www.govinfo.gov/about/policies).
  - The eCFR's text and GovInfo's edition of the same CFR section are two works, not merged: different editions, as of different days.
- **Locations.** A section is headed as it is cited, without its "§" ("240.10b-5 Employment of manipulative and deceptive devices", "1983 Civil action for deprivation of rights"), and each paragraph it numbers is a section of its own ("240.10b-5(b)", "78j(a)(1)", `./provisions.ts`), nested by the order of their markers: the CFR's (a), (1), (i), (A), then italic (1) and (i); the US Code's (a), (1), (A), (i), (I). A marker that could be a letter or a Roman numeral ("(i)" after "(h)") is read by the one after it. Text set after a list at the section's margin ("in connection with the purchase or sale of any security") and the source notes go under the bare citation ("240.10b-5"); the US Code's editorial and statutory notes keep their own headings. A Document starts with "About this text": the citation, the edition and the day it is as of. Its version date is that day (the eCFR's), the edition's revision date (the CFR's annual edition) or the date the US Code's edition is current through.
- No migration: `found_sources.work_key` holds `cfr:…` and `govinfo:…`.

## The Answer's Tool

`find_sources` (`src/core/sources/tool.ts`) runs the same search, under the same consent. Its results are public records, not the User's data, so the chat flow doesn't count them as "tool-results". Its Effects are network calls to each provider's service, which consent covers, so it runs without an approval card.

Only a Question that names scholarly works, or asks to find sources, is offered it (`src/core/sources/intent.ts`):
- papers, literature, preprints, publications, studies, a DOI, arXiv or PubMed; 论文, 文献, 预印本 or 综述;
- "find sources on …", "search for articles about …", "suggest some references"; "查找一些关于……的资料".
- a US citation ("17 CFR 240.10b-5", "42 U.S.C. § 1983"), or "find the federal regulations on …"; "查找……法规" (#304).

"Published", or "article" without such a verb, is left out: "When are the tide tables published?" and a contract's "Article 5" aren't asking for papers. Offered to every Answer, a model could send searches out, and put a consent request in front of the User, for a Question about their own Documents, and a small local model's window would carry the Tool for nothing. The Library's Find sources is there for anything the words miss. The assistant (#253) may widen this, for example within a conversation that has already searched.

A model that cites with structured output, as small local models do (ADR-0007), is offered no Tools, so it doesn't search. With qwen3.5:4b, "Find papers…" was answered from the model's own knowledge. Such a User finds sources in the Library.

What it finds is never cited with a marker. The Answer names the works, and its Tool-call row's "Show results" opens them in the Library to add.

## Considered options

- **Searching from the window (the renderer).** Rejected: the window reaches nothing on the network, and SEC data has no CORS.
- **A cached PDF from OpenAlex's content API.** Rejected: it costs $0.01 a PDF, 10% of the keyless daily budget.
- **PMC's pages or its OA Web Service.** Rejected: automated retrieval from PMC's pages isn't allowed, and the OA Web Service is retired in 2026. E-utilities is the way PMC allows.
- **Unpaywall for open copies.** Not needed: OpenAlex carries Unpaywall's data, and Unpaywall needs an email on every request.
- **Offering the Tool to every Answer.** Rejected for now, as above.
- **Keeping found PDFs in the Output location (#124).** Rejected for now. Saved text and web pages (#261) are kept in the data folder, so found copies are too, in its "Sources" folder. The first save would otherwise ask where new files go, in the middle of adding a paper. A copy deleted permanently is removed from it.

## Consequences

- Migration 49 adds `found_sources`: one row per Document, or reference record with no Document, that Find sources added. It holds the provider, the work's key, the address the text came from, its licence, version and version date, and when it was fetched. The record's fields are written as Look up's ("lookup") before the file is read, so a reading of the file fills only what the providers left empty, and the User's edits stay theirs.
- `Document.found` says where a found Document's text came from. `reference_records.document_id` stays NULL for a reference added alone. Adding the work later with an open copy links that record to the new Document.
- Tests never reach the services. A local stand-in (`tests/helpers/sourceServices.ts`) answers with responses recorded from each provider on 2026-10-11, trimmed (`tests/fixtures/sources/`). The core's tests also point every provider at an address nothing answers at, unless a test gives its stand-in.
