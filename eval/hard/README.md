# The hard tier

`npm run eval:hard` checks how retrieval holds up where the gating set (`eval/README.md`) is easy: Questions worded unlike their Documents, numbers read from tables, answers spread over pages or Documents, Questions the library can't answer, Questions in the other language, and two versions of one Document. It asks them of a library at scale, across domains, in every retrieval mode, with embeddings on, so the default keyword search reranked by the built-in model can be compared with hybrid search reranked. It is reported, and never gates.

Why: embeddings are off by default (ADR-0009, 2026-10-10), and search is keyword search (FTS5's BM25), the built-in cross-encoder over keyword search's top 20, and a translated second query for Questions in another language. That was decided on small, easy sets: on the gating set, keyword, vector and hybrid search tie (32, 31 and 32 of 40), and the every-format set is all hits. This tier is where vector search should earn its place if it ever does.

## Running it

```sh
npm run eval:hard
```

1. **Fetch.** The library's files are fetched into the evaluation's cache, `~/.cache/incarnamind-eval/hard/` (or `$INCARNAMIND_EVAL_CACHE/hard/`), and each is checked against the SHA-256 in `library.json`. A later run fetches nothing that's already there and unchanged. A file that has changed at its source, or can't be fetched, is left out and listed in the report, with the Questions that need it; the run goes on.
2. **Index.** A new temporary data folder, never the User's, gets every Document with `addDocuments`, processed by the core. As in `npm run eval`, the library turns embeddings on with the built-in model before adding them, as a User can in Settings, so the dense modes can be compared: the run stops with a message if no Passage was embedded.
3. **Check.** Each Question is checked against the text the app stored (`textProblems` in `lib/questions.ts`): each quote on its expected Units, on no other Unit of its Document, inside a Passage that covers them, and, for a near-duplicate, in no other version. A Question that fails is left out and listed.
4. **Search.** Each Question is searched with keyword, vector and hybrid search, then reranked by the built-in reranking model over keyword search's top 20 (what the search Tool does by default, and the gating set's gate) and over hybrid search's candidates (what it does with embeddings on). A cross-lingual Question is searched again with its hand-written translation.
5. **Time keyword search.** Each Question's query (and translation) runs again on the run's data folder with the app's own keyword SQL, ordered by `bm25(passages_fts)` as `keywordSearch` orders it, and ordered by FTS5's `rank` column (bm25() with the same weights), each timed once after a warm-up pass (`lib/keywordTiming.ts`). FTS5 has no top-k pruning, so this is the cost that grows with the library. Measurement only, at this library's size (about 48,000 Passages; the tier doesn't reach 100,000).
6. **Ask** (with a chat model). Embeddings are turned off again first, so Answers search as Users' do by default, as in `npm run eval`. Each Question is asked once, in a Mind of its own, and the Citations are scored per difficulty.
7. **Report** into `eval/results/<start time>-hard/`: `report.md`, `report.json` and, with a chat model, `reviewer-sheet.csv`.

With a chat model, set the same variables as `npm run eval` (setting them is the consent to send the Questions and Passages to it):

```sh
INCARNAMIND_EVAL_CHAT_KIND=anthropic \
INCARNAMIND_EVAL_CHAT_MODEL=claude-sonnet-5-5 \
INCARNAMIND_EVAL_CHAT_KEY=sk-ant-... \
npm run eval:hard
```

**Cost and time, roughly, on an Apple silicon Mac:**

- **Download:** 0.71 GB the first time (the manifest's total is checked to stay under 3 GB), a few minutes; arXiv's 50 papers come one every 3 seconds, as its terms ask.
- **Indexing:** about 48,000 Passages, every one embedded. Embedding takes most of the time, at the built-in model's 20 to 28 Passages a second: about 30 to 40 minutes. Keyword search alone would have every Document ready within a few minutes; the report gives both times.
- **Searching and reranking:** about 10 minutes.
- **Memory:** the run reports its peak; expect a few GB (the embedding and reranking models are about 1 GB each while loaded; the vectors about 75 MB).
- **Without a chat model:** about an hour all told, with nothing sent anywhere once the models are cached.
- **With a chat model:** 205 Answers more, each with up to 5 searches: two to three hours. At `claude-sonnet-5-5`'s $2 and $10 per million input and output tokens, with 25,000 to 50,000 input and about 1,000 output tokens an Answer, that is about $12 to $25.

`INCARNAMIND_EVAL_KEEP_DATA=1` keeps the data folder; `INCARNAMIND_EVAL_CACHE` moves the cache.

## The library

`library.json` lists 400 Documents (319 English, 81 Chinese) from 37 sources, each with its source, URL (or archive member, or repository path), SHA-256, size and page count, its domain and language, and, where it has them, its versions (`group`, `edition`) and its translation (`translationOf`). They are never committed.

| Domain | Where from |
|---|---|
| Contracts | 100 commercial contracts from CUAD v1 (one 106 MB archive, from which the run takes them out); Chinese model contracts from the State Administration for Market Regulation's model-contract library |
| Company reports and filings | FinanceBench's open sample (10-Ks, 10-Qs, 8-Ks and earnings releases), several companies in two or more years; Chinese companies' annual reports from the Shenzhen Stock Exchange; the gating set's JP Morgan ESG report |
| Academic papers | 50 arXiv papers from Qasper's test set; Chinese open-access papers (Hans Publishers, CC BY 4.0); the gating set's papers and Chinese Wikipedia articles |
| Government and international reports | UN, UN SDG Reports, IPCC, FAO, World Bank, UNDP, in English and Chinese; Federal Reserve, Census, IRS, USDA ERS; HM Treasury and other gov.uk reports; People's Bank of China and SAFE reports |
| Technical manuals | NIST, FAA, NASA, OCC, CISA, USGS, ready.gov; WHO's biosafety manual in both languages; Chinese environmental monitoring standards |
| Medical and regulatory | FDA drug labels and guidance; UKHSA patient group directions and training slides; VA/DoD guidelines; WHO guidelines in both languages; NMPA notices, guidelines and statistics; the gating set's ABPI Code |

| Domain | English | Chinese |
|---|---|---|
| Contracts | 100 | 10 |
| Company reports and filings | 41 | 9 |
| Academic papers | 54 | 21 |
| Government and international reports | 52 | 23 |
| Technical manuals | 33 | 6 |
| Medical and regulatory | 39 | 12 |

351 PDFs (16,494 pages), 27 Word, 14 Excel, 6 PowerPoint and 2 CSV files. 42 Documents are linked to their translation in the other language, and 50 groups hold versions of one Document (two years of a report or a company's filings, two quarters, two revisions of a standard or a label, with their translations). `report.md` gives what was added, per domain, language and format.

**Licences.** Each source records its licence or terms, the page they're on, and their own words permitting the download (`sources` in `library.json`); `tests/eval/hardLibrary.test.ts` checks each against an allowed list:

- CC BY 4.0 (CUAD, Hans Publishers), CC BY 3.0 IGO (World Bank, UNDP), Open Government Licence v3.0 (gov.uk), works of the US federal government (17 U.S.C. 105), and official documents of Chinese state organs (Copyright Law of the PRC, art. 5).
- arXiv's terms, which allow retrieving e-prints for personal use or research, at one request every 3 seconds.
- Non-commercial use only: FinanceBench (CC BY-NC 4.0), WHO and FAO (CC BY-NC-SA 3.0 IGO), the World Bank's China Economic Update (CC BY-NC 3.0 IGO), and the terms of the UN, the IPCC and the Shenzhen Stock Exchange (download for personal or non-commercial use). The evaluation of a free, open-source app (ADR-0002) is non-commercial; revisit these if that changes.
- FDA's drug labels are the makers' labelling, published by FDA, whose website policy says its content isn't copyrighted unless noted; they are filed under US government works with that caveat.
- The files are downloaded to the User's own cache and read there; nothing is redistributed.

**Left out on purpose:**

- SEC EDGAR and Wikimedia's APIs, whose access policies ask for contact details in each request: requests here name no one (the User-Agent is "IncarnaMind document evaluation"). FinanceBench's filings are fetched where FinanceBench publishes them instead.
- HKEXnews, whose terms allow no use they don't list; NICE, whose terms bar use in retrieval systems outside the UK; gov.cn's English site, whose content "shall not be republished or used in any form" without authorization; and the IMF and cninfo, whose terms couldn't be read.
- Hosts that refuse requests from scripts or answer with a JavaScript challenge, which a run mustn't get round: GAO, CBO, CRS, FEMA, CDC, BLS, NOAA and others; the Shanghai Stock Exchange's files (its terms would allow them); the State Council Information Office's white papers; the CDE's drug-review guidelines.

**Known to change:** a few US government URLs aren't versioned (Medicare & You; FDA guidance at `fda.gov/media/<id>/download`). When they are replaced, their checksums stop matching and the run leaves them out, saying so.

## The Questions

`questions.json`: each Question has one difficulty, a domain, a language, and the Passages its answer needs (`expected`: Document, Units, and a quote from them).

| Difficulty | What it tests | Expected |
|---|---|---|
| `easy` | The answer is in the Documents' own words | One Passage |
| `paraphrase` | Asked in words unlike the passage's | One Passage |
| `table-number` | A number read from a table (PDF, Word, Excel or CSV) | The row with its label |
| `multi-page` | Two facts on two pages of one Document | Two Passages |
| `cross-document` | Facts from two Documents | Two Passages |
| `unanswerable` | The library doesn't say | None: an Answer should cite nothing |
| `cross-lingual` | Asked in the other language from its Document's, with `translatedQuery` | One Passage, in a Document with no translation in the Question's language |
| `near-duplicate` | Two versions of a Document differ, and the Question names which | One Passage, whose quote is in no other version |

205 Questions:

| Difficulty | English | Chinese |
|---|---|---|
| easy | 24 | 5 |
| paraphrase | 24 | 8 |
| table-number | 19 | 7 |
| multi-page | 18 | 5 |
| cross-document | 15 | 5 |
| unanswerable | 19 | 6 |
| cross-lingual | 8 (about Chinese Documents) | 18 (about English Documents) |
| near-duplicate | 16 | 8 |

Per domain: contracts 33, filings 46, papers 32, reports 57, manuals 16, medical 21. 17 of them ask about Word, Excel, PowerPoint or CSV files; 64 come from benchmarks (CUAD 25, FinanceBench 24, Qasper 15).

**Sources.** CUAD's clause annotations (contracts), Qasper's evidence (papers) and FinanceBench's evidence pages (filings), each mapped to the pages the app stores and reworded where the original assumed the Document was in front of the reader (`benchmark` names the original). The rest were written for this library, in English and Chinese, from the Documents' text.

**Checks.**

- `tests/eval/hardQuestions.test.ts` checks the rules above, at least 15 Questions per difficulty with both languages in each, and every domain.
- Every quote was checked against the text the app's own processing extracts (`processFile`, pdf.js), one Document at a time: on its expected Units, on no other, inside a covering Passage; near-duplicates against their other versions. The run checks again on what it stored.
- Unanswerable Questions were checked by searching the whole library's extracted text for their subject.

## Scoring

- **Hit:** every Passage a Question needs is among the top 5 of a search, each by the gating set's rule (the expected Document, covering the expected Units, holding the quote). Multi-page and cross-document Questions also report how many were found in part.
- **Cross-lingual:** a Passage found by either the Question's own search or its translation's counts, as an Answer is told to search again in the Documents' language; the Question's own query alone is reported too.
- **Modes:** keyword, keyword + rerank (the search Tool's default), hybrid, hybrid + rerank (the search Tool with embeddings on), vector.
- **Reported:** per difficulty, per domain, per language, and per domain × difficulty in each mode; ranks per Question; indexing time (keyword search ready, every Document embedded, the embedding model's own time), Passages and peak memory; keyword search's time per query (median, 95th percentile) beside the reranker's.
- **Citations,** with a chat model, per difficulty: the gating set's figures, how many Answers cite a Document the Question needs (for a near-duplicate, the right version, and how many cite another), and, for unanswerable Questions, how many are answered without a Citation.

## Results

None yet: the first run is pending.

## Files

- `run.hard.ts` and `vitest.config.ts`: the run.
- `library.json`: the Documents and their sources; `lib/manifest.ts` reads and checks it.
- `lib/fetch.ts`: fetching into the cache, with checksums.
- `questions.json`: the Questions; `lib/questions.ts` reads and checks them, and checks them against the stored text.
- `lib/scoring.ts`: searches, reranking, hits and tallies; `lib/keywordTiming.ts`: keyword search's time per query; `lib/citations.ts`: Citations per difficulty; `lib/report.ts`: the reports.
- Tests (`npm test`, no network, no models): `tests/eval/hardLibrary.test.ts`, `tests/eval/hardQuestions.test.ts`, `tests/eval/hardScoring.test.ts`.
