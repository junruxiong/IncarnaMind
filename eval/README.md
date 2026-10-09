# Evaluation

`npm run eval` measures retrieval and Citation quality against the v1 design's Success Criteria (`docs/designs/v1-product-validation.md`), on the evaluation set the retrieval prototype settled ([ADR-0009](../docs/adr/0009-retrieval-storage-and-search.md)). Issue #31.

It drives the core's public interface in Node, the way the desktop app's UI does:

1. It creates a temporary data folder. It never touches the app's real data folder.
2. It adds the seven sample PDFs in `data/` and the five Chinese Wikipedia articles in `retrieval/fixtures/` with `addDocuments`, and waits until each is processed: text extracted, Passages built and embedded with the real built-in model, multilingual-e5-small.
3. It searches for each Question with `searchPassages` in hybrid, keyword and vector mode, and scores the top 5. A cross-lingual Question with a translated query is searched with that too.
4. It reranks what the search Tool hands a reranker (keyword search's top 10 and vector search's top 10, each Passage once) with the built-in reranking model, mmarco-mMiniLMv2-L12-H384, as the search Tool does by default, and scores the top 5 again: this is the gate. Other reranking candidates, when given, are scored the same way.
5. When a chat model is given, it asks each Question with `askQuestion` and scores the Citations of each Answer. The core reranks those Answers' searches with the real built-in reranking model, as the app does.
6. It does steps 1 to 5 again for the every-format set (see [Every format](#every-format)), in a temporary data folder of its own: Word, PowerPoint, Excel, CSV, Markdown, plain-text and PDF Documents, asked about the hard places in each. That set is reported per format and never gates.
7. It writes a report under `eval/results/` and prints a summary.

The command fails (exit code 1) when a gating target is missed. It isn't part of `npm test` or CI's default run.

## Running it

```sh
npm run eval
```

This runs retrieval only. It needs no keys and no network once the models are cached.

The first run downloads the two built-in models from Hugging Face, the embedding model (about 135 MB) and the reranking model (about 136 MB), into `~/.cache/incarnamind-eval/models/`, outside the repository. Later runs reuse them, offline. The core downloads and checks the files itself, against the SHA-256 hashes pinned in `src/core/embedding/model.ts` and `src/core/reranking/model.ts`: the data folder's `models/` is a link to the cache. If a model can't be downloaded, the run stops and says which, from where, and why.

The models run on Node worker threads (`lib/embedderWorker.ts`, `lib/rerankerWorker.ts`). They use the same code as the app's utility processes: `createOnnxEmbedder` and `createOnnxCrossEncoder`, served over the channels in `src/core/embedding/channel.ts` and `src/core/reranking/channel.ts`.

A run takes about two minutes on an Apple M2 Max, most of it spent embedding about 1,200 Passages, then reranking 60 searches. The every-format set adds 22 small Documents and 145 searches.

### Reranking candidates

The built-in reranking model, on by default in the app (Settings → Reranking), was chosen from three candidates, all multilingual cross-encoders under Apache-2.0, as int8 ONNX (`src/core/reranking/model.ts`). The built-in one always runs and gates; name others to compare with it, or all of them:

```sh
INCARNAMIND_EVAL_RERANK=all npm run eval
INCARNAMIND_EVAL_RERANK=mmarco-minilm,bge-m3 npm run eval
```

| Id | Model | Download |
|---|---|---|
| `mmarco-minilm` | cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 (built in, on by default) | 136 MB |
| `gte-multilingual` | Alibaba-NLP/gte-multilingual-reranker-base | 358 MB |
| `bge-m3` | BAAI/bge-reranker-v2-m3 | 588 MB |

- **Candidates:** the search Tool doesn't hand its reranker the fused list: it hands it keyword search's top 10 and vector search's top 10, each Passage once, so a hit only one of them found isn't pushed out by fusion first (#31: en-03 and en-14 were keyword-only hits, en-12 and zh-03 vector-only, and all four fell out of the fused top 5). The evaluation builds the same set from `searchPassages` in keyword and vector mode, with the Tool's own `topsOfEach`; a unit test checks that the Tool's reranker gets exactly that set. The report gives how many candidates there were per search: between 10 and 20.
- **How:** after the searches, each model in turn, the built-in one first, reranks every Question's candidates (and each translated query's), with the core's own reranking code (`createRerankingModel`: what the model reads, its scores) and the model on a worker thread (`lib/rerankerWorker.ts`), as the app's reranking utility process runs it.
- **Downloads:** the first run downloads each candidate into the model cache, next to the embedding model, checked against its pinned SHA-256 hashes; later runs reuse them. All three come to about 1.1 GB.
- **Time:** reranking 50 searches, plus 10 translated ones, at 20 candidates each, takes about half a minute with `mmarco-minilm`, a minute and a half with `gte-multilingual` and four minutes with `bge-m3` on an Apple M2 Max; fewer candidates take less.
- **Reported:** a row per reranking model, "hybrid + model", next to the search modes, the built-in one's marked as gating; how many candidates the reranked searches had (mean, fewest, most); and each model's download and time per search (mean, median, 95th percentile and slowest, after the first search, which loads the model). Only the built-in model's row gates.

### Citation quality

Citation quality needs a chat model. Give one with environment variables:

```sh
INCARNAMIND_EVAL_CHAT_KIND=anthropic \
INCARNAMIND_EVAL_CHAT_MODEL=<the current Claude Sonnet model id> \
INCARNAMIND_EVAL_CHAT_KEY=sk-ant-... \
npm run eval
```

- **Gating:** a cloud model (`anthropic`, `openai`, `google`, or `openai-compatible` with a server elsewhere) is the gating model. Its name is recorded in the report. Use one named model for runs you compare, such as the current Claude Sonnet.
- **Local models:** an Ollama model, or an `openai-compatible` server on this computer, is reported but never gates. For example: `INCARNAMIND_EVAL_CHAT_KIND=ollama INCARNAMIND_EVAL_CHAT_MODEL=llama3.2 npm run eval`. The `ollama` kind talks to Ollama's own `/api/chat`, as the app does; `INCARNAMIND_EVAL_CHAT_NUM_CTX` fixes its window and `INCARNAMIND_EVAL_CHAT_CITING` its citing mode (see the table below), e.g. `INCARNAMIND_EVAL_CHAT_NUM_CTX=8192 INCARNAMIND_EVAL_CHAT_CITING=tools INCARNAMIND_EVAL_MAX_ROUNDS=1`.
- **Consent:** setting the variables is the consent to send Questions and Passages to the chat model. The run declines automatic tagging's consent request, so no Document excerpts are sent for tagging. A local model has no consent step, so it also tags the Documents while the run asks its Questions, which slows the run.
- **Cost:** each Question is asked in a Mind of its own. Round 1 asks all 50. Each later round, up to 3 in all, asks again the gating Questions of any language that still has fewer than 30 Citations. That is between 50 and 130 Answers, each with up to 5 searches. The every-format set's 137 Questions are then asked once each: 137 Answers more.

### Cloud embeddings

Cloud embeddings are reported next to the built-in model and never gate. Give them with environment variables:

```sh
INCARNAMIND_EVAL_EMBED_KIND=openai INCARNAMIND_EVAL_EMBED_MODEL=text-embedding-3-small \
INCARNAMIND_EVAL_EMBED_KEY=sk-... npm run eval
```

- **How:** the run adds the Documents again, to a second temporary data folder. Before adding them, it chooses the provider through the core's public interface (`saveEmbeddingProvider`), as a User would in Settings. The core's own provider code (#32) then embeds the Passages and the searches. With `INCARNAMIND_EVAL_EMBED_BASE_URL`, the provider is an OpenAI-compatible server.
- **Consent:** setting the variables is the consent to send the Documents' text and the searches to the provider. The run accepts the "embeddings" flow's consent request and declines any other.
- **Size:** vectors keep the model's own size (1,536 for `text-embedding-3-small`, 3,072 for `gemini-embedding-001`). The core records the model and size with the vectors.
- **Cost:** embedding the corpus sends about 1,200 Passages, about 0.6 million tokens.

### Environment variables

Keys are read from these variables only, never from `OPENAI_API_KEY` and the like, so a key set for other tools can't make a run spend money.

| Variable | Default | |
|---|---|---|
| `INCARNAMIND_EVAL_CHAT_KIND` | none | `anthropic`, `openai`, `google`, `openai-compatible` or `ollama`. Turns on the Citation part. |
| `INCARNAMIND_EVAL_CHAT_MODEL` | none | The model id. |
| `INCARNAMIND_EVAL_CHAT_KEY` | none | Required for `anthropic`, `openai` and `google`. |
| `INCARNAMIND_EVAL_CHAT_BASE_URL` | none | Required for `openai-compatible`; Ollama's address for `ollama`. |
| `INCARNAMIND_EVAL_CHAT_NUM_CTX` | the app's choice | `ollama` only: the context window (`num_ctx`) every request carries, instead of the one the app chooses from this computer's memory, so local runs are comparable. The output cap follows it, as in the app. |
| `INCARNAMIND_EVAL_CHAT_CITING` | the app's rule | `ollama` only: `tools`, `structured-output` or `none`, how the model cites instead of the app's rule (`citingMode`), to compare the citing modes. |
| `INCARNAMIND_EVAL_EMBED_KIND` | none | `openai` or `google`. Turns on the cloud embedding run. |
| `INCARNAMIND_EVAL_EMBED_MODEL` | none | For example `text-embedding-3-small` or `gemini-embedding-001`. |
| `INCARNAMIND_EVAL_EMBED_KEY` | none | Required with `INCARNAMIND_EVAL_EMBED_KIND`. |
| `INCARNAMIND_EVAL_EMBED_BASE_URL` | none | An OpenAI-compatible server instead of OpenAI (`openai` only). |
| `INCARNAMIND_EVAL_RERANK` | none | `all`, or reranking candidates' ids separated by commas (see Reranking candidates). Adds their reranked modes next to the built-in model's, which always runs. |
| `INCARNAMIND_EVAL_CACHE` | `~/.cache/incarnamind-eval` | Where the built-in models are kept between runs. |
| `INCARNAMIND_EVAL_MIN_CITATIONS` | `30` | Citations each language needs for the Citation targets to count. |
| `INCARNAMIND_EVAL_MAX_ROUNDS` | `3` | Rounds of Questions at most. |
| `INCARNAMIND_EVAL_ANSWER_TIMEOUT_S` | `300` | An Answer that takes longer is stopped and counted as failed. |
| `INCARNAMIND_EVAL_KEEP_DATA` | off | `1` keeps the temporary data folders, for looking into a run. |

### In GitHub Actions

The **Evaluation** workflow (`.github/workflows/eval.yml`) runs `npm run eval` on demand only (Actions → Evaluation → Run workflow). It caches the model and uploads `eval/results/` as an artifact.

Retrieval needs no secrets. For the Citation part, fill in the `chat_kind` and `chat_model` inputs and add the key as the `INCARNAMIND_EVAL_CHAT_KEY` repository secret.

## What it reports

Each run writes a folder `eval/results/<start time>/` (gitignored) with:

- `report.md`: the results, for people.
- `report.json`: everything, including the top 5 of each search and each Answer's sentences, Citations and searches.
- `reviewer-sheet.csv`: only when Citations were scored. One row per "found" quote: the Question, the Answer sentence, the quote, the Document and the page.
- `reviewer-sheet-formats.csv`: the same for the every-format set; its "page" is the cited Unit.

The terminal summary at the end gives the same headline numbers.

### Retrieval

The evaluation set is `retrieval/questions.json`: 40 gating Questions (20 English, 20 Chinese) and 10 cross-lingual ones (8 Chinese Questions about English Documents, 2 English Questions about Chinese Documents).

- **Kinds:** fact lookups, numbers, definitions, Questions worded unlike their Document, and answers in two parts. One English answer runs across a page break (en-17, pages 33–34).
- **Checked against the stored text:** each quote was checked with `findQuote` against the page text the core stores, and against the Passages it builds. Each is on its expected pages, inside at least one Passage that covers them, and on no other page of its Document.
- **No Chinese quote crosses a page:** in the Chinese fixtures, pdf.js puts a page's section headings at the end of its text, so no sentence reads across a page break there.

- **Hit rule (ADR-0009):** a Question is a hit when one of the top 5 Passages from `searchPassages`:
  - belongs to the expected Document;
  - covers the expected pages;
  - contains the expected quote.

  The quote is matched as the Citation check matches quotes, with `findQuote`: both are normalised the same way and compared.
- **Gate:** what the search Tool does by default, "hybrid + mmarco-mMiniLMv2-L12-H384": hybrid search with the built-in embedding model, its candidates reranked by the built-in reranking model. It must find at least 80% of the gating Questions overall and in each language: 32 of 40, and 16 of 20 per language, with today's set. Since 2026-10-09 (#31) this is the shipped default, so the bar holds for what Users get; before, the gate was plain hybrid search.
- **Reported, not gating:** plain hybrid, keyword-only and vector-only search, the cross-lingual Questions, a cloud embedding model if one is given, and the reranked modes of other candidates given.
- **A translated second query:** each cross-lingual Question has a `translatedQuery`, the Question translated by hand into its Document's language. An Answer is told to search again in the Documents' language when the Question is in another one (the search Tool names their languages), and the translation stands in for that second search, without a chat model. The column "with a translated second query" counts a hit when either search's top 5 has one, in hybrid mode and each reranked mode. The translations are written and checked by hand, so this is the most the approach can bring: a model's own translation may find less.
- **Per Question:** the report gives the rank of the first hit in each mode. A rank in brackets is a near miss, between 6 and 20; for a cross-lingual Question, "a / b" is the rank for the Question, then for its translation. For each hybrid miss, it lists what the top 5 were and what each lacked.

### Citation quality

Each Answer is read back from its Mind, as the editor shows it, and split into sentences. Headings and code are left out. The figures are per language: English, Chinese, and the cross-lingual Questions apart.

| Figure | Definition | Target (per language) |
|---|---|---|
| Answers with a Citation | The share of Answers with at least one Citation. | reported |
| Time per Answer | The median time from asking to the Answer's end. | reported |
| Citations | How many Citations the Answers have. | at least 30, for the other targets to count |
| "Quote found" | The share of Citations whose check found the quote on the cited pages. | at least 90% |
| False "not found" | The share of Citations the check marked "not found" whose quote is on the cited pages under a looser normalisation. The looser normalisation keeps only letters and digits, ignoring case, accents, punctuation, spacing and hyphens. It is compared with the stored page text the check read. | at most 5% |
| Other "not found" | Split into: the quote is on other pages of the Document; the quote isn't in the Document (e.g. paraphrased); the cited pages break the page-range rule. | reported |
| "Can't check" | The cited pages have no text. | reported |
| Coverage | The share of sentences with at least one Citation. A Citation just after a sentence's full stop counts for that sentence. Every sentence is treated as drawn from Documents, so this is a lower bound: a sentence saying the Documents don't cover something counts as uncited. | reported |
| Dropped markers and records | From `answer.finished`: markers the model wrote without a valid record, which were removed, and records given for no marker, which were dropped. | reported |
| Support | The share of "found" quotes that support their sentence. A reviewer judges this in `reviewer-sheet.csv`. | at least 80% |

The run gates on the first three targets for a cloud model. Support is judged by hand:

1. Open `reviewer-sheet.csv` in a spreadsheet app. It is UTF-8 with a byte-order mark, so the Chinese text shows correctly.
2. In the `supports (y/n)` column, write `y` when the quote supports the Answer sentence and `n` when it doesn't.
3. Count the `y` rows. The target is at least 80% of the rows in each language.

### Every format

The gating set asks only about text PDFs. The every-format set, `retrieval/formats.json` (#70), asks about Word, PowerPoint, Excel, CSV, Markdown, plain text and PDF Documents, and the hard places in each: 137 Questions over 22 Documents, at least 3 English and 3 Chinese Questions for each format's hard place, and 8 cross-lingual ones (4 each way, each with a `translatedQuery`). Each has one answer, at one Location.

| Format | Documents | Hard places | Questions (English + Chinese) |
|---|---|---|---|
| Word | Riverside Library Renovation Report.docx, 社区食堂试点评估报告.docx, Coastal Flood Risk Review.docx | body text, table cell, footnote, section (only its heading says which, as two sections read alike), comment | 17 + 15 |
| PowerPoint | Coffee Subscription Launch Review.pptx, 新能源公交季度运营汇报.pptx, Quarterly Research Update.pptx | slide text, slide table, chart (title, series and categories from its cached values), speaker notes | 15 + 12 |
| Excel and CSV | Clinic Staffing Plan 2027.xlsx, 门店销售与库存2026.xlsx, Regional Revenue.xlsx, Bike Share Stations September 2026.csv, 小区垃圾分类统计2026年9月.csv, Orders.csv | cell (a value in a sheet's first block, or a labelled value), row with header (a row in a later block, which repeats the header row), second sheet | 12 + 9 |
| Markdown and plain text | Field Kit Setup Guide.md, 实验室数据管理规范.md, Night Shift Handbook.txt, 冷链仓库值班手册.txt | table, list, code block | 10 + 9 |
| PDF | Urban Heat in Six Districts.pdf, 城市绿地与夏季降温调查.pdf, and four image-only scans | table, two columns (one sentence runs from a column's foot onto the next page), footnote, figure caption, scanned | 15 + 15 |

- **Fixtures:** synthetic, Apache-2.0. The 14 in `retrieval/formats/files/` are written by `retrieval/formats/make-fixtures.py` (python-docx, python-pptx, openpyxl, reportlab), the same bytes each time; the others are the repository's own test fixtures, written by other libraries. See `retrieval/formats/ATTRIBUTION.md`.
- **Locations:** `expected.pages` are Unit numbers, as Citations count them (ADR-0011): a PDF's page, a deck's slide, a Word or Markdown section in reading order (a Word file's footnotes are its last Unit), a block of rows (later sheets continue the count), a block of lines.
- **Known gaps:** 12 Questions ask about text the readers don't index today: Word comments (the Word reader leaves them out, though the preview shows them) and scanned pages (no text layer; ADR-0011 leaves text recognition out). They are asked and reported, by hard place, and counted apart from their format's figures.
- **Checked against the stored text:** `tests/eval/formatSet.test.ts` processes each Document as the core does and checks each quote: on its expected Units and no others, citable together, and inside a Passage that covers them. A known gap's quote is on none of its Document's Units.
- **Run:** in a temporary data folder of its own, so the gating set's Documents, Questions and bar are unchanged. Retrieval as for the gating set, with the search Tool's default (hybrid search reranked by the built-in model) and plain hybrid search reported per format and language. With a chat model, each Question is asked once, and its Citations scored with the same figures, per format.
- **Not gating:** no bar is set yet. The proposed bars below are for the User to approve once the first run's numbers are in.

#### Proposed bars per format

To approve after the first run, for each of Word, PowerPoint, Excel and CSV, Markdown and plain text, and PDF, over the Questions that aren't known gaps:

- **Retrieval:** at least 80% with the search Tool's default, overall and in each language, as for the gating set. One Question is 7 to 11 points of a format's language here, so a bar that the first run misses by one Question is evidence of nothing; a format that misses by more has a problem to look into first.
- **Citations, with a cloud model:** the gating set's targets per format: at least 90% "Quote found" and at most 5% false "not found", over at least 20 Citations per format.
- **Known gaps:** reported, never gating, until their text is indexed.

## Citation-check cases

These cases of the Citation check are unit tests, so they run with `npm test` on every push. They are table-driven in `tests/core/citationCheck.test.ts`, which runs `checkCitation` on stored page text in both languages. Where an end-to-end test covers the same case through the core, with a real PDF, it is listed too.

| Case | English | Chinese | End-to-end and related tests |
|---|---|---|---|
| Hyphenation | A word split at a line end is joined. A compound broken after its hyphen keeps the hyphen. The quote may keep the hyphen with a space ("hyper- step"). A soft hyphen at a line end joins the word. | An English term split at a line end inside Chinese text, by a hyphen or a soft hyphen. | `tests/shared/quoteMatch.test.ts` ("matches a word split at a line end…", "matches a hyphenated compound…", "joins a word split by a soft hyphen…") and `tests/shared/text.test.ts` ("line-break hyphenation") |
| Ligatures | ﬁ, ﬂ and ﬃ on the page match plain letters. | A ligature in an English term inside Chinese text. | `tests/shared/quoteMatch.test.ts` ("maps back through characters NFKC changes") |
| Full-width punctuation | Full-width letters, digits and hyphen in the quote. | Full-width colon, comma and full stop, Chinese quotation marks and full-width brackets. | `tests/core/citations.test.ts` ("a Chinese quote across a page break is found, with full-width punctuation and spacing normalised") |
| Quote marks and dashes | Curly and angle quotes, apostrophes, en, em and long dashes. | Corner brackets and a double em dash. | `tests/shared/quoteMatch.test.ts` ("unifies the quote marks and dashes…") |
| Whitespace | Line breaks, tabs, blank lines and non-breaking spaces. | Line breaks and spaces inside Chinese text. | |
| Letter case | A quote starting mid-sentence, given a capital (from the 2026-10-07 run). | An English term in Chinese text, in another case. | `tests/shared/quoteMatch.test.ts` ("ignores letter case…") |
| Greek letters | ε matches the lunate ϵ and the mathematical 𝜖. | ε in Chinese text. | |
| Reference marks | A reference "[36]" written "[^36]" (from the 2026-10-07 run), and the other way round; another number isn't found. | "[2]" written "[^2]". | `tests/core/citations.test.ts` ("a quote that writes the page's reference [36] as [^36] is found…") |
| Ellipsis | Parts on the pages in order are found (from the 2026-10-07 run); a part shorter than 3 words or 15 letters, parts out of order and a reworded part aren't. | A Chinese ellipsis "……"; a part shorter than 15 characters. | `tests/shared/quoteMatch.test.ts` ("a quote with an ellipsis"), `tests/core/citations.test.ts` ("a quote with an ellipsis is found when each part is on the cited page…") |
| CJK text | A Chinese term in English text, whatever the spacing. | Radical look-alikes and the spaces pdf.js adds. Simplified characters don't match traditional ones (ADR-0009). | `tests/core/citations.test.ts` ("a Chinese quote is found in text that has radical look-alikes…") and `tests/shared/quoteMatch.test.ts` |
| A quote across a page break | Citing both pages; a word hyphenated across the break; citing only the first page ("not found"). | Citing both pages. | `tests/core/citations.test.ts` ("a quote across a page break is found once running headers, footers and page numbers are left out", and the Chinese one above), `tests/shared/citations.test.ts` (the viewer's highlight) |
| A range breaking the page-range rule | Three pages; a page outside the cited Passage. | The same. | `tests/core/citations.test.ts` ("The page-range rule") |
| The match is exact | A paraphrase, a quote with one word changed, and a quote from another page than the one cited aren't found. | A paraphrase, and a quote from another page. | `tests/core/citations.test.ts` ("a paraphrased quote is 'not found'") |

`tests/eval/recordedCitations.test.ts` also checks again the Citations a real run recorded (see Results).

### Every Location kind

`tests/eval/formatCitations.test.ts` checks the Citation check on each kind of Location, with Citations to the every-format set's Documents (`tests/fixtures/eval-format-citations.json`), written as models write them before any model was asked the set. Each is checked against the Units processing stores, as the core checks an Answer's Citations, and sorted as the evaluation sorts them.

| Location | Found | Not found, as it should be |
|---|---|---|
| PDF pages | A sentence over four lines of a narrow column; one that runs from a column's foot onto the next page, citing both pages; a table row; a footnote with its number; a figure caption over two lines; Chinese with "°C" for the page's "℃", and a table row the page reads as one line with its neighbours. | The sentence across the page break citing only its first page; a footnote on the page after its own; a paraphrase. |
| Slides | A slide's text; a table row with its cells' tabs read as spaces; a chart's series and title, from its cached values; speaker notes; two slides, the quote in the first one's notes. | Notes cited on the next slide; a paraphrase. |
| Word sections | Body text; a table row; a footnote, in the footnotes' section; a section with its heading; two sections. | A footnote cited at its sentence's section; the north wing's sentence at the south wing's section; a comment (not read). |
| Markdown sections | A table row as written, with its pipes; list items with their number or bullet; lines of a code block. | A code line cited at the next section. |
| Rows of workbooks and CSV files | Rows in a block that repeats the header, with numbers written another way (`1,077` for `1077`, `1453300` for `£1,453,300`, `17%` for `17.0%`); the repeated header with its row; a labelled value; the second sheet's rows; a CSV row across two blocks and the header between them. | A row cited in the wrong block or sheet; two blocks of different sheets (the Location rule); a number changed. |
| Lines of plain text | A table aligned with spaces; a list item with its dash; indented code; the second block. | A line cited in the block before its own; a paraphrase. |

Three quotes a model may well write are on their cited Unit but "not found", false "not found" in the report: a slide's table row written with pipes between its cells, a chart's value written with a thousands separator (`Jun: 1,560` for the cached `1560`; numbers are matched however they are written only in sheets' rows), and a Markdown table row without its pipes. The test keeps them as they are today; whether the check should accept them is for the User to decide.

## Results

### Every format: first run pending

The every-format set was built on 2026-10-09 (#70). Its first `npm run eval` hasn't run yet; its per-format table goes here once it has, with the bars it supports.

### 2026-10-09: reranking, which became the default

Measured at commit `672f4cf` with `INCARNAMIND_EVAL_RERANK=all`, on an Apple M2 Max (12 cores), Node v25.5.0, with other work running on the machine; the 40 + 10 Questions of today's set. The gate was still plain hybrid search then, and failed on English (13 of 20).

| Mode | English | Chinese | Gating set | Cross-lingual | Cross-lingual, with a translated second query |
|---|---|---|---|---|---|
| hybrid | 13/20 | 19/20 | 32/40 | 1/10 | 8/10 |
| keyword | 14/20 | 18/20 | 32/40 | 2/10 | – |
| vector | 11/20 | 20/20 | 31/40 | 1/10 | – |
| **hybrid + mmarco-mMiniLMv2-L12-H384 (now the gate)** | **16/20** | **20/20** | **36/40** | 2/10 | 7/10 |
| hybrid + gte-multilingual-reranker-base | 13/20 | 19/20 | 32/40 | 2/10 | 8/10 |
| hybrid + bge-reranker-v2-m3 | 16/20 | 20/20 | 36/40 | 2/10 | 8/10 |

| Reranking model | Download | Per search: mean | median | 95th percentile |
|---|---|---|---|---|
| mmarco-mMiniLMv2-L12-H384 | 136 MB | 861 ms | 799 ms | 1,460 ms |
| gte-multilingual-reranker-base | 358 MB | 3,371 ms | 3,150 ms | 5,968 ms |
| bge-reranker-v2-m3 | 588 MB | 10,441 ms | 8,710 ms | 17,506 ms |

The reranked searches had 16.1 candidates on average (12 to 20). mmarco-mMiniLMv2-L12-H384 reaches the bar in both languages, as bge-reranker-v2-m3 does at four times the download and twelve times the time, so it is the built-in reranking model, on by default (ADR-0005), and the gate is now its row. It finds en-05, en-12, en-14 and zh-03, which plain hybrid search missed, and loses none; en-03, en-07, en-13 and en-18 are still missed.

Searching again with the Question translated into its Document's language finds 7 or 8 of the 10 cross-lingual Questions, against 1 or 2 without: the search Tool now tells the model which languages the Documents are in, and to do so.

### 2026-10-07: the first 20 Questions

Measured on 2026-10-07 at commit `472d8d8`, on an Apple M2 Max (12 cores) with Node v25.5.0, using the built-in model. Two runs gave the same numbers and ranks. The 12 Documents made 1,199 Passages, added and processed in 43 to 51 s.

| Mode | English | Chinese | Gating set | Cross-lingual |
|---|---|---|---|---|
| hybrid (gating) | 7/10 | 9/10 | **16/20** | 1/5 |
| keyword | 7/10 | 8/10 | 15/20 | 1/5 |
| vector | 5/10 | 10/10 | 15/20 | 1/5 |

**Retrieval still misses the gate, in English.** Hybrid search finds 16 of 20, which meets the overall target, and Chinese passes with 9 of 10. English finds 7 of 10, not 8.

Hybrid search misses en-03, en-05, en-07 and zh-03. Their first hits are at ranks 19, 12, 20 and 16. en-03 is a keyword hit (rank 3) and zh-03 a vector hit (rank 1) that fusion pushes out of the top 5. en-05 ranks 7th with keyword search and 17th with vector search.

The cross-lingual results match ADR-0009's finding: the built-in model favours Documents in the Question's language.

### Before and after the Passage fix

Gating set hits, English + Chinese:

| Mode | Core at `faab607` (1,291 Passages) | Core at `472d8d8` (1,199 Passages) | Prototype (1,204 Passages) |
|---|---|---|---|
| hybrid | 15 (6 + 9) | **16 (7 + 9)** | 17 (8 + 9) |
| keyword | 15 (7 + 8) | 15 (7 + 8) | 14 (6 + 8) |
| vector | 15 (5 + 10) | 15 (5 + 10) | 15 (5 + 10) |

The fix brought back en-01 (rank 7 to 3). en-05 is still missed.

### Compared with the prototype, stage by stage

The prototype (`prototype/retrieval`) was rerun on this machine at 500/200. It gave its recorded numbers and the same rank for every Question. Each stage of the core was then compared with the prototype's on the same input.

- **Passages: the main cause, now fixed.** The core had kept #25's builder and only given it ADR-0009's sizes. That builder ends a Passage at the strongest sentence or paragraph break in its second half, and a page break counts as a paragraph break. The prototype fills a Passage with whole lines and starts the next with the last lines of it, at most 200 tokens of them, as LangChain's splitter does. The core's Passages were shorter (443 approximate tokens on average, against 477), and fewer crossed a page (384, against 599). In the prototype's own harness, with nothing else changed, the core's builder took hybrid search from 17 to 15 of 20 (English from 8 to 6). The core now builds Passages the prototype's way, and counts each line in whole tokens as the prototype did. Given the same page text, its English Passages are the prototype's exactly.
- **Keyword search:** the same terms, stopwords and ranks on the same Passages.
- **Vector search:** the same "query: " and "passage: " prefixes, Document-name prefix, L2-normalised vectors and 50 candidates. The core also normalises the query, which changes nothing. Two differences are deliberate:
  - The core runs onnxruntime-node 1.23.2, the last version with Intel macOS binaries (#26). The prototype ran 1.30.0, through transformers.js.
  - A text over 512 tokens keeps its end-of-text token in the core, which is how Hugging Face's tokenizers and sentence-transformers cut it. The prototype's transformers.js setup cut it off. About one Passage in six is that long.

  On the same runtime, the core and the prototype give identical vectors for texts within 512 tokens.
- **Fusion:** the same. Reciprocal rank fusion takes each list's top 50, with k = 60. The order in which ties are broken was ruled out before (#31).
- **Hit rule:** the same. The core's `findQuote` and the prototype's substring match agree on every Passage and Question.
- **Page text:** the core removes running headers, footers and page numbers (#30) and NUL characters, and joins CJK lines broken inside a paragraph. Dropping the 95 NUL characters in en-05's Document moves the boundaries of 28 of its 32 Passages. Without #30's removal, the core still finds 16 of 20, with 7 English.

**Why English stays at 7.** The English Question that the prototype finds and the core doesn't is en-05. It was the prototype's fifth-ranked hit, on the edge of the top 5. Its keyword rank is the same in both (7th), but its vector rank is 17th in the core against 10th. The vector differences above move it most: on the core's Passages, transformers.js on onnxruntime 1.30 puts it 12th. With onnxruntime 1.30, the prototype's truncation and no header removal all at once, the core puts en-05 7th and still finds 7 English Questions. The rest comes from the page text. Each difference is deliberate or correct, so no honest change meets the English target. ADR-0009 already noted that int8 differences of this size flipped an English Question in the prototype. With 10 Questions per language, one Question is the margin.

**The Citation part** hasn't been run with a cloud model yet, because that needs a paid key.

A smoke run with Ollama's `llama3.2`, a 3B model, ran from start to finish, but the model gave no valid Citations. It wrote 9 markers without records in English, and its Chinese searches were garbled, so the run isn't a measurement. That model is local, so the run didn't gate.

A run with Ollama's `mistral` on 2026-10-07 (local, so not gating) recorded 12 English and 7 Chinese Citations. The check marked 3 of the 12 English ones "not found" though their quotes were on the cited pages (25%, against a target of at most 5%):

- "If the initial hyper-step sizes are too large, …" (Gradient Descent The Ultimate Optimizer, p. 8): the page reads "… we find that if the initial …". The quote starts mid-sentence and the model gave it a capital.
- "… label smoothing of value ϵls = 0.1 [^36]. This hurts perplexity, …" (Attention Is All You Need, p. 8): the page's reference is "[36]", and the model wrote it in the syntax of our Citation markers. The "ϵ" is the page's own.
- "… in some of those tested... Clause 19.1 (18.1) Outcome or Risk Sharing Agreements" (ABPI Code of Practice, pp. 35–36): the model wrote "..." where the page has a full stop and a heading.

The check now ignores letter case, reads "[^36]" as "[36]", and finds a quote with an ellipsis part by part (ADR-0009). `tests/eval/recordedCitations.test.ts` checks the run's Citations again, copied in `tests/fixtures/eval-citations-2026-10-07.json`, against the stored text of their pages, without asking the model again: no false "not found" in either language (English 0 of 12, 8 found; Chinese 0 of 7, 1 found), and the 9 quotes that aren't on their pages (paraphrased, or on other pages) are still "not found".

`tests/eval/citations.test.ts` checks the Citation part through the core with a scripted model instead: found quotes, false "not found", wrong pages, coverage and rounds.

## Files

- **Entry point:**
  - `run.eval.ts`: the run, from start to finish.
  - `vitest.config.ts`: Vitest compiles the TypeScript and bundles the worker threads, as it does for the unit tests. Its config includes only `*.eval.ts`.
- **`lib/`:**
  - `config.ts`: the environment variables.
  - `evaluationSet.ts`: reads and checks `retrieval/questions.json`, or `retrieval/formats.json`.
  - `formats.ts`: the every-format set's figures, per format and per hard place.
  - `library.ts`: the temporary data folder, the core, and the Documents.
  - `embedder.ts` and `embedderWorker.ts`: the built-in model on a worker thread, and cloud embedders.
  - `rerank.ts` and `rerankerWorker.ts`: a reranking candidate on a worker thread, and its download and timings.
  - `retrieval.ts`: the hit rule, searches, reranked modes, the translated second query, and tallies.
  - `citations.ts`: asks Questions, reads Answers, and scores Citations.
  - `report.ts`: the reports and the summary.
- **Evaluation set:**
  - `retrieval/questions.json`: the Questions.
  - `retrieval/fixtures/`: the Chinese Documents, CC BY-SA 4.0; see `ATTRIBUTION.md` there.
  - `retrieval/formats.json`: the every-format set's Questions.
  - `retrieval/formats/`: its Documents (`files/`), the script that writes them, and `ATTRIBUTION.md`.
- **Tests of the evaluation itself:** these run with `npm test`, without the model.
  - `tests/eval/scoring.test.ts` checks how the evaluation scores sentences, Citations, hits and the reviewer sheet.
  - `tests/eval/citations.test.ts` runs the Citation part through a core with a scripted model.
  - `tests/eval/recordedCitations.test.ts` checks the Citations of the 2026-10-07 run again with today's check.
  - `tests/eval/formatSet.test.ts` checks the every-format set against the stored text; `tests/eval/formatScoring.test.ts`, its figures and reports; `tests/eval/formatCitations.test.ts`, the Citation check on every Location kind.

## The grouping check

`npm run eval:grouping` is the check before building the Library's grouping (#51): how Documents are grouped into Topics, on its own fixture set, with the same data folder setup and model. `npm run eval` doesn't run it. See `grouping/README.md`.

## The Organize benchmark

`npm run eval:organize` measures how well Organize puts Documents in the right Folder with the right Tags (ADR-0012), per classifier route, on 80 labelled English and Chinese Documents split into a tuning half and a held-out half. It reads the chat route's model from the same `INCARNAMIND_EVAL_CHAT_*` variables. `npm run eval` doesn't run it. See `organize/README.md` and `organize/RESULTS.md`.
