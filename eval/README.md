# Evaluation

`npm run eval` measures retrieval and Citation quality against the v1 design's Success Criteria (`docs/designs/v1-product-validation.md`), on the evaluation set the retrieval prototype settled ([ADR-0009](../docs/adr/0009-retrieval-storage-and-search.md)). Issue #31.

It drives the core's public interface in Node, the way the desktop app's UI does:

1. It creates a temporary data folder. It never touches the app's real data folder.
2. It adds the seven sample PDFs in `data/` and the five Chinese Wikipedia articles in `retrieval/fixtures/` with `addDocuments`, and waits until each is processed: text extracted, Passages built and embedded with the real built-in model, multilingual-e5-small.
3. It searches for each Question with `searchPassages` in hybrid, keyword and vector mode, and scores the top 5.
4. When a chat model is given, it asks each Question with `askQuestion` and scores the Citations of each Answer.
5. It writes a report under `eval/results/` and prints a summary.

The command fails (exit code 1) when a gating target is missed. It isn't part of `npm test` or CI's default run.

## Running it

```sh
npm run eval
```

This runs retrieval only. It needs no keys and no network once the model is cached.

The first run downloads the model, about 135 MB from Hugging Face, into `~/.cache/incarnamind-eval/models/`, outside the repository. Later runs reuse it. The core downloads and checks the files itself, against the SHA-256 hashes pinned in `src/core/embedding/model.ts`: the data folder's `models/` is a link to the cache.

The model runs on a Node worker thread (`lib/embedderWorker.ts`). It uses the same code as the app's embedding utility process: `createOnnxEmbedder`, served over the channel in `src/core/embedding/channel.ts`.

A run takes about a minute on an Apple M2 Max, most of it spent embedding about 1,200 Passages.

### Citation quality

Citation quality needs a chat model. Give one with environment variables:

```sh
INCARNAMIND_EVAL_CHAT_KIND=anthropic \
INCARNAMIND_EVAL_CHAT_MODEL=<the current Claude Sonnet model id> \
INCARNAMIND_EVAL_CHAT_KEY=sk-ant-... \
npm run eval
```

- **Gating:** a cloud model (`anthropic`, `openai`, `google`, or `openai-compatible` with a server elsewhere) is the gating model. Its name is recorded in the report. Use one named model for runs you compare, such as the current Claude Sonnet.
- **Local models:** an Ollama model, or an `openai-compatible` server on this computer, is reported but never gates. For example: `INCARNAMIND_EVAL_CHAT_KIND=ollama INCARNAMIND_EVAL_CHAT_MODEL=llama3.2 npm run eval`.
- **Consent:** setting the variables is the consent to send Questions and Passages to the chat model. The run declines automatic tagging's consent request, so no Document excerpts are sent for tagging. A local model has no consent step, so it also tags the Documents while the run asks its Questions, which slows the run.
- **Cost:** each Question is asked in a Mind of its own. Round 1 asks all 50. Each later round, up to 3 in all, asks again the gating Questions of any language that still has fewer than 30 Citations. That is between 50 and 130 Answers, each with up to 5 searches.

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
| `INCARNAMIND_EVAL_EMBED_KIND` | none | `openai` or `google`. Turns on the cloud embedding run. |
| `INCARNAMIND_EVAL_EMBED_MODEL` | none | For example `text-embedding-3-small` or `gemini-embedding-001`. |
| `INCARNAMIND_EVAL_EMBED_KEY` | none | Required with `INCARNAMIND_EVAL_EMBED_KIND`. |
| `INCARNAMIND_EVAL_EMBED_BASE_URL` | none | An OpenAI-compatible server instead of OpenAI (`openai` only). |
| `INCARNAMIND_EVAL_CACHE` | `~/.cache/incarnamind-eval` | Where the built-in model is kept between runs. |
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
- **Gate:** hybrid search, which is what the search Tool runs, with the built-in model must find at least 16 of 20, and at least 8 of 10 in each language.
- **Reported, not gating:** keyword-only and vector-only search, the cross-lingual Questions, and a cloud embedding model if one is given.
- **Per Question:** the report gives the rank of the first hit in each mode. A rank in brackets is a near miss, between 6 and 20. For each hybrid miss, it lists what the top 5 were and what each lacked.

### Citation quality

Each Answer is read back from its Mind, as the editor shows it, and split into sentences. Headings and code are left out. The figures are per language: English, Chinese, and the cross-lingual Questions apart.

| Figure | Definition | Target (per language) |
|---|---|---|
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

## Results

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
  - `evaluationSet.ts`: reads and checks `retrieval/questions.json`.
  - `library.ts`: the temporary data folder, the core, and the Documents.
  - `embedder.ts` and `embedderWorker.ts`: the built-in model on a worker thread, and cloud embedders.
  - `retrieval.ts`: the hit rule, searches and tallies.
  - `citations.ts`: asks Questions, reads Answers, and scores Citations.
  - `report.ts`: the reports and the summary.
- **Evaluation set:**
  - `retrieval/questions.json`: the Questions.
  - `retrieval/fixtures/`: the Chinese Documents, CC BY-SA 4.0; see `ATTRIBUTION.md` there.
- **Tests of the evaluation itself:** these run with `npm test`, without the model.
  - `tests/eval/scoring.test.ts` checks how the evaluation scores sentences, Citations, hits and the reviewer sheet.
  - `tests/eval/citations.test.ts` runs the Citation part through a core with a scripted model.
  - `tests/eval/recordedCitations.test.ts` checks the Citations of the 2026-10-07 run again with today's check.
