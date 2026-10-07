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

A run takes about a minute on an Apple M2 Max, most of it spent embedding about 1,300 Passages.

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
- **Cost:** each Question is asked in a Mind of its own. Round 1 asks all 25. Each later round, up to 3 in all, asks again the gating Questions of any language that still has fewer than 30 Citations. That is between 25 and 65 Answers, each with up to 5 searches.

### Cloud embeddings

Cloud embeddings are reported next to the built-in model and never gate. Give them with environment variables:

```sh
INCARNAMIND_EVAL_EMBED_KIND=openai INCARNAMIND_EVAL_EMBED_MODEL=text-embedding-3-small \
INCARNAMIND_EVAL_EMBED_KEY=sk-... npm run eval
```

- **How:** the run adds the Documents again, to a second temporary data folder. Before adding them, it chooses the provider through the core's public interface (`saveEmbeddingProvider`), as a User would in Settings. The core's own provider code (#32) then embeds the Passages and the searches. With `INCARNAMIND_EVAL_EMBED_BASE_URL`, the provider is an OpenAI-compatible server.
- **Consent:** setting the variables is the consent to send the Documents' text and the searches to the provider. The run accepts the "embeddings" flow's consent request and declines any other.
- **Size:** vectors keep the model's own size (1,536 for `text-embedding-3-small`, 3,072 for `gemini-embedding-001`). The core records the model and size with the vectors.
- **Cost:** embedding the corpus sends about 1,300 Passages, about 0.6 million tokens.

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

The evaluation set is `retrieval/questions.json`: 20 gating Questions (10 English, 10 Chinese) and 5 cross-lingual ones (Chinese Questions about English Documents).

- **Hit rule (ADR-0009):** a Question is a hit when one of the top 5 Passages from `searchPassages`:
  - belongs to the expected Document;
  - covers the expected pages;
  - contains the expected quote.

  The quote is matched as the Citation check matches quotes: both are normalised by `normaliseText` and compared with `findQuote`.
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
| Hyphenation | A word split at a line end is joined. A compound broken after its hyphen keeps the hyphen. | An English term split at a line end inside Chinese text. | `tests/shared/quoteMatch.test.ts` ("matches a word split at a line end…", "matches a hyphenated compound…") and `tests/shared/text.test.ts` ("line-break hyphenation") |
| Ligatures | ﬁ, ﬂ and ﬃ on the page match plain letters. | A ligature in an English term inside Chinese text. | `tests/shared/quoteMatch.test.ts` ("maps back through characters NFKC changes") |
| Full-width punctuation | Full-width letters, digits and hyphen in the quote. | Full-width colon, comma and full stop, Chinese quotation marks and full-width brackets. | `tests/core/citations.test.ts` ("a Chinese quote across a page break is found, with full-width punctuation and spacing normalised") |
| CJK text | A Chinese term in English text, whatever the spacing. | Radical look-alikes and the spaces pdf.js adds. Simplified characters don't match traditional ones (ADR-0009). | `tests/core/citations.test.ts` ("a Chinese quote is found in text that has radical look-alikes…") and `tests/shared/quoteMatch.test.ts` |
| A quote across a page break | Citing both pages; a word hyphenated across the break; citing only the first page ("not found"). | Citing both pages. | `tests/core/citations.test.ts` ("a quote across a page break is found once running headers, footers and page numbers are left out", and the Chinese one above), `tests/shared/citations.test.ts` (the viewer's highlight) |
| A range breaking the page-range rule | Three pages; a page outside the cited Passage. | The same. | `tests/core/citations.test.ts` ("The page-range rule") |
| The match is exact | A paraphrase isn't found. | The same. | `tests/core/citations.test.ts` ("a paraphrased quote is 'not found'") |

## Results

Measured on 2026-10-07 at commit `faab607`, on an Apple M2 Max (12 cores) with Node v25.5.0, using the built-in model. Four runs gave the same numbers. The 12 Documents made 1,291 Passages, added and processed in 45 to 75 s.

| Mode | English | Chinese | Gating set | Cross-lingual |
|---|---|---|---|---|
| hybrid (gating) | 6/10 | 9/10 | **15/20** | 1/5 |
| keyword | 7/10 | 8/10 | 15/20 | 1/5 |
| vector | 5/10 | 10/10 | 15/20 | 1/5 |

**Retrieval misses the gate.** Hybrid search finds 15 of 20, not 16. English finds 6 of 10, not 8. Chinese passes with 9 of 10.

Hybrid search misses en-01, en-03, en-05, en-07 and zh-03. Their first hits are at ranks 7, 11, 9, 20 and 15. en-01 and en-03 are keyword hits (ranks 1 and 4) that fusion with vector search pushes out of the top 5. zh-03 is a vector hit at rank 1 that fusion pushes out the same way.

The cross-lingual results match ADR-0009's finding: the built-in model favours Documents in the Question's language.

**Compared with the prototype (ADR-0009):**

- **Vector search:** on the gating Questions it matches the prototype exactly, 5 English and 10 Chinese.
- **Hybrid search:** the prototype found 17 of 20, with 8 English. The core loses en-01 and en-05, which were near misses in the prototype too, at ranks 4 and 5.
- **Vector ranks:** they are a little lower in the core for those two Questions: 15 against 11, and 13 against 10.

Two causes were tried and ruled out, each in a run that was then reverted:

- the order in which reciprocal rank fusion breaks ties: the prototype put the vector list first, the core puts the keyword list first;
- the removal of running headers, footers and page numbers (#30).

The core also builds more Passages than the prototype did: 1,291 against 1,204. So the gap may lie in how Passages are built. That is a follow-up, outside this ticket.

**The Citation part** hasn't been run with a cloud model yet, because that needs a paid key.

A smoke run with Ollama's `llama3.2`, a 3B model, ran from start to finish, but the model gave no valid Citations. It wrote 9 markers without records in English, and its Chinese searches were garbled, so the run isn't a measurement. That model is local, so the run didn't gate.

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
