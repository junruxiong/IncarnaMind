# Evaluation

`npm run eval` measures retrieval and Citation quality against the v1 design's Success Criteria (`docs/designs/v1-product-validation.md`), on the evaluation set the retrieval prototype settled ([ADR-0009](../docs/adr/0009-retrieval-storage-and-search.md)). Issue #31.

It drives the core's public interface in Node, the way the desktop app's UI does:

1. It creates a temporary data folder. It never touches the app's real data folder.
2. It turns embeddings on with the built-in model, as a User can in Settings (they are off by default, ADR-0009), adds the seven sample PDFs in `data/` and the five Chinese Wikipedia articles in `retrieval/fixtures/` with `addDocuments`, and waits until each is processed: text extracted, Passages built and embedded with the real built-in model, multilingual-e5-small.
3. It searches for each Question with `searchPassages` in hybrid, keyword and vector mode, and scores the top 5. A cross-lingual Question with a translated query is searched with that too.
4. It reranks keyword search's top 60 with the built-in reranking model, mmarco-mMiniLMv2-L12-H384, as the search Tool does by default, with embeddings off, and scores the top 5 again: this is the gate ("keyword top 60 + rerank"; see [Keyword + rerank](#keyword--rerank)). Then it reranks what the search Tool hands a reranker with embeddings on (keyword search's top 10 and vector search's top 10, each Passage once) with the same model ("hybrid + rerank", reported only). Other reranking candidates, when given, are scored both ways too. Then the built-in model reranks the candidates of seven other searches, with embeddings off, reported only (see [Other ways to find the candidates](#other-ways-to-find-the-candidates)).
5. When a chat model is given, it turns embeddings off again, as Users have them by default (the vectors stay, unused), asks each Question with `askQuestion` and scores the Citations of each Answer. The core reranks those Answers' searches with the real built-in reranking model, as the app does.
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

A run takes about two minutes on an Apple M2 Max, most of it spent embedding about 1,200 Passages for the hybrid and vector modes, then reranking 75 searches twice (keyword search's top 60, about 1.6 s a search, then hybrid search's candidates, about 0.6 s). The other ways to find the candidates add about seven minutes more, reranking 65 searches each: about a minute for each mode of 20 candidates and two for keyword search's top 40, as a reranker's time grows with its candidates. The every-format set adds 22 small Documents and 145 searches, reranked twice too.

### Reranking candidates

The built-in reranking model, on by default in the app (Settings → Reranking), was chosen from three candidates, all multilingual cross-encoders under Apache-2.0, as int8 ONNX (the built-in one in `src/core/reranking/model.ts`, the other two in `eval/lib/rerankingModels.ts`, because the app doesn't ship them). The built-in one always runs and gates; name others to compare with it, or all of them:

```sh
INCARNAMIND_EVAL_RERANK=all npm run eval
INCARNAMIND_EVAL_RERANK=mmarco-minilm,bge-m3 npm run eval
```

| Id | Model | Download |
|---|---|---|
| `mmarco-minilm` | cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 (built in, on by default) | 136 MB |
| `gte-multilingual` | Alibaba-NLP/gte-multilingual-reranker-base | 358 MB |
| `bge-m3` | BAAI/bge-reranker-v2-m3 | 588 MB |

- **Candidates:** with embeddings off, the default, the search Tool hands its reranker keyword search's top 60 (see [Keyword + rerank](#keyword--rerank)). With them on, it doesn't hand it the fused list: it hands it keyword search's top 10 and vector search's top 10, each Passage once, so a hit only one of them found isn't pushed out by fusion first (#31: en-03 and en-14 were keyword-only hits, en-12 and zh-03 vector-only, and all four fell out of the fused top 5). The evaluation builds the same sets from `searchPassages` in keyword and vector mode, the hybrid one with the Tool's own `topsOfEach`; unit tests check that the Tool's reranker gets exactly those sets. The report gives how many candidates there were per search: up to 60 for keyword + rerank, between 10 and 20 for hybrid + rerank.
- **How:** after the searches, each model in turn, the built-in one first, reranks every Question's candidates (and each translated query's), with the core's own reranking code (`createRerankingModel`: what the model reads, its scores) and the model on a worker thread (`lib/rerankerWorker.ts`), as the app's reranking utility process runs it.
- **Downloads:** the first run downloads each candidate into the model cache, next to the embedding model, checked against its pinned SHA-256 hashes; later runs reuse them. All three come to about 1.1 GB.
- **Time:** reranking 50 searches, plus 10 translated ones (before the 15 paraphrase Questions), at 20 candidates each, takes about half a minute with `mmarco-minilm`, a minute and a half with `gte-multilingual` and four minutes with `bge-m3` on an Apple M2 Max; fewer candidates take less. Keyword + rerank, with up to 60 candidates, takes two to three times as long: 1.6 s a search with `mmarco-minilm` on 2026-10-10.
- **Reported:** two rows per reranking model, "keyword top 60 + model" and "hybrid + model", next to the search modes, the built-in one's keyword row marked as gating; how many candidates each kind of reranked search had (mean, fewest, most); and each model's download and time per search in each mode (mean, median, 95th percentile and slowest, after the first search, which loads the model; each mode opens the model afresh, so its timings are its own). Only the built-in model's keyword row gates.

### Keyword + rerank

What the search Tool does by default since embeddings are off (ADR-0009, 2026-10-10), and so the gate: keyword search's top 60, reranked by the built-in reranking model ("keyword top 60 + model").

- **Candidates:** `searchPassages` in keyword mode, the top 60 (`keywordRerankCandidates` in `SEARCH_TOOL_PARAMETERS`, `src/core/documents/searchTool.ts`, which `KEYWORD_RERANK_DEPTH` in `lib/retrieval.ts` reads), as the search Tool hands its reranker with embeddings off. Fewer when keyword search finds fewer. A cross-lingual Question's translated query is reranked the same way.
- **Why 60:** keyword search's top 20, as many as the most the Tool hands a reranker with embeddings on, found 34 of 40, and missed Questions whose expected Passage keyword search ranks 21st to 60th. Its top 60 found 36 of 40 and 11 of the 15 paraphrase Questions, as hybrid + rerank did, for about twice the reranking time (see the results of 2026-10-10).
- **Hybrid + rerank** ("hybrid + model"), what the search Tool does while the User has embeddings on, is reported next to it, never gating, with the plain search modes: the gap between the two rows, on the gating set and on the paraphrase Questions, is what embeddings add. In the every-format set's table per format, "hybrid" and "hybrid + model: All" are the same comparison.
- **Before:** until 2026-10-09 the gate was plain hybrid search; then hybrid + rerank; on 2026-10-10, with embeddings off by default, keyword search's top 20 reranked, and later that day its top 60. Keyword search's top 20 is still reported, with the other ways to find the candidates.

### Other ways to find the candidates

Keyword search's top 20 reranked, the gate until keyword search's top 60, missed a Question when keyword search ranks its expected Passage below 20: on 2026-10-10, en-12 at 71, zh-03 at 52 and para-en-05 at 58. With embeddings off, the run also hands the built-in reranking model the candidates of seven other searches (`lib/searches.ts`), each a reranked mode of its own, reported next to the gate and never gating. They were built to be compared with keyword search's top 20: each hands the model 20 Passages unless the mode is about more, and the model reorders them by the Question itself. They run on the gating set's library only, and search the Questions themselves, not the translated queries.

| Mode | Candidates | Settings, and why |
|---|---|---|
| keyword top 20, keyword top 40 | Keyword search's top 20 (the gate before its top 60), or top 40. | What the gate's top 60 adds over fewer of keyword search's Passages, which take less time to rerank. |
| keyword + feedback terms | Pseudo-relevance feedback, RM3-style, with no model. Words are drawn from keyword search's top 10 Passages: each word's share of each Passage, weighted by the Passage's share of their BM25 scores for the Question, summed, then times the word's inverse document frequency. Stopwords (the search's own and more function words, in English and Chinese), numbers, words of one character and the Question's own words are left out, so it works on segmented Chinese words as on English ones. The best 10 are added to the Question's words, which keep half the weight. That query is searched with BM25, and its ranking fused with keyword search's own top 20 by reciprocal rank fusion. | 10 Passages and 10 terms, with the Question's words at half the weight: RM3's usual settings (Anserini's defaults). With ~500-token Passages, 10 of them give about 5,000 words to draw from; more terms drift from the Question. The inverse document frequency keeps a library's common words, in a library of a few Documents on one subject, from taking the 10 places. |
| keyword + rewrites | A chat model writes 2 other phrasings of the Question in the Documents' words. Keyword search's top 20 for the Question and for each phrasing, fused. | One short call per Question (`lib/rewrites.ts`). The prompt names the Documents and their languages, as the search Tool tells an Answer, and asks for the words the Documents would use where they answer. |
| keyword + sub-questions | The same, with the Question broken down into at most 3 one-hop questions when it asks several things, compares things or needs several steps, as the old command-line app refined a Question before searching. A Question that needs no breaking down is searched as it is. | One short call per Question. |
| small-to-big | Each Document cut again into sub-chunks of at most 128 approximate tokens, whole lines where they fit, with no overlap (`lib/subChunks.ts`), each mapped to the Passages that hold it (or, if none holds all of it, those it overlaps). BM25 ranks the sub-chunks, and the best 200 score their Passages, each Passage by its best sub-chunk. | The old command-line app searched ~100-token chunks first, then larger ones around them. The sub-chunks are indexed like Passages: the Document's name and then the words, segmented the same way, with the same tokenizer. |
| document first | The 3 Documents that hold most of keyword search's top 20 (each Passage counting 1 / (60 + its rank), as in fusion), then BM25 inside each with that Document's own term statistics, so a word it uses everywhere weighs little there. The Documents' rankings fused, each weighing its score as a share of the best Document's. | The old command-line app's second retrieval searched only inside the Documents its first one picked. Ranking the Documents by their Passages needs no index of its own; one row per Document would weigh a long Document's words against a short one's. |

- **Reached:** for every reranked mode, the gate's and hybrid + rerank's too, the report counts the Questions whose candidates held a Passage that meets the hit rule, before reranking: what any reranker could have found. It tells a search that never finds the Passage from a reranker that doesn't put it in the top 5.
- **Keyword search's rank, to 200:** each Question's expected Passage's rank in plain keyword search, looked for in its top 200 (the most `searchPassages` returns), in the per-Question table and under each of the gate's misses: how many candidates it would take.
- **Time:** per search, finding the candidates and reranking them, apart, in the reranking table and in the section's own. For rewrites and sub-questions, the chat model's call is given apart, as measured when it was made.
- **Chat model calls:** only when a chat model is given with `INCARNAMIND_EVAL_CHAT_*`; without one, the two modes are skipped, and the report says so. Each Question takes one call per mode. The answers are kept in `eval/results/query-rewrites.json`, by model and prompt, so later runs search the same queries and their figures compare; delete the file to ask again. The report gives the calls' time and tokens, and how many were made in the run and how many read from the file. A call that fails skips its mode, saying why.
- **Small-to-big's index:** built in memory, in a database of the evaluation's own, so the app's schema doesn't change. The report gives its size and build time, next to the Passages' own keyword index built the same way. It also counts how often scoring a Passage by its best sub-chunk, and by the sum of its sub-chunks' scores, put the expected Passage among the 20 candidates: only the best sub-chunk is reranked.
- **BM25 in memory:** keyword + feedback terms and document first score with BM25 computed in memory (`lib/corpus.ts`), as FTS5's `bm25()` computes it (k1 1.2, b 0.75), since FTS5 can't weigh one query term more than another, or score Passages by one Document's own term statistics. A unit test checks that it ranks and scores as FTS5 does.
- **Changing the default:** none of these changes the app's search. They were measured to choose what the search Tool does; keyword search's top 60, then one of them, was chosen (see the results of 2026-10-10).

### Citation quality

Citation quality needs a chat model. Give one with environment variables:

```sh
INCARNAMIND_EVAL_CHAT_KIND=anthropic \
INCARNAMIND_EVAL_CHAT_MODEL=<the current Claude Sonnet model id> \
INCARNAMIND_EVAL_CHAT_KEY=sk-ant-... \
npm run eval
```

- **Gating:** a cloud model (`anthropic`, `openai`, `google`, or `openai-compatible` with a server elsewhere) is the gating model. Its name is recorded in the report. Use one named model for runs you compare, such as the current Claude Sonnet.
- **Local models:** an Ollama model, or an `openai-compatible` server on this computer, is reported but never gates. For example: `INCARNAMIND_EVAL_CHAT_KIND=ollama INCARNAMIND_EVAL_CHAT_MODEL=llama3.2 npm run eval`. The `ollama` kind talks to Ollama's own `/api/chat`, as the app does, and cites as the app's rule says: a model under 7 billion parameters with structured output, a larger one that can call Tools in the Tool loop (ADR-0007, #67). `INCARNAMIND_EVAL_CHAT_NUM_CTX` fixes its window and `INCARNAMIND_EVAL_CHAT_CITING` its citing mode (see the table below), so the modes can be compared, e.g. `INCARNAMIND_EVAL_CHAT_NUM_CTX=8192 INCARNAMIND_EVAL_CHAT_CITING=tools INCARNAMIND_EVAL_MAX_ROUNDS=1`.
- **Consent:** setting the variables is the consent to send Questions and Passages to the chat model. The run declines automatic tagging's consent request, so no Document excerpts are sent for tagging. A local model has no consent step, so it also tags the Documents while the run asks its Questions, which slows the run.
- **Cost:** each Question is asked in a Mind of its own. Round 1 asks all 65. Each later round, up to 3 in all, asks again the gating Questions of any language that still has fewer than 30 Citations; the cross-lingual and paraphrase Questions are asked once. That is between 65 and 145 Answers, each with up to 5 searches. The every-format set's 137 Questions are then asked once each: 137 Answers more.
- **Short checks:** `INCARNAMIND_EVAL_QUESTIONS` names the Questions to ask, by id, and `INCARNAMIND_EVAL_FORMATS=off` skips the every-format set, so a local model can be tried on a few Questions in minutes. Retrieval still scores every Question. The Questions named are asked once each, with no further rounds, and the run never gates on Citations, whatever the model: the report and the summary say which Questions were asked. For example, ten gating Questions with a small model in Ollama:

  ```sh
  INCARNAMIND_EVAL_CHAT_KIND=ollama INCARNAMIND_EVAL_CHAT_MODEL=qwen3.5:4b \
  INCARNAMIND_EVAL_CHAT_NUM_CTX=8192 INCARNAMIND_EVAL_FORMATS=off \
  INCARNAMIND_EVAL_QUESTIONS=en-02,en-07,en-08,en-17,en-19,zh-02,zh-04,zh-08,zh-14,zh-17 \
  npm run eval
  ```

  These ten are the gating Questions whose Citations the long PDFs made hardest to get found (`tests/eval/smallModelCitations.test.ts`), each now handled by the engine or the check: an answer on the second page of a two-page Passage (en-02, en-08, zh-02), or across a page break (en-17), which structured output now places; JP Morgan's report, whose stored text lost its f-ligatures (en-07, en-19); pages in traditional characters (zh-02, zh-04, zh-14, zh-17); and a Passage over four pages (zh-08, zh-14). In this library of English and Chinese Documents, each Answer also makes one short call to translate its query and a second search; one whose quotes the check doesn't all find makes one more, for exact quotes, which the "Quotes asked again" rows count and time. Such a run should take about 20 minutes on an Apple M2 Max: about 12 to process and search the Documents, as always, seven of them for the other ways to find the candidates, then the ten Answers, while the model also tags the 12 Documents.

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
| `INCARNAMIND_EVAL_CHAT_KIND` | none | `anthropic`, `openai`, `google`, `openai-compatible` or `ollama`. Turns on the Citation part, and the keyword + rewrites and keyword + sub-questions modes. |
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
| `INCARNAMIND_EVAL_QUESTIONS` | all | Question ids separated by commas, e.g. `en-07,zh-02`, from either set: only these are asked, once each. Retrieval still scores every Question. A run that asks only some never gates on Citations, and its report says so. |
| `INCARNAMIND_EVAL_FORMATS` | `on` | `off` skips the every-format set: its Documents, searches and Questions. |
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

The evaluation set is `retrieval/questions.json`: 40 gating Questions (20 English, 20 Chinese), 10 cross-lingual ones (8 Chinese Questions about English Documents, 2 English Questions about Chinese Documents) and 15 paraphrase ones (8 English, 7 Chinese).

- **Kinds:** fact lookups, numbers, definitions, Questions worded unlike their Document, and answers in two parts. One English answer runs across a page break (en-17, pages 33–34).
- **Paraphrase Questions (`"paraphrase": true`):** each asks for a fact on one page, in words that avoid the target passage's distinctive words: synonyms and descriptions, the kind of Question a person asks without the text in front of them ("How many free trial packs of the same drug may one doctor be given in a twelve-month period?" for "No more than four samples of a particular medicine may be provided to an individual health professional during the course of a year"). Most of the gating Questions reuse their Document's own words, which favours keyword search; these measure what vector search adds when they don't (2026-10-10).
  - **The field:** optional, `true` or `false`; only `true` is kept, and a Question without it isn't one. A paraphrase Question is in its own language: it can't also be `crossLingual`, and has no `translatedQuery`. Their ids are `para-en-NN` and `para-zh-NN`.
  - **Reported apart, never gating:** a column "Paraphrase (English + Chinese)" in the retrieval table (hits in both languages, then in each), "(paraphrase)" after their ids in the per-Question table, and a "Paraphrase" column in the Citation table. They are out of the gating set's counts, its bar and its per-language counts, so those stay comparable with earlier runs, and they aren't asked again in later Citation rounds.
  - **Facts on one page:** each answer was chosen to be on one page only, and the Chinese ones from text in simplified characters, so keyword search misses them for their wording, not for their script (it couldn't match simplified and traditional characters when they were written; it reads them alike since ADR-0009's amendment).
- **Checked against the stored text:** each quote was checked with `findQuote` against the page text the core stores, and against the Passages it builds. Each is on its expected pages, inside at least one Passage that covers them, and on no other page of its Document.
- **No Chinese quote crosses a page:** in the Chinese fixtures, pdf.js puts a page's section headings at the end of its text, so no sentence reads across a page break there.

- **Hit rule (ADR-0009):** a Question is a hit when one of the top 5 Passages from `searchPassages`:
  - belongs to the expected Document;
  - covers the expected pages;
  - contains the expected quote.

  The quote is matched as the Citation check matches quotes, with `findQuote`: both are normalised the same way and compared.
- **Gate:** what the search Tool does by default, "keyword top 60 + mmarco-mMiniLMv2-L12-H384": keyword search's top 60, reranked by the built-in reranking model, with embeddings off. It must find at least 80% of the gating Questions overall and in each language: 32 of 40, and 16 of 20 per language, with today's set. The bar holds for what Users get: since 2026-10-10 (ADR-0009) embeddings are off by default, so this is the shipped default. Earlier that day the gate was keyword search's top 20 reranked; from 2026-10-09 (#31), hybrid search reranked; before that, plain hybrid search.
- **Reported, not gating:** plain hybrid, keyword-only and vector-only search, hybrid + rerank, the other ways to find the candidates, the cross-lingual and paraphrase Questions, a cloud embedding model if one is given, and the reranked modes of other candidates given.
- **A translated second query:** each cross-lingual Question has a `translatedQuery`, the Question translated by hand into its Document's language. An Answer is told to search again in the Documents' language when the Question is in another one (the search Tool names their languages), and the translation stands in for that second search, without a chat model. The column "with a translated second query" counts a hit when either search's top 5 has one, in hybrid mode and each reranked mode. The translations are written and checked by hand, so this is the most the approach can bring: a model's own translation may find less.
- **Per Question:** the report gives the rank of the first hit in each mode. A rank in brackets is a near miss, between 6 and 20; for a cross-lingual Question, "a / b" is the rank for the Question, then for its translation. A last column gives plain keyword search's rank, to 200. For each of the gate's misses, it lists what the top 5 were and what each lacked, where plain keyword search ranks the expected Passage, and what the other searches looked for: the feedback terms, the model's queries, the Documents searched inside.

### Citation quality

Each Answer is read back from its Mind, as the editor shows it, and split into sentences. Headings and code are left out. The figures are per language: English, Chinese, and the cross-lingual and paraphrase Questions apart, each a column of its own that the targets don't apply to.

| Figure | Definition | Target (per language) |
|---|---|---|
| Answers with a Citation | The share of Answers with at least one Citation. | reported |
| Time per Answer | The median time from asking to the Answer's end. | reported |
| Citations | How many Citations the Answers have. | at least 30, for the other targets to count |
| "Quote found" | The share of Citations whose check found the quote on the cited pages. | at least 90% |
| False "not found" | The share of Citations the check marked "not found" whose quote is on the cited pages under a looser normalisation. The looser normalisation keeps only letters and digits, ignoring case, accents, punctuation, spacing and hyphens, reads traditional Chinese characters as simplified ones, as the check does, and reads an f-ligature's letters (fi, fl, ff, ffi, ffl) as one "f": some PDFs' text lost the letters after it, so the page shows "finance" where the stored text reads "fnance" (JP Morgan's ESG report, nearly every such word), and a quote of the page as it shows is the check's miss, not the model's (the check forgives it too in such a Document, ADR-0009). It is compared with the stored page text the check read. | at most 5% |
| Other "not found" | Split into: the quote is on other pages of the Document; the quote isn't in the Document (e.g. paraphrased); the cited pages break the page-range rule. | reported |
| "Can't check" | The cited pages have no text. | reported |
| Coverage | The share of sentences with at least one Citation. A Citation just after a sentence's full stop counts for that sentence. Every sentence is treated as drawn from Documents, so this is a lower bound: a sentence saying the Documents don't cover something counts as uncited. | reported |
| Dropped markers and records | From `answer.finished`: markers the model wrote without a valid record, which were removed, and records given for no marker, which were dropped. | reported |
| Quotes asked again | From `answer.finished`'s `quoteRetry`: a local model answering with structured output whose quotes the check didn't find is asked once more for exact quotes, in one request (ADR-0007). How many Answers made that request; how many records it asked about, and how many it recovered (the check found the new quote, which replaced the old); the time it added to an Answer that made it, the median and the most; and the time all such requests added per Answer, counting those that made none. Cloud models and the Tool loop never make it. | reported |
| Support | The share of "found" quotes that support their sentence. A reviewer judges this in `reviewer-sheet.csv`. | at least 80% |

Below the figures, `report.md` lists why, Citation by Citation and Answer by Answer, for the gating set and per format alike:

- **Citations not found:** each Citation the check didn't find, with its outcome and the check's reason, the pages it cites, its Passage's pages, where its quote is in the Document under the looser normalisation (the first page that holds it, or two consecutive ones; "–" when it isn't there, e.g. paraphrased, or written in other characters), and the quote.
- **Answers without a Citation:** how the model cited, the markers removed and records dropped, what it searched for, and how the Answer begins.

The log prints the same lines as each Answer ends, so a run stopped before its report still says why, with what an Answer's request for exact quotes recovered and how long it took. The terminal summary adds a line per group whose Answers made such requests.

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
- **Known gaps:** 6 Questions ask about text the readers don't index today: scanned pages (no text layer; ADR-0011 leaves text recognition out). They are asked and reported, by hard place, and counted apart from their format's figures. The 6 Word comment Questions were known gaps too until #76, when the Word reader came to read comments into the section their mark is in; they count with Word's figures now.
- **Checked against the stored text:** `tests/eval/formatSet.test.ts` processes each Document as the core does and checks each quote: on its expected Units and no others, citable together, and inside a Passage that covers them. A known gap's quote is on none of its Document's Units.
- **Run:** in a temporary data folder of its own, so the gating set's Documents, Questions and bar are unchanged. Retrieval as for the gating set, with the search Tool's default (keyword search reranked by the built-in model) reported per format and language, and plain hybrid search and hybrid + rerank per format. With a chat model, each Question is asked once, and its Citations scored with the same figures, per format.
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
| Lost ligatures | In a Document whose text lost its f-ligatures ("fnance" where the page shows "finance"), the quote as the page shows it. Where the text keeps them, "flight" isn't found in "fight". | An English term in Chinese text, either way. | `tests/eval/recordedCitations.test.ts` (JP Morgan's report, the only one of the gating set's PDFs whose text lost them), `tests/core/citations.test.ts` ("…lost its f-ligatures…") |
| Full-width punctuation | Full-width letters, digits and hyphen in the quote. | Full-width colon, comma and full stop, Chinese quotation marks and full-width brackets. | `tests/core/citations.test.ts` ("a Chinese quote across a page break is found, with full-width punctuation and spacing normalised") |
| Quote marks and dashes | Curly and angle quotes, apostrophes, en, em and long dashes. | Corner brackets and a double em dash. | `tests/shared/quoteMatch.test.ts` ("unifies the quote marks and dashes…") |
| Whitespace | Line breaks, tabs, blank lines and non-breaking spaces. | Line breaks and spaces inside Chinese text. | |
| Letter case | A quote starting mid-sentence, given a capital (from the 2026-10-07 run). | An English term in Chinese text, in another case. | `tests/shared/quoteMatch.test.ts` ("ignores letter case…") |
| Greek letters | ε matches the lunate ϵ and the mathematical 𝜖. | ε in Chinese text. | |
| Reference marks | A reference "[36]" written "[^36]" (from the 2026-10-07 run), and the other way round; another number isn't found. | "[2]" written "[^2]". | `tests/core/citations.test.ts` ("a quote that writes the page's reference [36] as [^36] is found…") |
| Ellipsis | Parts on the pages in order are found (from the 2026-10-07 run); a part shorter than 3 words or 15 letters, parts out of order and a reworded part aren't. | A Chinese ellipsis "……"; a part shorter than 15 characters. | `tests/shared/quoteMatch.test.ts` ("a quote with an ellipsis"), `tests/core/citations.test.ts` ("a quote with an ellipsis is found when each part is on the cited page…") |
| CJK text | A Chinese term in English text, whatever the spacing. | Radical look-alikes and the spaces pdf.js adds. Simplified characters match traditional ones, character by character, either way; another word for the same thing doesn't (ADR-0009). | `tests/core/citations.test.ts` ("a Chinese quote is found in text that has radical look-alikes…") and `tests/shared/quoteMatch.test.ts` |
| A quote across a page break | Citing both pages; a word hyphenated across the break; citing only the first page ("not found"). | Citing both pages. | `tests/core/citations.test.ts` ("a quote across a page break is found once running headers, footers and page numbers are left out", and the Chinese one above), `tests/shared/citations.test.ts` (the viewer's highlight) |
| A range breaking the page-range rule | Three pages; a page outside the cited Passage. A record that names no page, for a Passage over three pages, cites the page its quote is on instead. | The same. | `tests/core/citations.test.ts` ("The page-range rule") |
| Table rows (#76) | A Markdown row without its pipes; a slide's or a Word file's row, stored with tabs between its cells, quoted with pipes; a cell changed or left out isn't found. | The same, for rows whose cells have no spaces around them once read. | `tests/shared/quoteMatch.test.ts` ("a table's row…"), `tests/core/locations.test.ts`, `tests/renderer/slides.test.ts` (the viewer's highlight) |
| A slide's figures (#76) | A chart's `1560` quoted as `1,560`, only when the quote isn't found as it is; another figure isn't found. | The same. | `tests/shared/numberMatch.test.ts` ("matching numbers in a slide…") |
| The match is exact | A paraphrase, a quote with one word changed, and a quote from another page than the one cited aren't found. | A paraphrase, and a quote from another page. | `tests/core/citations.test.ts` ("a paraphrased quote is 'not found'") |

`tests/eval/recordedCitations.test.ts` also checks again the Citations a real run recorded (see Results).

### Every Location kind

`tests/eval/formatCitations.test.ts` checks the Citation check on each kind of Location, with Citations to the every-format set's Documents (`tests/fixtures/eval-format-citations.json`), written as models write them before any model was asked the set. Each is checked against the Units processing stores, as the core checks an Answer's Citations, and sorted as the evaluation sorts them.

| Location | Found | Not found, as it should be |
|---|---|---|
| PDF pages | A sentence over four lines of a narrow column; one that runs from a column's foot onto the next page, citing both pages; a table row; a footnote with its number; a figure caption over two lines; Chinese with "°C" for the page's "℃", and a table row the page reads as one line with its neighbours. | The sentence across the page break citing only its first page; a footnote on the page after its own; a paraphrase. |
| Slides | A slide's text; a table row with its cells' tabs read as spaces, or with pipes between its cells; a chart's series and title, from its cached values, and a chart's value written with a thousands separator; speaker notes; two slides, the quote in the first one's notes. | Notes cited on the next slide; a paraphrase. |
| Word sections | Body text; a table row; a footnote, in the footnotes' section; a section with its heading; two sections; a comment, in the section its mark is in (#76). | A footnote cited at its sentence's section; the north wing's sentence at the south wing's section; a comment cited at another section, reworded, or with a word left out. |
| Markdown sections | A table row as written, with its pipes, or without them; list items with their number or bullet; lines of a code block. | A code line cited at the next section. |
| Rows of workbooks and CSV files | Rows in a block that repeats the header, with numbers written another way (`1,077` for `1077`, `1453300` for `£1,453,300`, `17%` for `17.0%`); the repeated header with its row; a labelled value; the second sheet's rows; a CSV row across two blocks and the header between them. | A row cited in the wrong block or sheet; two blocks of different sheets (the Location rule); a number changed. |
| Lines of plain text | A table aligned with spaces; a list item with its dash; indented code; the second block. | A line cited in the block before its own; a paraphrase. |

Three quotes a model may well write were on their cited Unit but "not found", false "not found" in the report, until #76: a slide's table row written with pipes between its cells, a chart's value written with a thousands separator (`Jun: 1,560` for the cached `1560`), and a Markdown table row without its pipes. The check finds them now: a pipe that breaks a table's cells reads as a space, in the quote and the text alike, and a slide's numbers are matched however their thousands are written when a quote isn't found as it is (`src/shared/quoteMatch.ts`). A row or a figure whose words or digits aren't on the Unit is still "not found".

## Results

### Every format: first run pending

The every-format set was built on 2026-10-09 (#70). Its first `npm run eval` hasn't run yet; its per-format table goes here once it has, with the bars it supports.

### 2026-10-10: keyword search's top 60, which became the default

Measured on 2026-10-10 with embeddings off by default and the other ways to find the candidates reported, on an Apple M2 Max; the rewrites and sub-questions by Claude Sonnet 5.5. Keyword search's top 20, the gate until then, is from the run earlier that day; it missed the bar in English (15 of 20).

| Mode | English | Chinese | Gating set | Paraphrase | Per search |
|---|---|---|---|---|---|
| **keyword top 60 + mmarco-mMiniLMv2-L12-H384 (the gate from then)** | **16/20** | **20/20** | **36/40** | 11/15 | 1.6 s |
| hybrid + mmarco-mMiniLMv2-L12-H384 (embeddings on) | 16/20 | 20/20 | 36/40 | 11/15 | 0.6 s |
| keyword top 20 + mmarco-mMiniLMv2-L12-H384 (the gate before) | 15/20 | 19/20 | 34/40 | 10/15 | 0.8 s |
| keyword + rewrites + mmarco-mMiniLMv2-L12-H384 | – | – | 38/40 | 12/15 | 0.8 s, and a 2.7 s chat model call |
| keyword + feedback terms, keyword + sub-questions, small-to-big, document first | – | – | 34/40 each | – | about 0.8 s |

Keyword search's top 60 found as many as hybrid + rerank, on the gating set and on the paraphrase Questions, with no embedding model: the cheapest that matched it, so it is what the search Tool hands its reranker with embeddings off (ADR-0009). Rewrites found the most, and 8 of the 10 cross-lingual Questions, but a chat model call before every search costs more than the reranking; in an Answer the model writes its own search queries, and may search again in other words. Feedback terms, sub-questions, small-to-big and document first found no more than keyword search's top 20.

### 2026-10-09: reranking, which became the default

Measured at commit `672f4cf` with `INCARNAMIND_EVAL_RERANK=all`, on an Apple M2 Max (12 cores), Node v25.5.0, with other work running on the machine; the 40 + 10 Questions of today's set. The gate was still plain hybrid search then, and failed on English (13 of 20).

| Mode | English | Chinese | Gating set | Cross-lingual | Cross-lingual, with a translated second query |
|---|---|---|---|---|---|
| hybrid | 13/20 | 19/20 | 32/40 | 1/10 | 8/10 |
| keyword | 14/20 | 18/20 | 32/40 | 2/10 | – |
| vector | 11/20 | 20/20 | 31/40 | 1/10 | – |
| **hybrid + mmarco-mMiniLMv2-L12-H384 (the gate from then until 2026-10-10)** | **16/20** | **20/20** | **36/40** | 2/10 | 7/10 |
| hybrid + gte-multilingual-reranker-base | 13/20 | 19/20 | 32/40 | 2/10 | 8/10 |
| hybrid + bge-reranker-v2-m3 | 16/20 | 20/20 | 36/40 | 2/10 | 8/10 |

| Reranking model | Download | Per search: mean | median | 95th percentile |
|---|---|---|---|---|
| mmarco-mMiniLMv2-L12-H384 | 136 MB | 861 ms | 799 ms | 1,460 ms |
| gte-multilingual-reranker-base | 358 MB | 3,371 ms | 3,150 ms | 5,968 ms |
| bge-reranker-v2-m3 | 588 MB | 10,441 ms | 8,710 ms | 17,506 ms |

The reranked searches had 16.1 candidates on average (12 to 20). mmarco-mMiniLMv2-L12-H384 reaches the bar in both languages, as bge-reranker-v2-m3 does at four times the download and twelve times the time, so it is the built-in reranking model, on by default (ADR-0005), and the gate was its row until embeddings went off by default (2026-10-10). It finds en-05, en-12, en-14 and zh-03, which plain hybrid search missed, and loses none; en-03, en-07, en-13 and en-18 are still missed.

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
  - `tests/eval/smallModelCitations.test.ts` shows, without a model, the ways a small model's Citations of the gating set's long PDFs aren't found (#67): records written by hand, through the core's citation session, over the stored text of their pages.
  - `tests/eval/formatSet.test.ts` checks the every-format set against the stored text; `tests/eval/formatScoring.test.ts`, its figures and reports; `tests/eval/formatCitations.test.ts`, the Citation check on every Location kind.

## The hard tier

`npm run eval:hard` asks harder Questions of a library at scale: public Documents across domains, fetched at evaluation time and never committed, in every retrieval mode with embeddings on, reported per domain and difficulty and never gating. `npm run eval` doesn't run it. See `hard/README.md`.

## The grouping check

The check before building the Library's grouping (#51) found that clustering Documents' vectors into Topics splits them by language, and ADR-0012 replaced Topics with Folders and Tags. It was removed on 2026-10-10 with the Topic code. Its fixture set stays in `grouping/`, because the classification comparison (`classification/`) reads it; `grouping/README.md` gives the set and the check's results.

## The Organize benchmark

`npm run eval:organize` measures how well Organize puts Documents in the right Folder with the right Tags (ADR-0012), per classifier route, on 80 labelled English and Chinese Documents split into a tuning half and a held-out half. It reads the chat route's model from the same `INCARNAMIND_EVAL_CHAT_*` variables. `npm run eval` doesn't run it. See `organize/README.md` and `organize/RESULTS.md`.
