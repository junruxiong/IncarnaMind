# The grouping check

`npm run eval:grouping` decides, on evidence, how Documents are grouped into Topics before the Library is built (#51). It is the "Check before building the grouping" of the Library design (`docs/designs/library-structure-view.md`), with the decision ledger's R0, R1, R2, R3, O5 and O8a.

It is part of the evaluation harness (`eval/README.md`): the same temporary data folder, the built-in model on a worker thread, the same model cache, and a report under `eval/results/`. `npm run eval` doesn't run it, and `npm test` runs only its unit tests.

## What it does

1. **Adds the fixture set** (`set.json`) to a new temporary data folder with `addDocuments`, and processes it with the built-in model, multilingual-e5-small. It never touches the app's real data folder.
2. **Builds three Document vectors** (R3), each Document's mean of its Passage vectors:
   - **names included**: the Passage vectors as stored, which include the Document's name (`passage: <name>\n<text>`), averaged by the vector index's `documentMeans` (R2);
   - **names removed**: that mean with the direction of the name's own vector taken out (one name embedding per Document);
   - **name-free**: every Passage embedded again without the name, for the check only.
3. **Groups each** with the core's grouping functions (`src/core/topics/grouping.ts`): spherical k-means, k-means++ starts, the best of 3 seeded runs, clusters under 3 dissolved, and the placement threshold. k is `round(√(N/2))` clamped to 4–40, raised to the number of subjects (12): the check uses at least as many clusters as subjects.
4. **Chooses the variant on the choosing set** and **reports the pass bars on the held-out set only** (O8a). The most hits on the choosing cases win; within one case of the best, the cheaper variant wins (names included, then names removed, then name-free), since one case is not evidence (ADR-0009).
5. **Runs the incremental cases** with the chosen variant: Documents added after the grouping are placed by the threshold, then the User's corrections and a Regroup.
6. **Measures the classifier fallback** (R0) when a classifier is given.
7. **Times** k-means at 5,000 Documents and the means on the core's thread at 100,000 Passages (O5).
8. **Writes a report** and prints a summary.

## Running it

```sh
npm run eval:grouping
```

It needs no keys and no network once the model is cached (it shares `npm run eval`'s cache, `~/.cache/incarnamind-eval/models/`). A run adds 47 Documents (1,380 Passages), embeds their Passages a second time for the name-free variant, and writes a generated database of about 420 MB to the system's temporary folder for the timing, which it deletes. It takes about 2 minutes on an Apple M2 Max.

The command fails (exit code 1) when a held-out bar is missed, or when a Regroup loses one of the User's changes.

## The fixture set

47 Documents on 12 subjects, grouped together as one library. Documents in `eval/grouping/fixtures/` are new for this check; see its `ATTRIBUTION.md` for sources and licences.

| Subject | English | Chinese | Deck or spreadsheet |
|---|---|---|---|
| Attention and the Transformer | Attention Is All You Need (PDF) | 维基百科-注意力机制 (PDF), 维基百科-Transformer架构 | Transformer architecture (deck) |
| Gradient descent | Gradient Descent The Ultimate Optimizer (PDF), Stochastic gradient descent · Wikipedia | 维基百科-梯度下降法 (PDF) | 梯度下降法讲义 (deck, Chinese) |
| Large language models | Language Models are Few-Shot Learners, Language Models are Unsupervised Multitask Learners (PDFs) | 维基百科-大型语言模型 (PDF), 维基百科-GPT-3 | |
| The UN and the SDGs | United Nations 2022 Annual Report (PDF) | 维基百科-可持续发展目标 (PDF), 维基百科-千年发展目标 | SDGs overview (deck) |
| ESG | JP Morgan 2022 ESG Report (PDF), Socially responsible investing · Wikipedia | 维基百科-环境、社会和公司治理 (PDF) | ESG reporting basics (deck) |
| Medicines and the pharmaceutical industry | ABPI Code of Practice (PDF), Pharmaceutical marketing · Wikipedia | 维基百科-药品 | |
| Tea | Tea · Wikipedia, Green tea · Wikipedia | 茶 · 维基百科 | 茶叶加工流程 (deck, Chinese) |
| Earthquakes | Earthquake · Wikipedia | 维基百科-地震, 维基百科-地震学 | usgs_significant_quakes_2023 (CSV, USGS) |
| Climate change | Climate change · Wikipedia | 维基百科-气候变化, 维基百科-温室气体 | co2_annmean_mlo (XLSX, NOAA) |
| Mars | Mars · Wikipedia | 维基百科-火星, 维基百科-火星探测 | Mars fact sheet (NASA) (XLSX) |
| Photosynthesis | Photosynthesis · Wikipedia, Chlorophyll · Wikipedia | 维基百科-光合作用 | 光合作用课件 (deck, Chinese) |
| Inflation | Inflation · Wikipedia, Consumer price index · Wikipedia | 维基百科-通货膨胀 | CPI-U 2015-2024 (XLSX, BLS) |

The PDFs are the evaluation's own: the seven samples in `data/` and the five Chinese Wikipedia articles in `eval/retrieval/fixtures/`. The tea pair is the app's example Documents (`resources/examples/`). Every subject has at least 3 Documents, so a cluster of one subject isn't dissolved for being under 3.

**Choosing set and held-out set.** The cases are split by subject, so no subject is in both:

| | English–Chinese pairs (both in one Topic) | Decks and spreadsheets (with a Document on their subject) |
|---|---|---|
| Choosing | attention, large language models, ESG, tea, earthquakes, inflation (6) | Transformer architecture, 茶叶加工流程, ESG reporting basics, the USGS CSV, CPI-U (5) |
| Held-out | gradient descent, the UN and the SDGs, climate change, Mars, photosynthesis (5) | 梯度下降法讲义, SDGs overview, the NOAA sheet, the NASA sheet, 光合作用课件 (5) |

Each set mixes a paper or report with a Wikipedia article (harder: different kinds of text) and two Wikipedia articles (easier).

**Shared names (R3).** Two rows: the 17 Documents named "维基百科-…" (11 subjects) and the 12 named "… · Wikipedia" (9 subjects). For each, the report counts the pairs on different subjects that land in one Topic: names pulling Documents together.

**Late arrivals.** The incremental cases leave 8 Documents out of the first grouping, then place each by the threshold: 5 on subjects already grouped (they should join their subject), and the 3 pharmaceutical ones, a subject not grouped yet (they should stay in "Not grouped yet").

**Corrections, then a Regroup.** The User renames the Topic holding "Tea · Wikipedia", moves up to 2 Documents of split pairs next to their partner (or, if none is split, the ABPI Code into the UN report's Topic), and makes a Topic "My data tables" from the USGS CSV and the CPI sheet. A Regroup must keep each: the renamed and frozen Topics keep their ids and Documents, and each moved Document stays where the User put it.

## Pass bars

On the held-out set, with the chosen variant:

- At least 4 of 5 pairs land in the same Topic.
- At least 3 of 5 decks or spreadsheets land with a Document on their subject.
- The founder judges at least 70% of a random 30-Document sample of their own library correctly placed. This needs the User: see below.

If the bars fail, the design's grouping section is revised to its fallback and reviewed again before the build.

## The classifier fallback (R0)

A model proposes the Topic list from the Documents' titles, then a classifier puts each Document in one Topic. It runs only when a chat model is given, with the evaluation's variables (`eval/README.md`): the chat model proposes the list and is measured as a classifier. Clef-Flash and Jev are measured beside it when given:

```sh
# Clef-Flash in Ollama, on this computer (ollama pull clef-flash first; Ollama 0.35.1 or later)
INCARNAMIND_EVAL_CHAT_KIND=ollama INCARNAMIND_EVAL_CHAT_MODEL=qwen3.5:4b \
INCARNAMIND_EVAL_CLEF_URL=http://127.0.0.1:11434 npm run eval:grouping

# Jev, and a cloud chat model
INCARNAMIND_EVAL_CHAT_KIND=anthropic INCARNAMIND_EVAL_CHAT_MODEL=<model id> INCARNAMIND_EVAL_CHAT_KEY=sk-ant-... \
INCARNAMIND_EVAL_JEV_KEY=... npm run eval:grouping
```

- **How:** Clef-Flash and Jev answer the same request at `/v1/systemone`: one yes/no question per Topic ("Is this Document mainly about “…”?"), in one request per Document, and the Topic most probably "yes" wins. The chat model chooses one Topic by name with structured output. Each sees what tagging sees: the Document's name, type and about 1,500 tokens from its beginning.
- **Ranking:** accuracy on the choosing set first; within one case, the faster wins. The report gives the held-out pairs and decks too.
- **Consent:** setting the variables is the consent to send the Documents' titles (to the chat model) and excerpts (to each classifier). A server on this computer receives them; nothing leaves it.
- **Cost:** one request for the list, then one per Document per classifier: 47 each.

## Timing

- **k-means at 5,000 Documents**: 384-dimensional unit vectors generated around 60 directions, k = 40, 3 seeded runs, on one thread as the grouping worker will run it (R1).
- **The means on the core's thread at 100,000 Passages** (O5): a generated database of 2,000 Documents and 100,000 Passages (1,500 characters of text and a 384-number vector each), then `documentMeans` from a fresh index three times. Loading the index plus computing the means must take at most 300 ms (the median of the three); otherwise the design's named fallback applies: the grouping worker reads the vectors itself, through its own read-only connection (WAL). The report also gives the means alone with the index loaded, the usual case after a first search.

## The founder's 30-Document sample

This bar needs the User's own library, so only the User runs it:

1. In your own checkout, on the branch with this check, with dependencies installed (`npm ci`).
2. Run the check on your library folder. It only reads the folder: nothing in it changes, and everything is indexed in a temporary data folder.

   ```sh
   INCARNAMIND_EVAL_GROUPING_FOLDER="/path/to/your/library" INCARNAMIND_EVAL_GROUPING_LIMIT=300 npm run eval:grouping
   ```

   The limit draws 300 files at random (with `INCARNAMIND_EVAL_GROUPING_SEED`, 51 by default). Leave it out to group them all; the built-in model embeds about 25 Passages a second, and the name-free variant embeds them twice.
3. Open `founder-sample.csv` in the new `eval/results/grouping-founder-…/` folder (UTF-8 with a byte-order mark). It has 30 random Documents, each three times, once per grouping, lettered A, B and C. Each row shows the Document's Topic as the Documents nearest the Topic's centre, or "Not grouped yet".
4. In "right Topic? (y/n)", write `y` when the Document belongs with those Documents (for "Not grouped yet": when it fits no Topic), and `n` when it doesn't.
5. Count the `y` rows of each letter, then look up the letters in `report.json` (`letters`). The bar: at least 21 of 30 (70%) for the winning variant.

## Environment variables

Besides the evaluation's own (`eval/README.md`: the chat model, the model cache, keeping the data folder):

| Variable | Default | |
|---|---|---|
| `INCARNAMIND_EVAL_CLEF_URL` | none | Ollama's address, e.g. `http://127.0.0.1:11434`. Measures Clef-Flash. |
| `INCARNAMIND_EVAL_CLEF_MODEL` | `clef-flash` | The model name in Ollama. |
| `INCARNAMIND_EVAL_JEV_KEY` | none | Measures Jev. |
| `INCARNAMIND_EVAL_JEV_URL` | TypeSafe's | A Jev-compatible server instead. |
| `INCARNAMIND_EVAL_JEV_MODEL` | `jev-latest` | |
| `INCARNAMIND_EVAL_GROUPING_FOLDER` | none | The founder's library: groups it and writes the sample, instead of the fixture run. |
| `INCARNAMIND_EVAL_GROUPING_LIMIT` | all | At most this many of its files, drawn at random. |
| `INCARNAMIND_EVAL_GROUPING_SEED` | `51` | Draws the files and the sample. |
| `INCARNAMIND_EVAL_GROUPING_TIMING` | `1` | `0` skips the timings. |

## What it reports

Each run writes `eval/results/grouping-<start time>/` (gitignored), or `grouping-founder-<start time>/` for the founder's library:

- `report.md`: the recommendation; the three variants on both sets, with how many Topics hold both languages; the same at the design's own k (5 for 47 Documents), a diagnostic that isn't a bar; the shared-name rows; the held-out bars with every case and why it missed; the chosen variant's Topics by subject; the incremental cases; the classifier fallback; the timings; the founder's procedure; the fixture table.
- `report.json`: everything.
- `founder-sample.csv`: the founder's sheet.

## Results

Measured on 2026-10-08 at commit `02a5216`, on an Apple M2 Max (12 cores) with Node v25.5.0, with the built-in model. Two runs gave the same groupings; the timings differed by a few percent. No classifier was given, and the founder's sample is still to do.

**The bars fail.** With every Document vector, 0 of the 5 held-out pairs land in one Topic (the bar is 4). Decks and spreadsheets pass with the chosen variant, 4 of 5 (the bar is 3).

| Document vector | Choosing: pairs | Choosing: decks | Held-out: pairs | Held-out: decks | Purity | Topics (not grouped yet) | Topics with both languages |
|---|---|---|---|---|---|---|---|
| **Names included (chosen)** | 0/6 | 3/5 | **0/5** | **4/5** | 35% | 6 (10) | 0 |
| Name's direction removed | 0/6 | 2/5 | 0/5 | 2/5 | 32% | 6 (9) | 0 |
| Name-free | 0/6 | 2/5 | 0/5 | 3/5 | 28% | 5 (11) | 0 |

Names included is chosen with 3 of 11 choosing cases, within a case of the others and the cheapest. Re-embedding every Passage without its name took 48 s for 1,380 Passages; the names took 0.1 s.

**Why: the Topics follow the language.** No Topic of any variant holds both English and Chinese Documents. At the design's own k (5 for 47 Documents, a diagnostic) it is the same: 0 pairs of 5 in either set, and still no Topic with both languages, though decks and spreadsheets rise to 5 of 5 with names included. Every pair misses because its two Documents are in two Topics of one language each, or one isn't grouped. ADR-0009 found the same bias in search: the built-in model favours Documents in the Question's language.

**Names aren't the cause (R3).** Of the 130 pairs of "维基百科-…" Documents on different subjects, 54 share a Topic with names included and 54 with name-free embeddings: they are together for being Chinese, not for their prefix. Removing the name's direction cuts that to 23, but puts all 12 "… · Wikipedia" English Documents in one Topic (63 of 63 pairs, against 5 and 6).

**Incremental cases.** 5 of 8 late arrivals were placed as expected: the 3 pharmaceutical Documents, a new subject, stayed in Not grouped yet, and 2 of 5 others joined their subject. The other 3 missed their Topic's threshold by a little (0.873 to 0.928 against 0.927 to 0.952): every late arrival's similarity to its nearest Topic's centre fell between 0.87 and 0.98, so a 10th-percentile threshold leaves little room. All 7 of the User's changes survived the Regroup: a renamed Topic, a Topic a Document was moved into, a Topic the User made, and 4 moved Documents.

**Timing.**
- k-means at 5,000 Documents (k = 40, 3 runs) took 1.7 to 2.0 s on one thread, on generated vectors that converged in 4 or 5 iterations; real libraries may need more. It belongs on the grouping worker (R1).
- Loading the index and computing the means at 100,000 Passages took 476 ms (median of 472, 476 and 496 ms; 481 ms in the first run), over the 300 ms bar. The means alone, with the index loaded, took 41 ms. So O5's named fallback applies: the grouping worker reads the vectors itself, through its own read-only connection.

**Recommendation.** Don't build the grouping as designed. As the design says, revise its grouping section and review it again before the build. Because the split is by language, the design's first fallback (clustering Passage vectors, from the same model) is unlikely to fix it. R0's classifier fallback doesn't depend on the embedding model's languages, and is the next thing to measure: run this check with a chat model, and with Clef-Flash or Jev if available. Two untested options for the review: an embedding model that aligns languages better (bge-m3 found 4 of 5 cross-lingual Questions in ADR-0009, against 2 for the built-in model), or removing a language direction from the means the way the names' direction is removed here.

## Files

- `grouping.check.ts`: the run; `vitest.config.ts` includes only `*.check.ts` here.
- `set.json` and `fixtures/`: the fixture set.
- `lib/set.ts` reads and checks the set; `lib/variants.ts` builds the three vectors; `lib/scoring.ts` scores and chooses; `lib/incremental.ts` runs the incremental cases; `lib/classifiers.ts` is the fallback; `lib/timing.ts` the timings; `lib/founder.ts` the founder's sheet; `lib/check.ts` puts the decisions together; `lib/report.ts` writes the reports; `lib/config.ts` reads the variables.
- Unit tests, run by `npm test`: `tests/core/topicsGrouping.test.ts` and `tests/core/documentMeans.test.ts` (the core), and `tests/eval/grouping*.test.ts` (the check).
