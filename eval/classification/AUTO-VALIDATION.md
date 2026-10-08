# Auto classification follow-up — 2026-10-08

The final Auto run produced the expected group for **29 of 29 documents**:
27/27 text-route cases and 2/2 visual-route cases, with no request errors.

The first run scored **28/29**. It rejected the NASA Perseverance poster before
inference because its 3,309 × 5,109 raster (16.9 megapixels) exceeded the original
16-megapixel limit. We raised that bounded preview limit to 20 megapixels so it
also accommodates a 300 dpi A3 scan, then repeated the complete unchanged set.
The 25-megapixel first/second-page rejection regressions still pass. The original
run is preserved under `eval/results/classification-auto/initial-16mp/`.
Expected labels, group descriptions and text-density routing thresholds were
unchanged. This is an implementation iteration on these cases, not a fresh holdout.

| Route | Expected groups | Timing in the final run |
| --- | ---: | --- |
| Tev1 4B, text | 27/27 | 4.48 s median; 4.46 s excluding the first call; 2.94–6.54 s range |
| Clef-Flash, image-only Perseverance poster | 1/1 | 23.59 s inference including switch/load; 1.87 s rendering |
| Clef-Flash, Mars classroom poster + text | 1/1 | 35.99 s inference; 0.96 s rendering |
| Combined Auto | 29/29 | About 185 s including preparation |

No speed or memory fallback occurred in this run. These are wall-clock observations
on the available **Apple M2 Max, 32 GiB RAM**, with Ollama 0.40.0, rather than a
controlled hardware benchmark. The 0.8B model was installed but unused by Auto.
Its accuracy remains the separate [fixed-model comparison](RESULTS.md).

## Inputs and interpretation

The frozen [manifest](auto-set.json) contains 29 cases absent from the earlier
26-document classifier comparison: 23 existing fixtures, five newly downloaded
public PDFs, and a synthetic spreadsheet expected to remain Unsorted. Formats:
13 Markdown, six PowerPoint, four Excel, one CSV and five PDF; 19 English and ten
Chinese documents. All ten Chinese cases passed. The twelve subject groups and
expected labels were fixed before inference; filenames supplied to the model were
neutral (`Document 1`, etc.). Source bytes were SHA-256 checked.

This is a small follow-up integration evaluation, not an independent training
holdout, the user's private document collection, or a general accuracy estimate.
Existing decks are generated fixtures, the subjects are fairly distinct, and three
of the five new PDFs concern Mars. Two visual cases cannot establish scan accuracy.
The routing heuristic uses text density; it does not recognise which figures are
necessary to understand a document.

New PDFs:

- [NASA Perseverance poster](https://science.nasa.gov/resource/mars-2020-perseverance-poster/): no extracted text, routed to images. The initial 16-megapixel cap rejected it before inference. With the revised 20-megapixel cap, its single page rendered and Clef correctly selected Mars without extracted text.
- [NASA Mars classroom poster](https://www.nasa.gov/stem-content/mars-the-red-planet-poster/): sparse first page, routed to images; both pages rendered and Clef correctly selected Mars.
- [NOAA ocean-acidification infographic](https://coralreef.noaa.gov/digital-corals/visual-media/infographics/ocean-acidification): sufficient text, correctly selected Climate change through Tev.
- [USGS Honshu earthquake poster](https://earthquake.usgs.gov/product/poster/20110710/us/1481141108501/poster.pdf): sufficient text despite its many maps, correctly selected Earthquakes through Tev.
- [NASA Surviving on Mars activity book](https://science.nasa.gov/wp-content/uploads/2023/10/Surviving_on_Mars.pdf): sufficient text across its ten pages, correctly selected Mars through Tev.

## Reproduction

Commands and source attribution are in [README.md](README.md). Raw per-document
results, source hashes, hardware and model metadata are in the ignored
`eval/results/classification-auto/` directory. The complete run's fingerprint is
`8bb046d7d19fb52a27a8260f83f29129e7e202dc11020ad0c638f785cfc61399`.
The initial run fingerprint was
`bdc562ce5b7820e8cbcbc0e7ad0edf72a4db69c9fd93224bfbd9cf14f61c03f7`.
Models match the digests in the original report: Tev1 4B `4d24c6f6d61a…`,
Clef-Flash `0a2a05d6a581…` and installed Tev1 0.8B `d6e7bb9bfe0f…`.

## Per-document observations

| Document ID | Expected | Actual | Route | Call time (s) |
| --- | --- | --- | --- | ---: |
| attention-deck | attention | attention | text | 6.54 |
| attention-zh2 | attention | attention | text | 4.72 |
| gd-deck | gradient-descent | gradient-descent | text | 2.94 |
| gd-sgd | gradient-descent | gradient-descent | text | 4.62 |
| lm-zh2 | language-models | language-models | text | 4.00 |
| sdg-deck | sdgs | sdgs | text | 3.07 |
| sdg-zh2 | sdgs | sdgs | text | 4.37 |
| esg-deck | esg | esg | text | 3.03 |
| esg-sri | esg | esg | text | 4.84 |
| pharma-marketing | pharma | pharma | text | 5.92 |
| pharma-zh | pharma | pharma | text | 5.15 |
| tea-deck | tea | tea | text | 3.00 |
| tea-green | tea | tea | text | 5.46 |
| quake-sheet | earthquakes | earthquakes | text | 4.62 |
| quake-zh2 | earthquakes | earthquakes | text | 4.55 |
| climate-sheet | climate | climate | text | 5.08 |
| climate-zh2 | climate | climate | text | 4.43 |
| mars-sheet | mars | mars | text | 4.27 |
| mars-zh2 | mars | mars | text | 4.48 |
| photo-deck | photosynthesis | photosynthesis | text | 3.01 |
| photo-chlorophyll | photosynthesis | photosynthesis | text | 4.94 |
| inflation-sheet | inflation | inflation | text | 4.99 |
| inflation-cpi | inflation | inflation | text | 4.62 |
| noaa-ocean-acidification | climate | climate | text | 3.87 |
| usgs-honshu-poster | earthquakes | earthquakes | text | 4.36 |
| nasa-surviving-mars | mars | mars | text | 4.19 |
| regional-revenue | Unsorted | Unsorted | text | 3.58 |
| nasa-perseverance | mars | mars | visual | 23.59 |
| nasa-red-planet | mars | mars | visual | 35.99 |

## Remaining validation limits

Only this Mac is available. Low-memory selection, explicit out-of-memory errors,
slow warm calls and hard deadlines are covered by deterministic tests; performance
and vision availability on actual 8–16 GiB machines remain unverified. PDF images
are bounded (embedded rasters above 20 megapixels are rejected); Word and PowerPoint images still use text extraction only. Starter
groups and arbitrary user-created overlapping groups need broader field testing.
