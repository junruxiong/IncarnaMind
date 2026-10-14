# Reference check evaluation

`npm run eval:references` checks the 50 references in `references.json` against Crossref itself, as the app checks a reference list (#711): 20 real works, with and without a DOI; 15 invented ones that look plausible, in English and Chinese; and 15 real ones with a wrong year or author, one by its DOI. It reports precision and recall for "not found" and "doesn't match", and every reference marked otherwise than expected, in `eval/results/<time>-references.md`. It reaches the network, one request a second in Crossref's public pool, so it takes about a minute, and neither `npm test` nor CI runs it.

## Results

### 2026-10-14: the first run

| Mark | Precision | Recall |
|---|---|---|
| Not found | 15/15 | 15/15 |
| Doesn't match | 15/15 | 15/15 |

All 20 real references were found. The set is of well-known works with whole titles; references with abbreviated or translated titles, or works Crossref doesn't hold (most Chinese journals, books without DOIs), are where it would miss, and the set should grow with them.
