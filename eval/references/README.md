# Reference check evaluation

`npm run eval:references` checks the 50 references in `references.json` against Crossref itself, as the app checks a reference list (#711): 20 real works, 15 of them journal articles with and without a DOI, and five harder ones (three conference papers without a DOI, an arXiv preprint and a Chinese book); 15 invented ones that look plausible, in English and Chinese; and 15 real ones with a wrong year or author, one by its DOI. It reports precision and recall for "not found" and "doesn't match", and every reference marked otherwise than expected, in `eval/results/<time>-references.md`. It reaches the network, one request a second in Crossref's public pool, so it takes about a minute, and neither `npm test` nor CI runs it.

## Results

### 2026-10-14

| Mark | Precision | Recall |
|---|---|---|
| Not found | 15/17 | 15/15 |
| Doesn't match | 15/16 | 15/15 |

16 of the 20 real references were found. The four that weren't are the limits of Crossref alone:

- "Attention is all you need" (NeurIPS 2017) and "Language models are few-shot learners" (NeurIPS 2020) have no Crossref record of their own: the first matched a 2025 record of the same title, so it "doesn't match" by its year; the second wasn't found.
- 周志华《机器学习》, a Chinese book, wasn't found: Crossref holds few Chinese books and journals.
- Adam, cited by its arXiv number, wasn't checked: its DOI is DataCite's, so it exists, but Crossref has no record to compare.

The app says so where it marks them: a reference not found by searching says that conference papers, preprints and many Chinese journals aren't all in Crossref.
