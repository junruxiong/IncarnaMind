# Reference check evaluation

`npm run eval:references` checks the 50 references in `references.json` against Crossref itself, as the app checks a reference list (#711): 20 real works, 15 of them journal articles with and without a DOI, and five harder ones (three conference papers without a DOI, an arXiv preprint and a Chinese book); 15 invented ones that look plausible, in English and Chinese, one with an invented DOI and one with an invented arXiv number; and 15 real ones with a wrong year or author, one by its DOI. An invented reference is caught when it is marked "not found" (red: the DOI or arXiv number it gives leads nowhere) or "not in Crossref" (amber: nothing like it in Crossref, though the work may exist). It reports precision and recall for those two together, for red alone and for "doesn't match", and every reference marked otherwise than expected, in `eval/results/<time>-references.md`. It reaches the network, one request a second in Crossref's public pool, so it takes about a minute, and neither `npm test` nor CI runs it.

## Results

### 2026-10-14

| Mark | Precision | Recall |
|---|---|---|
| Not found or not in Crossref | 15/17 | 15/15 |
| Not found (red) alone | 2/2 | 2/15 |
| Doesn't match | 15/16 | 15/15 |

Red went only to the two invented identifiers. 16 of the 20 real references were found; the four that weren't are the limits of Crossref alone:

- "Attention is all you need" (NeurIPS 2017) has no Crossref record of its own: it matched a 2025 record of the same title, so it "doesn't match" by its year.
- "Language models are few-shot learners" (NeurIPS 2020) and 周志华《机器学习》, a Chinese book, are "not in Crossref": amber, with the caveat that Crossref doesn't hold every conference paper, preprint or Chinese journal.
- Adam, cited by its arXiv number, wasn't checked: its DOI is DataCite's, so it exists, but Crossref has no record to compare.
