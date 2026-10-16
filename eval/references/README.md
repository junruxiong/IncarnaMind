# Reference check evaluation

`npm run eval:references` checks the 50 references in `references.json` against the services themselves, as the app checks a reference list (#711): Crossref first, OpenAlex when Crossref has no record or only one that differs, and arXiv for a preprint OpenAlex lacks. The set: 20 real works, 15 of them journal articles with and without a DOI, and five harder ones (three conference papers without a DOI, an arXiv preprint and a Chinese book); 15 invented ones that look plausible, in English and Chinese, one with an invented DOI and one with an invented arXiv number; and 15 real ones with a wrong year or author, one by its DOI. An invented reference is caught when it is marked "not found" (the DOI or arXiv number it gives leads nowhere) or "not listed" (nothing like it in Crossref or OpenAlex, though the work may exist). It reports precision and recall for those two together, for "not found" alone and for "doesn't match", how many real references each service found, and every reference marked otherwise than expected, in `eval/results/<time>-references.md`. It reaches the network, one request a second in Crossref's public pool, so it takes about a minute, and neither `npm test` nor CI runs it.

## Results

### 2026-10-16, reading years past pages and CoRR numbers, within a year of the record's

The same as on 2026-10-14: the changes for real papers' lists (a year within one of the record's, a CoRR number read as arXiv's, arXiv's record weighed against OpenAlex's) cost no invented or mismatched reference here, whose wrong years are all two or more years off.

### 2026-10-14, with OpenAlex and arXiv

| Mark | Precision | Recall |
|---|---|---|
| Not found or not listed | 15/16 | 15/15 |
| Not found alone | 2/2 | 2/15 |
| Doesn't match | 15/16 | 15/15 |

18 of the 20 real references were found, two of them in OpenAlex: GPT-3's NeurIPS paper, by its title, and Adam, by its arXiv number's DOI. Two of Crossref's four coverage misses remain:

- "Attention is all you need" (NeurIPS 2017): Crossref and OpenAlex both record it as 2025, so it "doesn't match" by its year. A Crossref record that differs is weighed against OpenAlex's, but OpenAlex's own record of the work, its arXiv DOI's included, is dated 2025 too.
- 周志华《机器学习》, a Chinese book, is in neither: amber, "not in Crossref or OpenAlex".

### 2026-10-14, Crossref alone

| Mark | Precision | Recall |
|---|---|---|
| Not found or not in Crossref | 15/17 | 15/15 |
| Not found alone | 2/2 | 2/15 |
| Doesn't match | 15/16 | 15/15 |

16 of the 20 real references were found; GPT-3's paper and 周志华《机器学习》 were "not in Crossref", "Attention is all you need" didn't match by its year, and Adam wasn't checked.
