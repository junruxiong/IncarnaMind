# The Word corpus

Issue #84, for the Word work in #74: the files that #104 (import) and #105 (export back into the original) are checked against. Two sets, committed here with where each file came from and under what licence:

- **`documents/`: 30 realistic .docx files**, in English and Chinese: contracts, memos, numbered clauses, tables, footnotes and endnotes, fields, headers and footers, comments, content controls and tracked changes.
- **`tracked-changes/`: 20 files with someone else's tracked changes**, the kind a User gets back from a counterparty.

`saved/` will hold six files re-saved by hand in Word for Mac, WPS and Google Docs; see [`TO-SAVE.md`](TO-SAVE.md). It does not exist until a person has done that.

The whole corpus is about 1 MB.

## Where the files come from

| Origin | Files | Licence | Notes |
|---|---|---|---|
| `synthetic` | 18 in `documents/synthetic/` | Apache-2.0 (written for this repository) | Invented text and names, written by [`generate-synthetic.py`](generate-synthetic.py) (standard library only; the same script gives the same bytes). All the Chinese files are here, and all the contracts and memos. |
| `libreoffice` | 9 in `documents/`, 18 in `tracked-changes/` | MPL-2.0 | Test documents from [LibreOffice core](https://github.com/LibreOffice/core) (`sw/qa/extras/ooxmlexport/data`, `ooxmlimport/data`), pinned to commit `7094a2f2`. Each file's link is in the table. |
| `apache-poi` | 3 in `documents/`, 2 in `tracked-changes/` | Apache-2.0 | Test documents from [Apache POI](https://github.com/apache/poi) (`test-data/document`), pinned to commit `f3688f45`. |

Files are byte for byte as published (`sha256` in [`manifest.json`](manifest.json)); none is edited.

**Licence caveats, for the owner.**
- LibreOffice publishes its source under MPL-2.0 ([libreoffice.org/about-us/licenses](https://www.libreoffice.org/about-us/licenses/), `COPYING.MPL`). Its test documents carry no licence of their own: they are contributed with bug reports and fall under the repository's licence. That is the best statement available, not a per-file one. MPL-2.0 is file-level copyleft, which is fine for fixtures that are copied unchanged and never linked into the app; they keep their own licence and this README keeps the attribution.
- Apache POI's `legal/LICENSE` is Apache-2.0, and its test files are covered by it in the same way.

**Why not 3GPP change requests.** The issue suggested them. 3GPP's documents state "© 3GPP Organizational Partners (ARIB, ATIS, CCSA, ETSI, TSDSI, TTA, TTC). All rights reserved", and that no part may be reproduced except as authorised by written permission, in all media (the front matter of every specification); [3gpp.org/about-us/legal-matters](https://www.3gpp.org/about-us/legal-matters) sends anyone who wants to reuse its material to its Copyright Form. There is no open licence, so committing them is not allowed here. (3gpp.org also answered a scripted request with HTTP 403 when this was checked.) Instead the 20 tracked-changes files are public test documents that carry other authors' insertions, deletions, moves and formatting changes: 20 of the 45 with tracked changes among the 1,100 .docx files in the two projects' test folders, picked for having no personal names in the authors. If 3GPP ever grants written permission, its change requests can be added as a third set. Ask them first.

**Personal data.** The files carry no e-mail addresses or phone numbers (checked for every file). Some real files carry the names or handles of the people who wrote the test documents, in the tracked changes and the file properties (the last column below lists the change authors). These are the names of open-source contributors as they appear in the public repositories. Files whose authors have a full personal name in either place were left out in favour of files with a handle, "Author" or an empty name. If someone asks for a file to go, remove it.

**What the Chinese coverage is.** None of the real files has Chinese text, so the Chinese files are all synthetic: contracts and memos with Chinese clause numbering (第一条), mixed Chinese and Latin text, a bilingual table, and East Asian fonts and language settings.

## What a feature count is

Per file, the table counts what a round-trip has to keep (`features.ts`): tracked changes (insertions, deletions, moves and formatting changes), footnotes and endnotes (not Word's separators), fields (complex and simple), header and footer parts, comments, tables, list paragraphs, content controls, and images. "Kinds" is how many of those ten the file has at least one of. `manifest.json` holds the same counts and `npm run eval:word` fails when a file no longer has them.

## Running it

```sh
npm run eval:word
```

For each file it runs these steps, and prints one line per file:

1. **package**: reads the file with the app's own ZIP and XML readers, and checks the things Word's repair prompt is about, since LibreOffice repairs silently and tells you nothing: every XML part is well-formed, every relationship points at a part that is there, every part has a content type, and the feature counts match `manifest.json`.
2. **opens in LibreOffice**: converts the file to PDF with `soffice --headless`, using a profile in a temporary folder. If it writes a PDF, the file opened. "Opens" here means LibreOffice loads it; it is not a promise that Word does, which is why a person re-saves some files in Word, WPS and Google Docs.

LibreOffice is found as `INCARNAMIND_SOFFICE`, then `soffice`, then `libreoffice` on the PATH. When it isn't there the second step is skipped and says so, unless `INCARNAMIND_WORD_REQUIRE_LIBREOFFICE=1`, which CI sets: then a missing LibreOffice fails the run.

A file that is known to fail a step is marked `known` in `sources.json` with the reason; it is reported and does not fail the run. Today that is one file (`tdf121176.docx`, with a stray ZIP entry).

**The round-trip.** Once the Word export (#105) exists, its steps are added to `STEPS` in `corpus.word.ts`: import and export each file, then the same LibreOffice open on the result, and whether untouched parts and paragraphs are byte-identical (#84's acceptance criteria, #105's). Each step reports per file and a failure names the file and the step.

CI runs this as the "Word corpus" job (`.github/workflows/ci.yml`), installing LibreOffice Writer on the runner.

## Adding or changing a file

1. Put the file under `documents/`, `tracked-changes/` or `saved/<app>/` and add its line to `sources.json` (origin, language, licence, source URL pinned to a commit; `known` if it fails a check).
2. `INCARNAMIND_WORD_UPDATE=1 npm run eval:word` rewrites `manifest.json` and the table below.
3. Re-run `python3 eval/word/generate-synthetic.py` after changing the generator; commit the files it writes.

## The files

Lang: `en`, `zh`, or `n/a` for a file with little or no prose. Columns after Kinds: tracked changes, footnotes, endnotes, fields, header/footer parts, comments, tables, list paragraphs, content controls, images (`·` is none).

<!-- files:start -->

### Documents (30)

| File | Lang | Licence | Source | Kinds | Tracked | Fn | En | Fields | H/F | Comments | Tables | Lists | Controls | Images | Change authors |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `libreoffice/FDO76312.docx` | en | MPL-2.0 | [libreoffice](https://github.com/LibreOffice/core/blob/7094a2f2374c6b68c4038d9f05019942ba2caa09/sw/qa/extras/ooxmlexport/data/FDO76312.docx) | 6 | · | · | · | 2 | 2 | · | 1 | 5 | 7 | 1 |  |
| `libreoffice/tdf124637_sectionMargin.docx` | en | MPL-2.0 | [libreoffice](https://github.com/LibreOffice/core/blob/7094a2f2374c6b68c4038d9f05019942ba2caa09/sw/qa/extras/ooxmlexport/data/tdf124637_sectionMargin.docx) | 5 | · | 1 | · | 6 | 6 | · | · | 7 | 2 | · |  |
| `libreoffice/tdf103976.docx` | n/a | MPL-2.0 | [libreoffice](https://github.com/LibreOffice/core/blob/7094a2f2374c6b68c4038d9f05019942ba2caa09/sw/qa/extras/ooxmlexport/data/tdf103976.docx) | 4 | · | · | · | 1 | 1 | · | 2 | 2 | · | · |  |
| `libreoffice/fdo69649.docx` | n/a | MPL-2.0 | [libreoffice](https://github.com/LibreOffice/core/blob/7094a2f2374c6b68c4038d9f05019942ba2caa09/sw/qa/extras/ooxmlexport/data/fdo69649.docx) | 3 | · | · | · | 24 | 6 | · | · | · | 1 | · |  |
| `libreoffice/NumberedList.docx` | n/a | MPL-2.0 | [libreoffice](https://github.com/LibreOffice/core/blob/7094a2f2374c6b68c4038d9f05019942ba2caa09/sw/qa/extras/ooxmlexport/data/NumberedList.docx) | 4 | · | · | · | 1 | 1 | · | 2 | 2 | · | · |  |
| `libreoffice/multi-page-toc.docx` | n/a | MPL-2.0 | [libreoffice](https://github.com/LibreOffice/core/blob/7094a2f2374c6b68c4038d9f05019942ba2caa09/sw/qa/extras/ooxmlexport/data/multi-page-toc.docx) | 3 | · | · | · | 5 | 1 | · | · | · | 1 | · |  |
| `libreoffice/tdf126994_lostPageBreak.docx` | n/a | MPL-2.0 | [libreoffice](https://github.com/LibreOffice/core/blob/7094a2f2374c6b68c4038d9f05019942ba2caa09/sw/qa/extras/ooxmlexport/data/tdf126994_lostPageBreak.docx) | 3 | · | · | · | 9 | 1 | · | · | · | 2 | · |  |
| `libreoffice/fdo79008.docx` | en | MPL-2.0 | [libreoffice](https://github.com/LibreOffice/core/blob/7094a2f2374c6b68c4038d9f05019942ba2caa09/sw/qa/extras/ooxmlexport/data/fdo79008.docx) | 3 | · | · | · | 2 | 1 | 1 | · | · | · | · |  |
| `libreoffice/tdf170236.docx` | n/a | MPL-2.0 | [libreoffice](https://github.com/LibreOffice/core/blob/7094a2f2374c6b68c4038d9f05019942ba2caa09/sw/qa/extras/ooxmlimport/data/tdf170236.docx) | 2 | · | · | · | · | 6 | 1 | · | · | · | · |  |
| `apache-poi/form_footnotes.docx` | en | Apache-2.0 | [apache-poi](https://github.com/apache/poi/blob/f3688f45b0de8934fb32736ba14efc5bbca4a580/test-data/document/form_footnotes.docx) | 6 | · | 1 | · | 69 | 1 | · | 6 | 10 | · | 1 |  |
| `apache-poi/Bug54849.docx` | en | Apache-2.0 | [apache-poi](https://github.com/apache/poi/blob/f3688f45b0de8934fb32736ba14efc5bbca4a580/test-data/document/Bug54849.docx) | 5 | · | 1 | 1 | · | 2 | · | 2 | · | 13 | · |  |
| `apache-poi/testComment.docx` | en | Apache-2.0 | [apache-poi](https://github.com/apache/poi/blob/f3688f45b0de8934fb32736ba14efc5bbca4a580/test-data/document/testComment.docx) | 2 | · | · | · | · | · | 1 | · | · | · | 1 |  |
| `synthetic/en-contract-lease.docx` | en | Apache-2.0 | synthetic | 6 | · | 1 | 1 | 4 | 2 | · | 2 | 2 | · | · |  |
| `synthetic/en-contract-nda.docx` | en | Apache-2.0 | synthetic | 4 | · | · | · | 3 | 2 | 3 | · | 11 | · | · |  |
| `synthetic/en-contract-redline.docx` | en | Apache-2.0 | synthetic | 5 | 11 | · | · | 2 | 2 | 1 | · | 9 | · | · | Counterparty Counsel, Reviewer A |
| `synthetic/en-contract-services.docx` | en | Apache-2.0 | synthetic | 7 | · | 1 | 1 | 3 | 2 | · | 2 | 18 | 2 | · |  |
| `synthetic/en-memo-board.docx` | en | Apache-2.0 | synthetic | 5 | 5 | · | · | 2 | 2 | · | 1 | 3 | · | · | Finance Lead |
| `synthetic/en-memo-plain.docx` | en | Apache-2.0 | synthetic | 2 | · | · | · | · | 2 | · | 1 | · | · | · |  |
| `synthetic/en-memo-policy.docx` | en | Apache-2.0 | synthetic | 5 | 4 | 1 | · | 2 | 2 | 1 | · | · | · | · | HR, Legal |
| `synthetic/en-policy-numbered.docx` | en | Apache-2.0 | synthetic | 3 | · | · | · | 2 | 2 | · | · | 7 | · | · |  |
| `synthetic/en-report-quarterly.docx` | en | Apache-2.0 | synthetic | 6 | · | 1 | 1 | 4 | 2 | · | 1 | · | 1 | · |  |
| `synthetic/en-report-sections.docx` | en | Apache-2.0 | synthetic | 3 | · | 1 | · | 1 | 4 | · | · | · | · | · |  |
| `synthetic/zh-bilingual-contract.docx` | zh | Apache-2.0 | synthetic | 6 | · | 1 | · | 2 | 2 | 1 | 1 | · | 1 | · |  |
| `synthetic/zh-contract-redline.docx` | zh | Apache-2.0 | synthetic | 5 | 7 | · | · | 2 | 2 | 1 | · | 6 | · | · | 审核人甲, 对方律师 |
| `synthetic/zh-contract-sale.docx` | zh | Apache-2.0 | synthetic | 5 | · | 1 | · | 2 | 2 | 2 | · | 6 | · | · |  |
| `synthetic/zh-contract-service.docx` | zh | Apache-2.0 | synthetic | 7 | · | 1 | 1 | 2 | 2 | · | 2 | 9 | 2 | · |  |
| `synthetic/zh-memo-meeting.docx` | zh | Apache-2.0 | synthetic | 4 | · | · | · | 2 | 2 | · | 2 | 2 | · | · |  |
| `synthetic/zh-memo-notice.docx` | zh | Apache-2.0 | synthetic | 4 | 2 | · | · | · | 2 | · | 1 | 3 | · | · | 财务负责人 |
| `synthetic/zh-policy-rules.docx` | zh | Apache-2.0 | synthetic | 4 | · | · | · | 2 | 2 | 1 | · | 8 | · | · |  |
| `synthetic/zh-report-annual.docx` | zh | Apache-2.0 | synthetic | 5 | · | 1 | 1 | 4 | 2 | · | 1 | · | · | · |  |

### Files with someone else's tracked changes (20)

| File | Lang | Licence | Source | Kinds | Tracked | Fn | En | Fields | H/F | Comments | Tables | Lists | Controls | Images | Change authors |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `libreoffice/tdf123054.docx` | n/a | MPL-2.0 | [libreoffice](https://github.com/LibreOffice/core/blob/7094a2f2374c6b68c4038d9f05019942ba2caa09/sw/qa/extras/ooxmlexport/data/tdf123054.docx) | 3 | 138 | · | · | · | · | · | 1 | 15 | · | · | Szerző |
| `libreoffice/bnc821804.docx` | en | MPL-2.0 | [libreoffice](https://github.com/LibreOffice/core/blob/7094a2f2374c6b68c4038d9f05019942ba2caa09/sw/qa/extras/ooxmlimport/data/bnc821804.docx) | 1 | 12 | · | · | · | · | · | · | · | · | · | unknown1, unknown10, unknown11, unknown12, unknown2, unknown3, unknown4, unknown5, unknown6, unknown7, unknown8, unknown9 |
| `libreoffice/tdf104797.docx` | en | MPL-2.0 | [libreoffice](https://github.com/LibreOffice/core/blob/7094a2f2374c6b68c4038d9f05019942ba2caa09/sw/qa/extras/ooxmlexport/data/tdf104797.docx) | 1 | 7 | · | · | · | · | · | · | · | · | · | Tekijä |
| `libreoffice/tdf121784.docx` | n/a | MPL-2.0 | [libreoffice](https://github.com/LibreOffice/core/blob/7094a2f2374c6b68c4038d9f05019942ba2caa09/sw/qa/extras/ooxmlexport/data/tdf121784.docx) | 3 | 6 | 1 | 1 | · | · | · | · | · | · | · | User2  |
| `libreoffice/tdf125894.docx` | en | MPL-2.0 | [libreoffice](https://github.com/LibreOffice/core/blob/7094a2f2374c6b68c4038d9f05019942ba2caa09/sw/qa/extras/ooxmlexport/data/tdf125894.docx) | 3 | 4 | · | · | 1 | 6 | · | · | · | · | · | Szerző |
| `libreoffice/tdf116084.docx` | en | MPL-2.0 | [libreoffice](https://github.com/LibreOffice/core/blob/7094a2f2374c6b68c4038d9f05019942ba2caa09/sw/qa/extras/ooxmlexport/data/tdf116084.docx) | 2 | 2 | · | · | · | 6 | · | · | · | · | · | Szerző |
| `libreoffice/tdf121176.docx` | en | MPL-2.0 | [libreoffice](https://github.com/LibreOffice/core/blob/7094a2f2374c6b68c4038d9f05019942ba2caa09/sw/qa/extras/ooxmlexport/data/tdf121176.docx) | 2 | 2 | · | · | · | 6 | · | · | · | · | · | Szerző |
| `libreoffice/tdf127814.docx` | n/a | MPL-2.0 | [libreoffice](https://github.com/LibreOffice/core/blob/7094a2f2374c6b68c4038d9f05019942ba2caa09/sw/qa/extras/ooxmlexport/data/tdf127814.docx) | 5 | 2 | · | · | 1 | 2 | · | 1 | · | 1 | · | Szerző |
| `libreoffice/tdf128156.docx` | n/a | MPL-2.0 | [libreoffice](https://github.com/LibreOffice/core/blob/7094a2f2374c6b68c4038d9f05019942ba2caa09/sw/qa/extras/ooxmlexport/data/tdf128156.docx) | 4 | 2 | · | · | 2 | 9 | · | · | · | 2 | · | Szerző |
| `libreoffice/groupshape-trackedchanges.docx` | n/a | MPL-2.0 | [libreoffice](https://github.com/LibreOffice/core/blob/7094a2f2374c6b68c4038d9f05019942ba2caa09/sw/qa/extras/ooxmlexport/data/groupshape-trackedchanges.docx) | 1 | 4 | · | · | · | · | · | · | · | · | · | vmiklos |
| `libreoffice/fdo65265.docx` | en | MPL-2.0 | [libreoffice](https://github.com/LibreOffice/core/blob/7094a2f2374c6b68c4038d9f05019942ba2caa09/sw/qa/extras/ooxmlexport/data/fdo65265.docx) | 1 | 2 | · | · | · | · | · | · | · | · | · | JL |
| `libreoffice/fdo65632.docx` | n/a | MPL-2.0 | [libreoffice](https://github.com/LibreOffice/core/blob/7094a2f2374c6b68c4038d9f05019942ba2caa09/sw/qa/extras/ooxmlexport/data/fdo65632.docx) | 2 | 1 | 1 | · | · | · | · | · | · | · | · | vmiklos |
| `libreoffice/n789482.docx` | n/a | MPL-2.0 | [libreoffice](https://github.com/LibreOffice/core/blob/7094a2f2374c6b68c4038d9f05019942ba2caa09/sw/qa/extras/ooxmlexport/data/n789482.docx) | 2 | 1 | · | · | 1 | · | · | · | · | · | · | John |
| `libreoffice/fdo79668.docx` | en | MPL-2.0 | [libreoffice](https://github.com/LibreOffice/core/blob/7094a2f2374c6b68c4038d9f05019942ba2caa09/sw/qa/extras/ooxmlexport/data/fdo79668.docx) | 2 | 14 | · | · | · | · | · | · | 4 | · | · | Pashupati |
| `libreoffice/fdo79535.docx` | n/a | MPL-2.0 | [libreoffice](https://github.com/LibreOffice/core/blob/7094a2f2374c6b68c4038d9f05019942ba2caa09/sw/qa/extras/ooxmlexport/data/fdo79535.docx) | 1 | 2 | · | · | · | · | · | · | · | · | · | Pashupati |
| `libreoffice/tdf107969.docx` | n/a | MPL-2.0 | [libreoffice](https://github.com/LibreOffice/core/blob/7094a2f2374c6b68c4038d9f05019942ba2caa09/sw/qa/extras/ooxmlexport/data/tdf107969.docx) | 3 | 6 | 1 | · | · | · | · | · | · | · | 1 | Ooker |
| `libreoffice/n830205.docx` | n/a | MPL-2.0 | [libreoffice](https://github.com/LibreOffice/core/blob/7094a2f2374c6b68c4038d9f05019942ba2caa09/sw/qa/extras/ooxmlexport/data/n830205.docx) | 5 | 1027 | · | · | 1 | 4 | · | 3 | 30 | · | · | JimBob |
| `libreoffice/bnc837302.docx` | n/a | MPL-2.0 | [libreoffice](https://github.com/LibreOffice/core/blob/7094a2f2374c6b68c4038d9f05019942ba2caa09/sw/qa/extras/ooxmlexport/data/bnc837302.docx) | 1 | 1 | · | · | · | · | · | · | · | · | · |  |
| `apache-poi/delins.docx` | en | Apache-2.0 | [apache-poi](https://github.com/apache/poi/blob/f3688f45b0de8934fb32736ba14efc5bbca4a580/test-data/document/delins.docx) | 3 | 82 | · | · | 14 | · | · | · | 19 | · | · | pavel |
| `apache-poi/Tika-792.docx` | n/a | Apache-2.0 | [apache-poi](https://github.com/apache/poi/blob/f3688f45b0de8934fb32736ba14efc5bbca4a580/test-data/document/Tika-792.docx) | 2 | 6 | · | · | · | 6 | · | · | · | · | · | Author |

<!-- files:end -->
