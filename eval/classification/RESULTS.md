# Local classification results - 2026-10-08

**Tev1 4B and Clef-Flash classified all 26 sample documents correctly with text alone. Tev1 4B took about 40% of Clef's time and 46% of its reported model allocation.** Tev1 0.8B was much faster but left three relevant documents Unsorted rather than assigning a wrong group. Adding PDF pages to Clef did not improve its score and made PDF classification about five times slower.

Tested on an Apple M2 Max with 32 GiB RAM, Ollama 0.40.0, and the MLX/mxfp8 model variants. All inference was local. No application database, API keys, embeddings, or paid model calls were used.

| Input | Correct, all samples | Correct PDFs | Warm median per PDF | Maximum reported model allocation |
| --- | ---: | ---: | ---: | ---: |
| Tev1 0.8B, text | 23/26 | 9/12 | 0.66 s | 2.00 GB |
| Tev1 4B, text | 26/26 | 12/12 | 3.99 s | 7.55 GB |
| Clef-Flash, text | 26/26 | 12/12 | 9.89 s | 16.38 GB |
| Clef-Flash, text + PDF pages | 26/26 | 12/12 | 53.13 s | 19.49 GB |
| Clef-Flash, page images only (diagnostic) | 12/12 | 12/12 | 43.37 s | 16.15 GB |

The first request of each variant is excluded from warm medians. PDF rendering is recorded separately and excluded from model time. Allocation comes from Ollama `/api/ps`; it is not peak process RSS or whole-system memory. These measurements used one development machine: downloads and some validation work overlapped parts of the earlier run. Timings are indicative, not controlled hardware benchmarks.

For all document formats combined, the warm medians were 0.63 seconds for Tev1 0.8B, 3.73 seconds for Tev1 4B, 9.30 seconds for Clef text-only, and 11.09 seconds for Clef with PDF images. That last overall median hides the much larger cost on the twelve PDFs. The 4B run took 96 seconds of classification calls in total, with zero request errors.

Tev1 4B was measured in a follow-up run on the same machine, after its download and validation checks finished. The source hashes, group definitions, titles, extracted excerpts, and prepared page images were verified identical to the earlier inputs. The evaluation fingerprint changed because the 4B variant was added and page images became opt-in; all text variants already explicitly disabled images, so their inference behavior is unchanged. Baseline metadata and input signatures were preserved separately rather than relabelled as a fresh run.

## What changed in the app

- The suggested local model is now `tev1:0.8b`; classification remains an explicit user choice.
- Clef-Flash can receive up to three PDF page images. Enable the page-image checkbox in Library classification settings. It is off by default, including for previously saved connections.
- PDFs with no extracted text can be classified when this option is enabled. This does not add OCR text, search, or citation support.
- DOCX, PPTX, spreadsheets, and text files still use text excerpts. Saved Jev connections and cloud connections remain text-only.
- Previews are bounded, cancellable, checked against the indexed file hash, and not stored by the app. Manual group assignments continue to win over automatic results.
- Tev now shortens an excerpt and retries, at most twice, only when Ollama explicitly rejects its 2,048-token limit. Two such rejections discovered in the initial trial are handled in the final run.

## Interpretation and limits

Tev1 4B is the best measured tradeoff for text grouping on this Mac: it corrected all three 0.8B misses and matched Clef on this sample with substantially lower time and memory use. Select `tev1:4b` in the Library's local model field. Keep Tev1 0.8B for tighter memory budgets or faster background grouping when reviewing Unsorted documents is acceptable. The app's lightweight initial suggestion remains 0.8B; this follow-up did not change saved model settings or the default.

Use Clef's page-image option for PDFs whose visual content is needed. This sample gives no accuracy reason to use Clef instead of Tev1 4B for text-rich documents, or to enable images for every PDF. All three models choose among predefined or user-created groups; this classification path requires no embeddings.

The 26 samples comprise twelve PDFs, twelve English/Chinese Markdown documents, and two Unsorted controls. Expected subject labels and group descriptions were fixed before inference, from the existing evaluation fixtures. These are twelve custom subject groups, not the onboarding starter groups. The models use their production text budgets, so Tev receives a shorter excerpt than Clef.

Clef also classified all twelve PDFs correctly with a neutral filename and no extracted text. These are additional probes of the same PDFs, not twelve new independent samples or a scanned-document corpus. Text remains visible inside the rendered pages, so this demonstrates reading page images; it does not isolate understanding of diagrams or establish accuracy on real-world scans.

All eleven Chinese samples were correct for all three models. The small, clearly titled fixture set cannot establish general multilingual accuracy or performance on users' own group definitions. No descriptions or labels were tuned to improve the scores.

## Per-document outcomes

“Correct” includes correctly choosing Unsorted for the two controls. A dash means no image-only probe was run.

| Sample | Expected group | Tev 0.8B text | Tev 4B text | Clef text | Clef + pages | Images only |
| --- | --- | --- | --- | --- | --- | --- |
| attention-paper | attention | Correct | Correct | Correct | Correct | Correct |
| attention-zh | attention | Correct | Correct | Correct | Correct | Correct |
| gd-paper | gradient-descent | Unsorted | Correct | Correct | Correct | Correct |
| gd-zh | gradient-descent | Correct | Correct | Correct | Correct | Correct |
| lm-gpt3 | language-models | Correct | Correct | Correct | Correct | Correct |
| lm-gpt2 | language-models | Correct | Correct | Correct | Correct | Correct |
| lm-zh | language-models | Correct | Correct | Correct | Correct | Correct |
| sdg-report | sdgs | Unsorted | Correct | Correct | Correct | Correct |
| sdg-zh | sdgs | Correct | Correct | Correct | Correct | Correct |
| esg-report | esg | Correct | Correct | Correct | Correct | Correct |
| esg-zh | esg | Correct | Correct | Correct | Correct | Correct |
| pharma-code | pharma | Unsorted | Correct | Correct | Correct | Correct |
| tea-en | tea | Correct | Correct | Correct | Correct | - |
| tea-zh | tea | Correct | Correct | Correct | Correct | - |
| quake-en | earthquakes | Correct | Correct | Correct | Correct | - |
| quake-zh | earthquakes | Correct | Correct | Correct | Correct | - |
| climate-en | climate | Correct | Correct | Correct | Correct | - |
| climate-zh | climate | Correct | Correct | Correct | Correct | - |
| mars-en | mars | Correct | Correct | Correct | Correct | - |
| mars-zh | mars | Correct | Correct | Correct | Correct | - |
| photo-en | photosynthesis | Correct | Correct | Correct | Correct | - |
| photo-zh | photosynthesis | Correct | Correct | Correct | Correct | - |
| inflation-en | inflation | Correct | Correct | Correct | Correct | - |
| inflation-zh | inflation | Correct | Correct | Correct | Correct | - |
| software-license | Unsorted | Correct | Correct | Correct | Correct | - |
| sales-orders | Unsorted | Correct | Correct | Correct | Correct | - |

## Reproduce and inspect

See [the evaluation README](README.md) for commands and input selection. Per-request predictions, timings, source hashes, selected pages, model digests, and machine metadata are saved locally in `eval/results/classification/`. Those generated files are ignored by Git; this report records the complete outcome matrix.

Installed model digests:

- `clef-flash:latest`: `0a2a05d6a581223a9397e6fd2ea92486d4b1a54ec1125065be0c2170ff97ba75`
- `tev1:0.8b`: `d6e7bb9bfe0feda2548249c5fe9dde6113cf62d4a7eb0f95f5612573c2f2146e`
- `tev1:4b`: `4d24c6f6d61a48d9b72902f4805c5f1da53010736b834b0372377fb4bb7e53d3` (4.36 GB download, MLX/mxfp8)

Validation: the full unit suite passed 1,035 tests before the final default/connection adjustments. The final Library and Jev checks passed 33 tests, the six page-image tests passed again after strengthening the image-routing assertion, and all four Electron Library flows passed. Type checking, lint, production build, and whitespace checks passed.

For the 4B follow-up, input preparation, the real 26-document evaluation, type checking, evaluation-source lint, and whitespace checks passed. Only the evaluation variant and documentation changed. Local follow-up artifacts are `tev-4b-text.json`, `tev-4b-run-info.json`, `tev-4b-input-verification.json`, and `comparison-summary.json`; the earlier run is preserved under `baseline-0.8b-clef/`.
