# Local document classification comparison

See the [2026-10-08 measured results](RESULTS.md) for the completed local comparison.

Run `npm run eval:classification` with Ollama on `http://127.0.0.1:11434` and
`tev1:0.8b`, `tev1:4b`, and `clef-flash` already installed. This deliberately does not run in
`npm test`, download models, read credentials or open the application database.

The comparison uses the production decision classifier and PDF preview worker:

- Tev1 0.8B, title and bounded text excerpt.
- Tev1 4B, the same title and bounded text excerpt.
- Clef-Flash, title and bounded text excerpt.
- Clef-Flash, the same text plus up to three PDF page images. Other formats use text.
- A separate diagnostic: Clef-Flash sees only the page images of the same twelve
  PDFs, with a neutral filename and no extracted text. These are not independent
  documents or a representative scanned-document dataset.

The fixed set has 26 documents: all twelve PDFs from `eval/grouping/set.json`,
English/Chinese Markdown pairs for its other six subjects, this repository's
software licence, and the existing orders CSV. The last two should be Unsorted.
The twelve custom subject groups and expected labels are fixed before inference.
Existing fixture provenance is in `eval/grouping/fixtures/ATTRIBUTION.md` and
`eval/retrieval/fixtures/ATTRIBUTION.md`. The filenames are informative; this is a
small integration comparison, not a general accuracy claim or a test of the
starter groups. Group names/descriptions are English. No prompt/label tuning is
performed to improve this set's scores.

Results and temporary prepared inputs go under the ignored
`eval/results/classification/` directory. Each request saves its result, and a
rerun resumes matching results. A fingerprint covers the classifier, transport,
rendering code, evaluation definition and every source file’s SHA-256. Delete a variant's JSON to rerun it.
Errors count as incorrect; the test completing means the measurement completed,
not that a quality threshold was met. Three consecutive request failures stop it.

`CLASSIFICATION_PREPARE_ONLY=1 npm run eval:classification` prepares inputs without
calling a model. `CLASSIFICATION_VARIANTS=tev-text npm run eval:classification`
runs one variant; comma-separated names are also accepted (`tev-4b-text`, `clef-text`,
`clef-images`, `clef-images-only`).

To repeat the 4B comparison alone after installing `tev1:4b`:

```sh
CLASSIFICATION_VARIANTS=tev-4b-text npm run eval:classification
```

Only selected variants need their models installed. Preserve the result directory
before changing the evaluation definition: regenerated summaries include only
reports with the current fingerprint.

Timings are wall time around the actual classification call, including any
context-size retry. Rendering time is recorded separately. The first request
includes model loading; warm medians exclude it. `/api/ps` supplies loaded model
and VRAM allocation sizes, not peak whole-system memory or process RSS. Models
are run sequentially and unloaded after each variant. Ollama's download traffic
may overlap the small-model run; do not treat these as controlled performance
benchmarks.

PDF previews include the opening page and two pages with the most image/path
operators among pages 2–12. Pages are JPEGs with a 1,600-pixel longest edge,
bounded at 2 MB each. Source PDFs are limited to 100 MB, checked against their
indexed SHA-256, and rendered in a worker with a 60-second limit. The app does
not retain previews or OCR text. Classification of a PDF with no extracted text
does not make it searchable or citable. DOCX and PPTX images are not included.
The Library starts with text-only input; PDF page images are an explicit switch
for local Clef-Flash. The local model field initially suggests Tev1 0.8B.

The Library also offers **Auto · local models**, which routes readable text to
Tev1 4B and PDFs with sparse text to Clef-Flash with pages, with a smaller-model
fallback for slow calls or limited memory. Its routing thresholds and limits are
recorded in the [design notes](../../docs/designs/library-structure-view.md#automatic-local-routing--implemented-2026-10-08).
The fixed-model evaluation here measures model choices independently; its scores
are not an accuracy benchmark of the Auto routing heuristic.

API reference: [Ollama Clef-Flash](https://ollama.com/library/clef-flash) accepts
bare base64 `images` at `/v1/systemone`. The app sends these only to the local
Clef-Flash decision connection. [Tev1](https://ollama.com/library/tev1) remains
text-only; explicit 2,048-token rejections shorten the excerpt, at most twice.


## Follow-up Auto validation and packaged smoke

The separate follow-up set in `auto-set.json` has 29 cases not used in the first
classifier comparison: 23 existing public-source excerpts, generated decks and
sheets, five newly downloaded public PDFs, and one synthetic Unsorted control.
The twelve groups and expected labels were fixed before inference. Model-facing
filenames are neutral. Public PDFs are stored only in the ignored results folder;
the manifest pins their URLs and SHA-256 hashes.

```sh
python3 eval/classification/fetch-auto-fixtures.py
CLASSIFICATION_PREPARE_ONLY=1 npx vitest run --config eval/classification/auto.config.ts
npx vitest run --config eval/classification/auto.config.ts
```

This runs the production Auto controller, text extraction and preview worker,
with real installed models and no embeddings or app database. Each complete run
starts fresh to preserve the controller’s timing state; it writes per-document
results immediately under `eval/results/classification-auto`. Errors count as
misses. The report is [AUTO-VALIDATION.md](AUTO-VALIDATION.md). This remains a small
fixture check, not a representative user corpus or an independent training holdout.
Only the available 32 GiB M2 Max was measured; tests of limited-memory and slow-call
fallback are simulations, not measurements on an 8–16 GiB computer.

PDF rendering rejects embedded rasters above 20 million pixels. The worker also
checks PDF.js stream errors, because this version can otherwise resolve a blank
preview after dropping an oversized image. Such files require a lower-resolution
copy or manual grouping. The model receives no preview when rendering fails.

After building the production installer, verify the packaged native renderer:

```sh
npm run build
CSC_IDENTITY_AUTO_DISCOVERY=false npx electron-builder --mac dmg --arm64 --publish never -c.mac.identity=- -c.mac.notarize=false
npx playwright test --config eval/classification/packaged.config.ts
```

The smoke launches `dist/mac-arm64/IncarnaMind.app` (or the path set in
`INCARNAMIND_PACKAGED_EXECUTABLE`), uses a disposable data folder and a local scripted decision
endpoint, and checks image-only PDF extraction, Auto’s visual route, native canvas
JPEG generation and the Library UI. It uses the production build; the test-hook
flag suppresses update checks. The disposable profile uses Chromium’s mock Keychain
so it never needs the user’s stored key; real Keychain integration is outside this
smoke. It does not measure model quality.
