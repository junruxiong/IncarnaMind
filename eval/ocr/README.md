# Reading scanned pages: the OCR comparison

`npm run eval:ocr` compares the ways IncarnaMind could read a scanned page's text on the User's computer (#397), on a test set of scanned pages, and times the app's own path. It chose PaddleOCR's PP-OCRv5 mobile models on onnxruntime-node (ADR-0011's amendment of 2026-10-12).

## The set

`tests/fixtures/scans/`: eight A4 pages drawn at 200 dpi by `make-scans.ts` from synthetic text, each with its expected text, line by line, in `set.json`. Six were drawn before any engine was tuned; two (`heldout`) after, to check that the tuning holds.

| Page | Language | What it is |
|---|---|---|
| `en-clean` | English | A contract's payment and termination clauses, Times New Roman |
| `en-noisy` | English | A field report in Georgia, rotated 1.3°, blurred, noise, uneven light, heavy JPEG |
| `zh-clean` | Simplified Chinese | A lease's clauses, in a heavy Song face |
| `zh-noisy` | Simplified Chinese | An operations report in Hiragino Sans GB, rotated −1.1°, blurred, noise, heavy JPEG |
| `table` | English and Chinese | Two ruled tables of figures |
| `zh-hant` | Traditional Chinese | A short notice |
| `en-small` (held out) | English | A memo in 9 pt Arial, rotated 0.6°, light noise |
| `zh-light` (held out) | Simplified Chinese | A meeting notice in 10 pt STHeiti Light, rotated −0.7°, light noise |

`read.json` is what the chosen model read on each page, with each line's box: the smoke tests' stand-in for the model reads it, so a quote is washed where the real model puts it (`INCARNAMIND_OCR_RECORD=1 npm run eval:ocr` writes it again).

## What it measures

- **Character error rate (CER):** edits to turn what was read into the expected text, per expected character, both normalised as the Citation check normalises them. Lines are compared in order, so reading a table's columns instead of its rows counts against an engine.
- **Quotes found:** quotes of 8 words (English) or 12 characters (Chinese) starting at each line and running on into the next, and each table row, looked for in what was read with the Citation check's own matcher (`findQuote`). This is what a Citation of a scanned page needs.
- **Time a page:** the faster of two reads, the first read of the run not timed. PaddleOCR runs as the app runs it, with half the cores (at most four).

Every engine reads English, Simplified and Traditional Chinese at once, as the app must: it doesn't know a scan's language before reading it. Windows.Media.Ocr reads one language at a time, so it gets each page's own language: a favour the app couldn't give it.

## Results, 2026-10-12

Apple M2 Max, macOS 26, under heavy load from other work (load average 25–80 on 12 cores): times are slower than on an idle machine, and only comparable with each other. Windows.Media.Ocr ran on GitHub's `windows-latest` runner (Windows Server 2025), with the Simplified and Traditional Chinese OCR packs installed, reading each page with each language's engine; its readings were scored here.

| Engine | All | English | Chinese | Noisy | Table | Held out | Quotes found | Time a page | What it adds |
|---|---|---|---|---|---|---|---|---|---|
| **PP-OCRv5 mobile** (chosen) | **0.4%** | 0.4% | 0.5% | 0.5% | 0.3% | 0.8% | **79/91** | 0.94 s | 21.5 MB download |
| PP-OCRv5 server | 1.2% | 2.3% | 0.5% | 2.1% | 0.3% | 2.0% | 71/91 | 14.4 s | 173 MB download |
| PP-OCRv4 mobile | 7.7% | 13.7% | 4.1% | 9.3% | 4.1% | 9.4% | 41/91 | 1.28 s | 15.6 MB download |
| macOS Vision (accurate) | 1.1% | 1.0% | 0.1% | 0.5% | 5.3% | 0.3% | 76/91 | 0.81 s | a Swift helper; macOS only |
| Windows.Media.Ocr, each page's own language | 8.6% | 0.2% | 1.4% | 0.7% | 62.9% | 1.1% | 64/91 | 0.21 s\* | a helper; Windows only, and each language's OCR pack |
| Windows.Media.Ocr, as set up in English | 57.9% | 0.2% | 99.9% | 50.0% | 62.9% | 50.0% | 38/91 | 0.21 s\* | the same |
| Tesseract 5, tessdata_best | 8.9% | 0.2% | 17.1% | 6.3% | 1.9% | 4.2% | 66/91 | 1.74 s | 41.5 MB of models, and tesseract.js |
| Tesseract 5, tessdata_fast | 10.7% | 0.6% | 20.6% | 9.7% | 1.6% | 8.4% | 64/91 | 1.15 s | 8.9 MB of models, and tesseract.js |

\* On GitHub's Windows runner, a different and idle machine: not comparable with the other times. Windows.Media.Ocr gives its lines as it finds them, with no reading order of ours: it reads the tables' columns one after the other (62.9% on `table`); the app could put its words in order as it does Vision's. Without the table it had 0.9% of characters wrong and 63 of 78 quotes, given each page's language. Set up in English, as most Windows computers outside China are, it reads no Chinese at all; with the Chinese pack only, English drops to 4.3% wrong and 39 of 91 quotes.

The three PaddleOCR models are read the same way, each line at 15 pixels (see below); read at full size, PP-OCRv5 server had 2.0% of the six first pages' characters wrong, and 55 of 73 quotes.

The app's own path, the set as one scanned PDF processed as a Document is (pdf.js finds and draws each page at 200 dpi, the model reads it, its Unit and Passages are built): 8 pages in 8.7 s, 1.09 s a page with the drawing, and 81 of 91 quotes found in the stored text.

Per page (CER, quotes found, time):

| Engine | en-clean | en-noisy | zh-clean | zh-noisy | table | zh-hant | en-small | zh-light |
|---|---|---|---|---|---|---|---|---|
| PP-OCRv5 mobile | 0.4%, 14/17 | 0.3%, 13/15 | 0.7%, 11/13 | 0.0%, 11/11 | 0.3%, 12/13 | 0.0%, 4/4 | 0.5%, 7/10 | 1.1%, 7/8 |
| PP-OCRv5 server | 0.1%, 17/17 | 4.6%, 1/15 | 0.3%, 13/13 | 0.0%, 11/11 | 0.3%, 12/13 | 0.0%, 4/4 | 2.3%, 6/10 | 1.7%, 7/8 |
| PP-OCRv4 mobile | 8.0%, 4/17 | 16.0%, 0/15 | 1.0%, 13/13 | 2.3%, 10/11 | 4.1%, 6/13 | 11.3%, 1/4 | 17.1%, 0/10 | 1.7%, 7/8 |
| macOS Vision | 1.5%, 12/17 | 0.8%, 12/15 | 0.0%, 13/13 | 0.5%, 11/11 | 5.3%, 8/13 | 0.0%, 4/4 | 0.6%, 8/10 | 0.0%, 8/8 |
| Tesseract, best | 0.0%, 17/17 | 0.1%, 15/15 | 14.1%, 7/13 | 16.7%, 4/11 | 1.9%, 9/13 | 29.9%, 0/4 | 0.5%, 7/10 | 7.9%, 7/8 |
| Tesseract, fast | 0.0%, 17/17 | 0.8%, 13/15 | 15.4%, 7/13 | 21.2%, 5/11 | 1.6%, 10/13 | 29.9%, 0/4 | 0.9%, 5/10 | 15.8%, 7/8 |

## Why PP-OCRv5 mobile

- **The same everywhere.** It runs on onnxruntime-node, which the app already ships for its embedding and reranking models, with the same results on macOS, Windows and Linux. The system engines can't be that: Linux has none; Vision is macOS only; Windows.Media.Ocr reads one language at a time and only with that language's OCR pack, which a Windows set up in English doesn't have for Chinese (it read none of the Chinese pages), so the same scan would read differently on each system.
- **The most accurate here,** with the most quotes found, in English and Chinese, on clean and noisy pages and on tables, and on the two pages drawn after it was tuned. Vision is about as good and a little faster, on macOS only. Tesseract reads English as well, but Chinese poorly (14–21% of characters wrong; whole lines of Chinese read as Latin letters), and the app would ship it as tesseract.js, slower than the command measured here.
- **Small and fast enough:** two pinned files, 21.5 MB together (PP-OCRv5 server is 173 MB and 15 times slower), downloaded the first time a scanned page needs reading; about a second a page on a busy laptop, in a utility process that uses at most half the cores.
- **One model for Simplified and Traditional Chinese, English and Japanese,** at no extra cost (PP-OCRv4's model has no Traditional characters).
- **Licence and provenance:** the models are Apache-2.0 (PaddleOCR). The files are RapidOCR's ONNX conversions (Apache-2.0), downloaded from copies on Hugging Face whose bytes match RapidOCR's own release on ModelScope (`onnx/PP-OCRv5`); their SHA-256 hashes are pinned in `src/core/documents/scan/model.ts`. The recognition model carries its character list inside it.

### Reading lines at 15 pixels

PP-OCRv5's models read sharp, large lines worse than small, soft ones: from a 200 dpi page, read as PaddleOCR reads them, the mobile model dropped simple, common Chinese characters (一, 二, 人, 日, 月, 方, 支): 19% of `zh-clean`'s characters, 12% of `zh-hant`'s, 5.6% of the six first pages' over all, and 50 of 73 quotes. The native PaddleOCR 3.0 pipeline with the official PaddlePaddle models, and three other ONNX conversions, drop the same characters, so it is the model, not this code. Each line is now read first at 15 pixels high, each pixel the mean of the picture's under it, then scaled to the 48 pixels the model reads (`READ_HEIGHT` in `src/core/documents/scan/recognize.ts`). Boxes read 12 to 16 pixels high all did well; 15 had the fewest errors on the six pages; 18 and more brought the dropped characters back. The two held-out pages, drawn afterwards, read as well (0.8%).

## Running it

```sh
npm run eval:ocr
INCARNAMIND_OCR_ENGINES=ppocr-v5-mobile,vision npm run eval:ocr   # some of them
INCARNAMIND_OCR_ENGINES=pipeline npm run eval:ocr                 # only the app's own path
INCARNAMIND_OCR_RECORD=1 npm run eval:ocr                         # write read.json again
```

| Id | Engine | Needs |
|---|---|---|
| `ppocr-v5-mobile` | PP-OCRv5 mobile, the app's | nothing: downloaded into the cache |
| `ppocr-v5-server` | PP-OCRv5 server | nothing: downloaded into the cache |
| `ppocr-v4-mobile` | PP-OCRv4 mobile | nothing: downloaded into the cache |
| `vision` | macOS Vision, through `vision.swift` | macOS, and `swiftc` (Xcode's command-line tools) |
| `windows` | Windows.Media.Ocr, through `windows-ocr.ps1` | Windows, with the OCR packs of the pages' languages |
| `tesseract-fast`, `tesseract-best` | Tesseract | the `tesseract` command; its models are downloaded into the cache |
| `pipeline` | the app's own path, on PP-OCRv5 mobile | as `ppocr-v5-mobile` |

Models are downloaded once, checked against their pinned SHA-256 hashes, into `~/.cache/incarnamind-eval/` (`INCARNAMIND_EVAL_CACHE` to put them elsewhere), and later runs are offline. `INCARNAMIND_OCR_RUNS` sets how many times each page is read (2). A run writes `eval/results/ocr/<start time>/report.md` (ignored), with what each engine read on each page.
