# Document formats, and where a Citation points

IncarnaMind reads more than PDF, Markdown and plain text, and a Citation's Location is the unit each format's readers use. Settled with the User on 2026-10-07, after the research in `docs/research/document-formats.md`.

**Formats, in waves:**
- **Now:** Word (`.docx`), PowerPoint (`.pptx`), Excel (`.xlsx`) and `.csv`, alongside PDF, Markdown and TXT.
- **Second wave, after the research:** `.html`, `.epub`, `.rtf`, OpenDocument (`.odt`, `.odp`, `.ods`) and the legacy binary Office formats (`.doc`, `.ppt`, `.xls`).
- **Out:** images and scanned pages that need text recognition.

**Locations:**
- PDF: one or two pages ("p. 4"), as before.
- PowerPoint: one or two slides ("slide 4"). Speaker notes count as part of their slide.
- Excel and CSV: a sheet and a range of rows ("Revenue, rows 12–14"). A CSV has a single sheet.
- Markdown: the section the quote sits under ("§ Methods"). Plain text: a range of lines ("lines 120–134"). This replaces citing the whole Document.
- Word: the section the quote sits under ("§ 2.1 Sensitivity"), as for Markdown. A preview's page breaks follow only explicit breaks, so its page numbers wouldn't match Word's, and a wrong page undermines a checkable Citation.

**Checking quotes from spreadsheets.** The cited rows' cells, joined in reading order, must contain the quote's words and numbers, the same way text quotes are checked. Number formatting is normalised first, so "4,812" matches "4812". Spreadsheet Citations are checked, not marked "can't check", because numbers are what analysts cite most and get wrong most.

**Reading and previews.**
- **Reading:** our own small zip-and-XML reader extracts text from `.docx`, `.pptx`, `.xlsx` and `.csv`, with no dependencies.
- **Word previews** are pages as the file sets them, with docx-preview (Apache-2.0): page size and margins, headers and footers, footnotes and endnotes, list numbering and bullets, fonts, colours, tables and images. Comments are notes in a margin beside the pages, as Word's markup area shows them, their text highlighted. A document that names no font is in Times New Roman, as in Word; Office's own fonts (Calibri, Cambria, Aptos), where they aren't installed, fall back to the app's bundled Source Sans 3 and Source Serif 4, never to the browser's Times.
- **PowerPoint previews** are a slide-by-slide outline: title, text, images, then speaker notes. A true slide renderer comes later, once one shows images and tables reliably under the app's Content-Security-Policy.
- **Excel previews** are the workbook as Excel shows it, read by our own reader in the same pass as its text: fonts, fills, borders, alignment and wrapping, rich text, merged cells, column widths and row heights, hidden rows and columns, frozen panes, gridlines on or off, notes, and values in their own number formats (dates, times, percentages, currency, accounting, scientific, fractions, colours and conditions). Pictures are shown; a chart is a labelled placeholder unless the file keeps a picture of it. Sheet tabs are at the bottom, and the cited cells are washed. Only the rows in view are drawn. What is indexed and quoted is unchanged (ISO dates, the plain number for formats it doesn't know): the look is for the eye only.
- **CSV previews** are a plain grid, its columns fitted to their values.
- **Markdown previews** are formatted, as GitHub renders them, by our own reader, so every rendered character maps back to the file and the quote is washed where it is: headings, lists (nested, numbered, tasks), tables, quotes, code highlighted by its language, links, and pictures beside the file. The main process serves those only from the file's own folder or below it, and only images; pictures from the web aren't loaded and show a placeholder. "Source" shows the text as written. **Plain text** stays plain, in the Mind's serif, or monospaced when it reads as code.
- **Second wave:** we read text with our own code where the format is simple (HTML, EPUB, RTF, OpenDocument), and show text-only previews. The legacy binary formats are decided when we reach them. SheetJS, now published only from its own site, is the likely route for `.xls`.
- **LibreOffice:** not used in v1. It is a separate 285 MB+ install, with frequent security advisories in its importers. Later, at most, it could give an optional exact-layout preview when it is already installed, but never the Locations Citations point at.

## Considered options

- **Every format now**, including legacy and OpenDocument: more coverage, but those likely need LibreOffice, which the research is weighing.
- **LibreOffice for every Office format**: exact layouts and real Word pages, but a large separate install with a poor security record for opening untrusted files.
- **Pages for everything**: one unit everywhere, but slides, sheets and Markdown have no pages their readers would recognise.
- **Spreadsheet Citations marked "can't check"**: simpler, but it leaves unchecked the claims most worth checking.
- **A library to read workbooks' styles** (2026-10-09, after `docs/research/vscode-previews.md`): `hucre` (MIT, no dependencies, about 43 KB gzipped) drew the fixture workbook about as our reader does, but not frozen panes. Our reader also covers hidden rows and columns, row heights, rich text, pictures, notes and Excel's number formats. It reads the look in the same streaming pass as the text, under the same row, cell and zip-bomb limits, and adds nothing to the app. `numfmt` (MIT) stays the fallback if our number formats fall short. `@silurus/ooxml` draws on a canvas and needs `wasm-unsafe-eval`.
- **A Markdown library** (markdown-it and the like): fuller CommonMark, but its HTML has no map back to the file below the block, so a quote couldn't be washed where it is.

## Consequences

- **Units.** Each row of `document_pages` is now a Unit (migration 22): `page` is its number, and `kind`, `label` (JSON) and `anchors` (JSON) say what it is. PDFs keep their pages; a TXT or Markdown file stored whole becomes Unit 1 of kind `text`, so the old versions Citations quote stay checkable.
- **Processing.** `PROCESSING_VERSION` is 5, but only kinds whose Units changed are processed again at startup (`CURRENT_SINCE`): Markdown and TXT, not PDFs, whose Passages and vectors stay.
- **Block sizes.** A spreadsheet or CSV Unit holds up to 100 rows and about 1,600 characters (about 400 tokens), and every block after a sheet's first repeats the header row, so a Passage of about 500 tokens takes in a block or two and knows its columns. Passages never run from one sheet into the next. A plain-text Unit holds up to 50 lines and about 4,000 characters; a Word or Markdown section is split into parts past 6,000 characters. At most 20,000 rows (500,000 cells) of a workbook or CSV are read; the grid says so when it stops.
- **Spreadsheet values** are stored as Excel shows them ("£350,200", "4.2%", ISO dates), and quotes are matched with number formatting normalised, so a raw "350200" matches too.
- **A workbook's look** is read only for the preview (`readWorkbook`'s `layout`); processing doesn't read it, so no Document is processed again for it.
- **A Markdown file's pictures** come from the Document protocol under the Document's own URL (`documentImageUrl`), checked by the core (`openDocumentImage`): the path written in the file, resolved with links followed, inside the file's folder, an image's extension, at most 20 MB. The viewer turns them into `data:` URLs, so the Content-Security-Policy is unchanged.
