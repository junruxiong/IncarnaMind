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
- **Word previews** are page-like, with docx-preview (Apache-2.0).
- **PowerPoint previews** are a slide-by-slide outline: title, text, images, then speaker notes. A true slide renderer comes later, once one shows images and tables reliably under the app's Content-Security-Policy.
- **Excel and CSV previews** are a grid with sheet tabs, highlighting the cited rows.
- **Second wave:** we read text with our own code where the format is simple (HTML, EPUB, RTF, OpenDocument), and show text-only previews. The legacy binary formats are decided when we reach them. SheetJS, now published only from its own site, is the likely route for `.xls`.
- **LibreOffice:** not used in v1. It is a separate 285 MB+ install, with frequent security advisories in its importers. Later, at most, it could give an optional exact-layout preview when it is already installed, but never the Locations Citations point at.

## Considered options

- **Every format now**, including legacy and OpenDocument: more coverage, but those likely need LibreOffice, which the research is weighing.
- **LibreOffice for every Office format**: exact layouts and real Word pages, but a large separate install with a poor security record for opening untrusted files.
- **Pages for everything**: one unit everywhere, but slides, sheets and Markdown have no pages their readers would recognise.
- **Spreadsheet Citations marked "can't check"**: simpler, but it leaves unchecked the claims most worth checking.

## Consequences

- **Units.** Each row of `document_pages` is now a Unit (migration 22): `page` is its number, and `kind`, `label` (JSON) and `anchors` (JSON) say what it is. PDFs keep their pages; a TXT or Markdown file stored whole becomes Unit 1 of kind `text`, so the old versions Citations quote stay checkable.
- **Processing.** `PROCESSING_VERSION` is 5, but only kinds whose Units changed are processed again at startup (`CURRENT_SINCE`): Markdown and TXT, not PDFs, whose Passages and vectors stay.
- **Block sizes.** A spreadsheet or CSV Unit holds up to 100 rows and about 1,600 characters (about 400 tokens), and every block after a sheet's first repeats the header row, so a Passage of about 500 tokens takes in a block or two and knows its columns. Passages never run from one sheet into the next. A plain-text Unit holds up to 50 lines and about 4,000 characters; a Word or Markdown section is split into parts past 6,000 characters. At most 20,000 rows (500,000 cells) of a workbook or CSV are read; the grid says so when it stops.
- **Spreadsheet values** are stored as Excel shows them ("£350,200", "4.2%", ISO dates), and quotes are matched with number formatting normalised, so a raw "350200" matches too.
