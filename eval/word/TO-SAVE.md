# Files for a person to save

Issue #84 needs six files saved by real word processors: two in Word for Mac, two in WPS and two in Google Docs. A program can write a .docx, but only these applications show how they rewrite a file they didn't make, which is what the export (#105) has to survive. Nobody but the owner opens these files in their apps, so this is a list of steps.

Each file is a synthetic one from `eval/word/documents/synthetic/` (invented text, no personal data). Save each one **as .docx under the same name** into the folder shown.

| App | Folder | File | Why this file |
|---|---|---|---|
| Word for Mac | `eval/word/saved/word-mac/` | `en-contract-services.docx` | Numbered clauses, a table of contents field, page fields, footnote, endnote, content controls |
| Word for Mac | `eval/word/saved/word-mac/` | `zh-contract-service.docx` | Chinese clause numbers (第一条), footnote, endnote, East Asian fonts |
| WPS | `eval/word/saved/wps/` | `en-contract-nda.docx` | Comments, cross-reference field with a bookmark |
| WPS | `eval/word/saved/wps/` | `zh-contract-redline.docx` | Chinese tracked changes by two authors, a comment |
| Google Docs | `eval/word/saved/google-docs/` | `en-memo-policy.docx` | Tracked changes (Google shows them as suggestions), a comment, a footnote |
| Google Docs | `eval/word/saved/google-docs/` | `en-report-quarterly.docx` | A table, fields, footnote, endnote, a content control |

## Steps

For each row:

1. Open the file from `eval/word/documents/synthetic/` in the app. Don't edit anything. If the app asks to enable editing, or to convert the file, accept.
2. Save a copy as .docx:
   - **Word for Mac:** File > Save As, format "Word Document (.docx)", name unchanged, folder as in the table. If Word offers to update fields or repair, say so in your hand-back note.
   - **WPS:** File > Save As, type "Microsoft Word Files (*.docx)", name unchanged. Don't pick "WPS Writer Files (*.wps)".
   - **Google Docs:** upload the file to Drive and open it with Google Docs, then File > Download > Microsoft Word (.docx). Move the download into the folder and give it the original name (Google adds the title). If you are asked whether to turn tracked changes into suggestions, accept.
3. Create the folder if it isn't there. Keep the original name; one file per row.

When all six are saved:

```sh
INCARNAMIND_WORD_UPDATE=1 npm run eval:word   # adds them to manifest.json and the README table
npm run eval:word                              # every file should still pass
git add eval/word && git commit -s -m "Add the files saved in Word for Mac, WPS and Google Docs (#84)"
```

Or hand the six files back and someone else runs those lines. Say in the hand-back note anything the app did that surprised you (a repair prompt, a lost comment, a converted field). That is the point of the exercise.

Then #84's last box (a person saves the WPS, Google Docs and Word for Mac files) can be ticked.
