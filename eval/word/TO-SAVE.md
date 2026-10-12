# Files for a person to save

Issue #84 needs files saved by real word processors, because only those show how an application rewrites a file it didn't make, which is what the export (#105) has to survive. A program can write a .docx, but it can't be those applications.

## Done

- **Word for Mac:** `saved/word-mac/en-contract-services.docx` and `zh-contract-service.docx`, saved by the owner on 2026-10-12.

## Not possible

- **WPS:** not installed on the owner's computer, so the two WPS files are dropped unless someone with WPS saves them (steps below, same as for Google Docs but with File > Save As > "Microsoft Word Files (*.docx)"; folder `saved/wps/`; files `en-contract-nda.docx` and `zh-contract-redline.docx`).

## Still to do: Google Docs

Two files, each a synthetic one from `eval/word/documents/synthetic/` (invented text, no personal data). Save each **as .docx under the same name** into `eval/word/saved/google-docs/`.

| File | Why this file |
|---|---|
| `en-memo-policy.docx` | Tracked changes (Google shows them as suggestions), a comment, a footnote |
| `en-report-quarterly.docx` | A table, fields, footnote, endnote, a content control |

For each:

1. Upload the file to Google Drive and open it with Google Docs. Don't edit anything. If you are asked whether to turn tracked changes into suggestions, accept.
2. File > Download > Microsoft Word (.docx).
3. Move the download into `eval/word/saved/google-docs/` and give it the original name (Google adds the title). Create the folder if it isn't there.

When both are saved:

```sh
INCARNAMIND_WORD_UPDATE=1 npm run eval:word   # adds them to manifest.json and the README tables
npm run eval:word                              # every file should still pass
git add eval/word && git commit -s -m "Add the files saved in Google Docs (#84)"
```

Or hand the two files back and someone else runs those lines. Say in the hand-back note anything the app did that surprised you (a lost comment, a converted field). That is the point of the exercise.

Then #84's last box (a person saves files in Word for Mac and Google Docs; WPS is not available) can be ticked.
