# Files for a person to save

Issue #84 needs files saved by real word processors, because only those show how an application rewrites a file it didn't make, which is what the export (#105) has to survive. A program can write a .docx, but it can't be those applications.

## Done

- **Word for Mac:** `saved/word-mac/en-contract-services.docx` and `zh-contract-service.docx`, saved by the owner on 2026-10-12.
- **Google Docs (web):** `saved/google-docs/en-memo-policy.docx` and `en-report-quarterly.docx`, each uploaded to Drive, opened in Google Docs, saved as a Google Doc and exported as .docx, 2026-10-12.

## Still to do: WPS

WPS Office CN 12.1.29181 is installed. Two files from `eval/word/documents/synthetic/`, saved **as .docx under the same name** into `eval/word/saved/wps/`:

| File | Why this file |
|---|---|
| `en-contract-nda.docx` | Comments, a cross-reference field with a bookmark |
| `zh-contract-redline.docx` | Chinese tracked changes by two authors, a comment |

For each:

1. Open the file in WPS. Don't edit anything. If WPS asks to convert it or to enable editing, accept.
2. File > Save As, type "Microsoft Word Files (*.docx)", the original name. Don't pick "WPS Writer Files (*.wps)".
3. Put the file in `eval/word/saved/wps/`, or hand it back in `~/Downloads/IncarnaMind-Word-save/saved/wps/`.

Then:

```sh
INCARNAMIND_WORD_UPDATE=1 npm run eval:word   # adds them to manifest.json and the README tables
npm run eval:word                              # every file should still pass
```

and commit `eval/word` with a signed-off commit. Say in the hand-back note anything the app did that surprised you (a lost comment, a converted field). That is the point of the exercise. Then #84's last box can be ticked.
