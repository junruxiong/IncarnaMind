# IncarnaMind

**Write and ask your Documents in the same place.** Answers that draw on your Documents cite the page, and you can check that the quote is there.

IncarnaMind is a local-first desktop app for macOS, Windows and Linux. You write in a Mind and ask Questions where you write; each Answer lands in your draft with Citations to the pages it drew on. There's no account and no server: your Minds and the search index stay in one data folder on your computer, and your Documents stay where you keep them. Answers come from the model you choose, including local models through Ollama.

> v1 is in development; there's no release yet. Looking for the 2023 Python command-line tool? See [The old command-line tool](#the-old-command-line-tool).

## Features

- **Minds.** A Mind is a draft made of Notes you write, with headings, lists, code and LaTeX math, and of Questions and their Answers. Type `/` to insert a Question anywhere.
- **Questions and Answers.** Ask a Question anywhere in a Mind and its Answer streams in below it. The Answer sees the Blocks above the Question, except Notes you switch off, and searches your Documents. Answers are editable text: stop or regenerate them, and pick the model for each Question.
- **Checkable Citations.** Each Citation names the page it cites and carries a short quote. When the Answer finishes, IncarnaMind checks the quote against the text of that page and shows **Quote found on p. 12**, **Quote not found on p. 12** or **Can't check** (the page has no text, or the Document was deleted). "Found" means the quote is on the page, not that it supports the sentence. Clicking a Citation opens the Document at the cited page, with the quote highlighted when it was found.
- **Documents.** Link the folders where you keep your files, such as a papers folder or Zotero's storage folder, or add single files. IncarnaMind indexes PDF, TXT and Markdown files where they are, never moves, renames or changes them, and keeps up as you add, edit, move or delete files. Search matches their words and reranks the best matches with a small built-in model, so a Document can be searched, offline, as soon as it is read. You can turn on embeddings, which also match meaning, at the cost of slower indexing: a built-in model that runs on your CPU (a one-time 135 MB download), OpenAI, Google, an OpenAI-compatible server or Ollama. Reranking can use Cohere or Voyage AI instead. The viewer shows a PDF's outline, and you can open any Document in its default app or show it in its folder. When a file changes after you cited it, the Citation still checks against the text it quoted.
- **Folders and Tags.** Each Linked folder shows its own subfolders, or a flat list for Zotero-style folders, and files added on their own sit under Other Documents. Tags are applied automatically by your chat model, or by TypeSafe Jev if you add a Jev key. You can add or remove any Tag, and Tags you set yourself are never changed automatically.
- **Search scope.** Type `@` in a Question to limit its search to Folders (with their sub-Folders), Tags or single Documents.
- **Connectors (MCP).** Connect MCP servers, either a command run on your computer or a remote server by URL with OAuth sign-in, or import them from Claude Desktop or Cursor. Answers can call their Tools. A Tool that may change something asks you first, unless you choose **Always allow**.
- **Skills.** Import Skills, standard `SKILL.md` folders or zips, for Answers to follow. A Skill's scripts (Python, JavaScript or shell) run on your computer only after you approve each run, and one switch turns them all off. Three Built-in Skills ship with the app: summarise a Document, write a literature review across Documents, and turn a Mind into a report.
- **Export.** Export a Mind to Word (.docx), with Citations as footnotes, or to Markdown. Citations whose quote wasn't found or can't be checked are marked "[unverified]". To back up your Minds, the index and settings, copy the data folder. Your Documents are your own files, so back them up the way you already do.
- **Models.** OpenAI, Anthropic, Google, any OpenAI-compatible server, or local models through Ollama: when Ollama is running, one click picks a local model and downloads it.
- **Privacy.** IncarnaMind collects no usage data. Before it first sends anything to an outside service, such as your chat provider, a cloud embedding model or a Connector, a dialog lists what is sent and to whom, and nothing is sent until you allow it. Where a provider offers the choice, IncarnaMind asks it not to store your requests (for OpenAI, every request carries `store: false`). **Settings → Privacy** lists every flow and lets you revoke it. Crash reports are off unless you opt in, and they're scrubbed of file paths, Document text and Mind content. A log in the data folder (**Settings → Open logs folder**) records what IncarnaMind did and what went wrong, never your content or keys, and stays on your computer unless you send it. API keys are encrypted with your system's keychain.
- **Languages.** The interface is in English and Simplified Chinese.

## Install

Until the first release, run the app from source (see [Development](#development)).

Download the installer for your system from [GitHub Releases](https://github.com/junruxiong/IncarnaMind/releases):

- **macOS:** `IncarnaMind-<version>-mac-arm64.dmg` for Apple silicon, or `-mac-x64.dmg` for Intel. Open it and drag IncarnaMind into Applications.
- **Windows:** `IncarnaMind-<version>-win-x64.exe`.
- **Linux:** `IncarnaMind-<version>-linux-x86_64.AppImage`. Make it executable (`chmod +x IncarnaMind-*.AppImage`) and run it. If it doesn't start on Ubuntu 22.04 or later, install FUSE 2: `sudo apt install libfuse2` (`libfuse2t64` on 24.04 and later).

IncarnaMind checks for a new version each time it starts. On Windows, in the Linux AppImage and in a signed macOS build, it downloads the update in the background and installs it when you quit, or straight away if you choose **Restart now**.

**Windows: "Windows protected your PC".** The installer isn't code-signed yet, so Microsoft Defender SmartScreen stops it the first time. Click **More info**, then **Run anyway**. If your browser flags the download, choose **Keep**.

**macOS: opening an unsigned build.** Until the macOS build is signed and notarized, macOS refuses to open it with "Apple could not verify 'IncarnaMind' is free of malware". To open it anyway:

1. Open IncarnaMind once, and click **Done** in the warning.
2. Open **System Settings → Privacy & Security**, scroll down to **Security**, and click **Open Anyway** next to the message about IncarnaMind.
3. Confirm with your password or Touch ID, then click **Open Anyway** again.

macOS remembers your choice. Or, in Terminal: `xattr -dr com.apple.quarantine /Applications/IncarnaMind.app`.

An unsigned macOS build can't update itself. When a new version is out, IncarnaMind tells you and offers its download page; install it the same way, replacing the old app.

## Development

You need Node.js 24 or newer (`.nvmrc`). After cloning:

```shell
npm install
npm run dev          # start the app in development
npm run typecheck    # TypeScript, strict
npm run lint         # Biome (npm run format fixes what it can)
npm test             # Vitest: drives the core's public interface; no keys, no network
npm run test:smoke   # Playwright: builds the app and drives it in Electron
npm run eval         # the retrieval evaluation (see below)
```

Keep development data apart from your real data folder by pointing the app at another one: `INCARNAMIND_DATA_DIR=/tmp/incarnamind-dev npm run dev`.

Embeddings are off by default: document search is keyword search, reranked. Turned on in Settings, it runs a built-in embedding model, multilingual-e5-small (int8 ONNX, 135 MB), on your CPU in an Electron utility process. The app downloads it from Hugging Face into the data folder (`models/`) the first time a Document needs it, and checks each file's SHA-256; after that, indexing works offline. The tests use a deterministic fake instead. `INCARNAMIND_REAL_MODEL=1 npm test` also runs one test with the real model, downloading it unless `INCARNAMIND_MODEL_DIR` points at a folder holding its files. On Linux x64, set `ONNXRUNTIME_NODE_INSTALL=skip` when you run `npm install`, or onnxruntime-node also downloads its CUDA libraries, which the app doesn't use.

**Evaluation.** `npm run eval` measures retrieval, and optionally Citation quality, against the v1 targets. It adds the sample PDFs in `data/` and the Chinese articles in `eval/retrieval/fixtures/` to a temporary data folder and searches for each Question in `eval/retrieval/questions.json`. Retrieval needs no keys; the first run downloads the embedding and reranking models into `~/.cache/incarnamind-eval/`. It isn't part of `npm test` or CI's default run. See [`eval/README.md`](eval/README.md) for the Citation part, the variables and the latest results.

### Packaging and releases

[electron-builder](https://www.electron.build) packages the app (config: `electron-builder.yml`), and electron-updater updates installed apps from GitHub Releases (`src/main/updater.ts`). To package locally, into `dist/`:

```shell
npm run dist:dir     # the unpacked app, for a quick check (e.g. dist/mac-arm64/IncarnaMind.app)
npm run dist         # the installers for the current system; never published
```

**Cutting a release:**

1. Set the version, commit it and tag it: `npm version 1.2.3`. This updates `package.json` and `package-lock.json`, commits, and creates the tag `v1.2.3`.
2. Push the commit and the tag: `git push && git push origin v1.2.3`.
3. The tag starts the **Release** workflow (`.github/workflows/release.yml`). It checks that the tag matches `package.json`, creates a draft GitHub Release, and builds and uploads:
   - for macOS, a dmg and a zip, each for Apple silicon and Intel;
   - for Windows, an NSIS installer;
   - for Linux, an AppImage;
   - the `latest*.yml` files that auto-update reads.
4. Review the draft, edit its notes, try the installers, and click **Publish release**. Installed apps only see an update once its release is published.

**macOS signing and notarization.** The workflow signs and notarizes the macOS app when these repository secrets exist. Without `CSC_LINK`, it builds an unsigned (ad-hoc signed) app and still succeeds.

| Secret | Value |
| --- | --- |
| `CSC_LINK` | The "Developer ID Application" certificate and its private key, exported from Keychain Access as a `.p12` file and base64-encoded (`base64 -i certificate.p12 \| pbcopy`) |
| `CSC_KEY_PASSWORD` | The password of that `.p12` file |
| `APPLE_ID`, `APPLE_APP_SPECIFIC_PASSWORD`, `APPLE_TEAM_ID` | To notarize with an Apple ID: the Apple ID, an [app-specific password](https://support.apple.com/102654) and the team ID |
| `APPLE_API_KEY`, `APPLE_API_KEY_ID`, `APPLE_API_ISSUER` | Or, to notarize with an App Store Connect API key: the contents of the `.p8` file, its key ID and the issuer ID |

The Windows installer stays unsigned in v1; signing through the SignPath Foundation is a follow-up.

**Crash reports.** Users can opt in to crash reports in **Settings → Privacy**; they go to Sentry, scrubbed of file paths, Document text, Mind content, Questions and Answers (`src/main/crashScrubber.ts`). Only a build made with a Sentry DSN offers them: set `MAIN_VITE_SENTRY_DSN` when building (electron-vite reads it from the environment or a `.env.local` file). The release workflow passes the `SENTRY_DSN` repository secret; without it, releases don't offer crash reports. Never commit a DSN.

## Contributing

Bugs, ideas and the specs for new work are [GitHub issues](https://github.com/junruxiong/IncarnaMind/issues). New issues are triaged with five labels:

| Label | Meaning |
| --- | --- |
| `needs-triage` | The maintainer needs to evaluate it |
| `needs-info` | Waiting on the reporter for more information |
| `ready-for-agent` | Fully specified, ready for an AFK agent |
| `ready-for-human` | Needs a person to implement it |
| `wontfix` | Won't be actioned |

`docs/agents/` holds the conventions for working on issues, for people and coding agents alike: the issue tracker and its `gh` commands ([`issue-tracker.md`](docs/agents/issue-tracker.md)), the labels ([`triage-labels.md`](docs/agents/triage-labels.md)) and how to use the domain docs ([`domain.md`](docs/agents/domain.md)).

When you change code:

- Use the words defined in [`CONTEXT.md`](CONTEXT.md) (a Passage, not a chunk; a Connector, not a plugin), and say so when a change goes against an ADR.
- Test through the core's public interface, as the UI uses it, and assert what a User would see. The core tests need no API keys and no network.
- Keep `src/core` free of Electron imports; Biome enforces it.
- Run `npm run typecheck`, `npm run lint` and `npm test` before opening a pull request. CI runs them, and the smoke tests, on every push.

## Architecture

[`CONTEXT.md`](CONTEXT.md) is the glossary, and [`docs/adr/`](docs/adr/) records the decisions behind the design, from the local-first Electron app (ADR-0004, ADR-0006) to SQLite storage (ADR-0008) and retrieval (ADR-0009).

In short: the React UI in the renderer talks over a typed bridge to the core, a TypeScript module in the Electron main process with no Electron imports. The core keeps each Mind as a Yjs document and everything else in SQLite, in the data folder. Documents are indexed in place from Linked folders, never copied (ADR-0010). Answers come from a tool-calling loop (ADR-0007), through the AI SDK.

Code lives in `src/`: `core` (app logic), `main` (Electron main process), `preload` (the typed bridge to the UI), `renderer` (React UI) and `shared` (i18n dictionaries, bridge names and the text normaliser).

## The old command-line tool

The original Python command-line tool lives at the `v1-python-cli` tag.

The Django backend and Next.js frontend of the old hosted web app, which this app replaces, are in the Git history up to commit `d9f7fab`.

## Citing IncarnaMind

If you want to cite IncarnaMind, please use this BibTeX entry:

```bibtex
@misc{IncarnaMind2023,
  author = {Junru Xiong},
  title = {IncarnaMind},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub Repository},
  howpublished = {\url{https://github.com/junruxiong/IncarnaMind}}
}
```

## Licence

IncarnaMind is licensed under the [Apache License 2.0](LICENSE).

The app bundles three fonts, all under the [SIL Open Font License 1.1](https://openfontlicense.org): Source Serif 4 (from `@fontsource-variable/source-serif-4`), Source Sans 3 (from `@fontsource-variable/source-sans-3`) and JetBrains Mono (from `@fontsource/jetbrains-mono`). Their licences ship with the app, in the renderer's `licenses/` folder.
