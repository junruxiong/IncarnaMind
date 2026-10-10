<!-- README.md and README.zh-CN.md match section for section. Change them together. -->

<p align="center"><b>English</b> · <a href="README.zh-CN.md">简体中文</a></p>

<h1 align="center">IncarnaMind</h1>

<p align="center">
  <b>The AI notebook that checks its own citations.</b><br>
  Write and ask in one place, with answers from your own files and the AI model you choose. Free and open source, on your computer.
</p>

<p align="center">
  <a href="#download"><b>Download</b></a> ·
  <a href="#what-you-can-do">What you can do</a> ·
  <a href="#your-files-stay-yours">Privacy</a> ·
  <a href="https://github.com/junruxiong/IncarnaMind/discussions">Discussions</a>
  <br><br>
  <img alt="Desktop app for macOS, Windows and Linux" src="https://img.shields.io/badge/macOS%20%C2%B7%20Windows%20%C2%B7%20Linux-desktop%20app-17191C">
  <a href="LICENSE"><img alt="License: Apache-2.0" src="https://img.shields.io/github/license/junruxiong/IncarnaMind?color=17191C"></a>
  <!-- After the first release, add: <a href="https://github.com/junruxiong/IncarnaMind/releases/latest"><img alt="Latest release" src="https://img.shields.io/github/v/release/junruxiong/IncarnaMind?color=17191C"></a> -->
</p>

<p align="center"><img src="docs/images/demo.gif" width="880" alt="Nine screens in turn: pointing at a Citation shows its quote; opening it shows the PDF page with the quote highlighted; all your Documents in every format, with Folders and Tags; asking about selected cells in an Excel sheet; a comparison table across ten contracts; a Word contract with tracked changes and comments; notes beside a live meeting transcript; background Tasks, running and scheduled; and Settings with Claude, GPT, DeepSeek and a local model through Ollama."></p>

<p align="center"><sub>Asking about a selection, comparison tables, Word tracked changes and comments, meetings and background Tasks are <a href="#coming-next">coming next</a>; the rest is in the app today.</sub></p>

## Download

**IncarnaMind isn't released yet.** Once it is, the [Releases page](https://github.com/junruxiong/IncarnaMind/releases) will have installers for macOS (Apple silicon and Intel), Windows and Linux. To hear when, click **Watch → Custom → Releases** at the top of this page. To try it now, run it from source with Node.js 24 or newer:

```shell
git clone https://github.com/junruxiong/IncarnaMind.git && cd IncarnaMind
npm install && npm run dev
```

<details>
<summary>If your system won't open it</summary>

- **Windows** says "Windows protected your PC": the installer isn't code-signed yet. Click **More info**, then **Run anyway**.
- **macOS** says it can't verify IncarnaMind: until the app is notarized, open it once and click **Done**, then **System Settings → Privacy & Security → Open Anyway**. Or run `xattr -dr com.apple.quarantine /Applications/IncarnaMind.app`. An unsigned Mac app can't update itself: IncarnaMind tells you when a new version is out and offers its download page.
- **Linux**: run `chmod +x IncarnaMind-*.AppImage`. On Ubuntu 22.04 or later, also `sudo apt install libfuse2` (`libfuse2t64` on 24.04 and later).

</details>

## What you can do

**Write and ask in one place.** A Mind is a notebook for one topic: ask a Question from the box at the foot of the Mind, and the Answer lands in your draft, at your cursor, as text you can edit.

**Check every answer.** Answers cite where each claim came from and carry the quote, which is checked against that place: a green check if it's there, an amber mark if it isn't, and a click opens the page with the quote highlighted.

**Use the files you already have.** Link the folders you already use, even Zotero's storage folder: IncarnaMind reads the PDF, Word (comments included), PowerPoint, Excel, CSV, Markdown and plain-text files in them where they are, never moves or changes them, and keeps up as they change.

**Keep a big library tidy.** Tags are added for you, IncarnaMind can sort Documents into your Folders, and the choices you make yourself are kept.

**Choose your AI.** Use your own OpenAI, Anthropic or Google key, any OpenAI-compatible server, or a free local model through Ollama, so nothing leaves your computer.

**Hand it in.** Export a Mind to Word with its citations as footnotes, or to Markdown. A citation that didn't pass the check is marked "[unverified]".

**Connect apps and Skills that ask first.** Add Connectors (MCP servers), or bring over the ones you set up in Claude Desktop or Cursor, and Skills (standard `SKILL.md` folders), three of them built in; a tool that could change something, or a Skill's script, runs only after you say yes.

**English and Chinese.** The whole app is in both.

## Your files stay yours

There's no account and no IncarnaMind server. Your Minds and the search index stay in one data folder on your computer, and your files stay where you keep them, unchanged. A question and the passages found for it go only to the AI model you picked, or nowhere with a local model. API keys are encrypted in your system's secure storage. Usage data isn't collected. Crash reports are off unless you turn them on, and are stripped of your content.

- **You're asked before anything leaves.** The first time IncarnaMind would send anything to an outside service, it shows what would be sent and to whom, and waits for your OK. **Settings → Privacy** lists each one and lets you take it back.
- **Providers are asked not to keep it.** Where a provider offers the choice, IncarnaMind asks it not to store your requests; every request to OpenAI says so.
- **Nothing runs without you.** Before a Connector's tool that could change something, or a Skill's script, runs, IncarnaMind asks you: allow it once, always allow it, or deny it. **Settings → Tools** lists every rule you've set, and you can revoke any of them.

<details>
<summary><b>Questions</b></summary>

- **Is it free?** Yes: it's open source under the Apache 2.0 licence. With a cloud AI model, you pay its provider for what you use; a local model costs nothing.
- **Do I need an API key?** No. If [Ollama](https://ollama.com) is running, IncarnaMind can download a local model in one click. Or paste a key from OpenAI, Anthropic or Google, or connect any OpenAI-compatible server.
- **Does it work offline?** Searching your files does: it works on the words in them, and meaning-based search (embeddings) is optional and off by default. Answers need an AI model, so with a local model everything works offline.
- **Will it change my files?** No. It reads them where they are, and writes only to its own data folder and the files you export.
- **What does "Quote found" mean?** That the quoted words are in the place the citation names, not that they prove your sentence, which is why the page is one click away. "Can't check" means there's no text to compare against, as on a scanned page.
- **How do I back up?** Copy the data folder: it holds your Minds, the search index and your settings.

</details>

## Coming next

Planned, not available yet, with no dates (see the [roadmap](https://github.com/junruxiong/IncarnaMind/issues/74)): meetings with a live transcript that answers can cite; Word files opened and exported back with tracked changes; your drafts, and other people's documents, checked against your files; comparison tables across documents; AI edits you accept or reject; and research Tasks that run in the background.

## Contributing

Code, testing with your own documents, translations and ideas are all welcome: start with [CONTRIBUTING.md](CONTRIBUTING.md), [open an issue](https://github.com/junruxiong/IncarnaMind/issues/new/choose) or ask in [Discussions](https://github.com/junruxiong/IncarnaMind/discussions).

## History

IncarnaMind began in 2023 as a Python command-line tool for chatting with your PDFs, starred by about 800 of you, and was rebuilt in 2026 as a desktop app around one idea: an answer from your files is only useful if you can check it. The old tool lives on at the [`v1-python-cli`](https://github.com/junruxiong/IncarnaMind/tree/v1-python-cli) tag. Thank you for sticking around, and if IncarnaMind is useful to you, a star helps other people find it.

<details>
<summary>Citing IncarnaMind</summary>

```bibtex
@misc{IncarnaMind2023, author = {Junru Xiong}, title = {IncarnaMind}, year = {2023},
  publisher = {GitHub}, journal = {GitHub Repository},
  howpublished = {\url{https://github.com/junruxiong/IncarnaMind}}}
```

</details>

## Licence

[Apache License 2.0](LICENSE). The fonts that ship with the app (Source Serif 4, Source Sans 3 and JetBrains Mono) are under the [SIL Open Font License 1.1](https://openfontlicense.org).
