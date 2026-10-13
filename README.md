<!-- README.md and README.zh-CN.md match section for section. Change them together. -->

<p align="center"><b>English</b> · <a href="README.zh-CN.md">简体中文</a></p>

<h1 align="center">IncarnaMind</h1>

<p align="center">
  <b>The AI notebook that checks its own citations.</b><br>
  Write and ask in one place, with answers from your own files and the AI model you choose. Free and open source, on your computer.
</p>

<p align="center"><a href="https://tally.so/r/rjN2jL?ref=readme"><img src="docs/images/notify-en.svg" width="278" height="40" alt="Get notified when the alpha opens"></a></p>

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

<p align="center"><img src="docs/images/demo.gif" width="880" alt="Eight screens in turn: a Mind with the assistant beside it, its Answer's citations checked; the Answer's steps unfolded, showing what it searched and the sources it used; a Citation opened beside the note with the quote highlighted on the PDF page; the assistant's change waiting in a Mind to accept or reject; the Library with every Document, Folders and Tags; three groups of tabs side by side; a meeting written in a Mind while the assistant answers from the live transcript; and the app in Chinese."></p>

<p align="center"><sub>The assistant changing a Mind for you to accept or reject, and answers from a live meeting transcript are <a href="#coming-next">coming next</a>; the rest is in the app today.</sub></p>

**IncarnaMind** is a free, open-source desktop app for macOS, Windows and Linux where you ask questions about your own documents (PDF, Word, PowerPoint, Excel and more) and write with AI. Every answer cites the exact page, and IncarnaMind checks that each quoted passage is really there. It works with your ChatGPT plan, your own API key or a local model through Ollama.

## Download

**IncarnaMind isn't released yet.** Once it is, the [Releases page](https://github.com/junruxiong/IncarnaMind/releases) will have installers for macOS (Apple silicon and Intel), Windows and Linux. To hear when, [leave your e-mail](https://tally.so/r/rjN2jL?ref=readme) or click **Watch → Custom → Releases** at the top of this page. To try it now, run it from source with Node.js 24 or newer:

```shell
git clone https://github.com/junruxiong/IncarnaMind.git && cd IncarnaMind
npm install && npm run dev
```

Windows releases are signed through the SignPath Foundation's free signing for open-source projects: see the [Code signing policy](docs/code-signing-policy.md).

<details>
<summary>If your system won't open it</summary>

- **Windows** says "Windows protected your PC": the installer isn't code-signed yet. Click **More info**, then **Run anyway**.
- **macOS** says it can't verify IncarnaMind: until the app is notarized, open it once and click **Done**, then **System Settings → Privacy & Security → Open Anyway**. Or run `xattr -dr com.apple.quarantine /Applications/IncarnaMind.app`. An unsigned Mac app can't update itself: IncarnaMind tells you when a new version is out and offers its download page.
- **Linux**: run `chmod +x IncarnaMind-*.AppImage`. On Ubuntu 22.04 or later, also `sudo apt install libfuse2` (`libfuse2t64` on 24.04 and later).

</details>

## What you can do

**Write and ask in one place.** A Mind is a notebook for one topic. Beside it sits the assistant: ask it about the Mind you're in or your files, and the Answer comes back there, in a conversation, with its citations checked. Your draft stays as you wrote it until you pick **Add to note**, which puts the Answer at your cursor as text you can edit, citations and all, with a link back to the conversation it came from.

**Check every answer.** Answers cite where each claim came from and carry the quote, which is checked against that place: a green check if it's there, an amber mark if it isn't, and a click opens the page with the quote highlighted. Unfold an Answer's steps to see what it searched and which sources it used.

**Take notes in a meeting.** Start recording and a live transcript runs beside your Mind, transcribed on this computer. When the call ends, save it as a Document, and your notes can cite what was said.

**Use the files you already have.** Link the folders you already use, even Zotero's storage folder: IncarnaMind reads the PDF, Word (comments included), PowerPoint, Excel, CSV, Markdown and plain-text files in them where they are, never moves or changes them, and keeps up as they change.

**Find and add sources.** Search OpenAlex, arXiv and PubMed and add open-access papers as Documents, save a web page by its address so it can be cited and checked like a file, and import your references from Zotero or any CSL JSON, BibTeX or RIS file.

**Write with AI you can check.** Select text and choose Edit with AI: the changes come back for you to accept or reject. Turn on suggestions as you write, and when you pause, grey text offers how the paragraph could go on, only where your Documents support it and always with a citation; Tab accepts.

**Keep a big library tidy.** Tags are added for you, IncarnaMind suggests a Folder for each Document for you to accept, and the choices you make yourself are kept.

**Choose your AI.** Already pay for ChatGPT Plus or Pro? Continue with ChatGPT and ask on your plan, no key needed. Or use your own key from OpenAI, Anthropic, Google, DeepSeek, Qwen, SiliconFlow or OpenRouter, and more, such as Kimi, GLM, Mistral, xAI or any OpenAI-compatible server, or a free local model through Ollama, so nothing leaves your computer.

**Ask from the AI agent you already use** (experimental, off by default). Connect Claude, Codex, Cursor and others in **Settings → Agents**, on their own plan: they read only the folders you choose, every quote they get from IncarnaMind is checked, and what they read goes to their AI service under its terms. [How to connect each one](docs/agents.md).

**Hand it in.** Export a Mind to Word or Markdown, with its citations and bibliography in the style you pick: APA, MLA, Chicago, Harvard, IEEE, Vancouver, GB/T 7714 or IncarnaMind's numbered style. A citation that didn't pass the check is marked "[unverified]".

**Connect apps and Skills that ask first.** Add Connectors (MCP servers), or bring over the ones you set up in Claude Desktop or Cursor, and Skills (standard `SKILL.md` folders), three of them built in; a tool that could change something, or a Skill's script, runs only after you say yes.

**English and Chinese.** The whole app is in both.

## Your files stay yours

There's no account and no IncarnaMind server. Your Minds and the search index stay in one data folder on your computer, and your files stay where you keep them, unchanged. A question and the passages found for it go only to the AI model you picked, or nowhere with a local model. API keys are encrypted in your system's secure storage. Usage data is collected only if you agree, as [anonymous events](docs/privacy.md#usage-data), never your files or questions, and you can turn it off any time in **Settings → Privacy**. Crash reports are off unless you turn them on, and are stripped of your content.

- **You're asked before anything leaves.** The first time IncarnaMind would send anything to an outside service, it shows what would be sent and to whom, and waits for your OK. **Settings → Privacy** lists each one and lets you take it back.
- **Providers are asked not to keep it.** Where a provider offers the choice, IncarnaMind asks it not to store your requests; every request to OpenAI and xAI says so, and requests to OpenRouter ask it to use only providers that keep and train on nothing.
- **Nothing runs without you.** Before a Connector's tool that could change something, or a Skill's script, runs, IncarnaMind asks you: allow it once, always allow it, or deny it. **Settings → Privacy** lists every rule you've set, and you can revoke any of them.

<details>
<summary><b>Questions</b></summary>

- **Is it free?** Yes: it's open source under the Apache 2.0 licence. With a cloud AI model, you pay its provider for what you use; a local model costs nothing.
- **Do I need an API key?** No. Continue with ChatGPT on a Plus or Pro plan, or, if [Ollama](https://ollama.com) is running, download a local model in one click. If you pay for Claude, connect the Claude app in **Settings → Agents** (experimental) and ask your Documents there. Or paste a key from OpenAI, Anthropic, Google, DeepSeek, Qwen, SiliconFlow or OpenRouter, and more in Settings, such as Kimi, GLM, Mistral, xAI or any OpenAI-compatible server.
- **Is it an alternative to NotebookLM (now Gemini Notebook)?** Yes. IncarnaMind runs on your computer and your files stay there, it works with the model you choose (your own key, a ChatGPT plan, or a local model through Ollama), every citation is checked against the page it quotes, and a Mind exports to Word with its citations.
- **Can I use my ChatGPT plan?** Yes, with ChatGPT Plus or Pro. In **Settings → Models**, choose **Continue with ChatGPT** and sign in in your browser: Questions then go to OpenAI under your account and count against your plan's limits, with no key.
- **Does it work offline?** Yes, with a local model through Ollama: reading your files, search and the citation check all run on your computer. Cloud models need the internet. Meaning-based search (embeddings) is optional and off by default.
- **Will it change my files?** No. It reads them where they are, and writes only to its own data folder and the files you export.
- **What does "Quote found" mean?** That the quoted words are in the place the citation names, not that they prove your sentence, which is why the page is one click away. "Can't check" means there's no text to compare against, as on a scanned page.
- **How do I back up?** Copy the data folder: it holds your Minds, the search index and your settings.

</details>

## Coming next

Planned, not available yet, with no dates (see the [roadmap](https://github.com/junruxiong/IncarnaMind/issues/74)): asking during a meeting, with answers that cite its live transcript; the assistant editing your Minds, each change waiting for you to accept it; Word files opened and exported back with tracked changes; your drafts, and other people's documents, checked against your files; comparison tables across documents; and research Tasks that run in the background.

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

[Apache License 2.0](LICENSE). The fonts that ship with the app (Source Serif 4, Source Sans 3 and JetBrains Mono) are under the [SIL Open Font License 1.1](https://openfontlicense.org). The citation styles and locales come unmodified from the [Citation Style Language](https://citationstyles.org/) project, under [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/); they are formatted by [Hayagriva](https://github.com/typst/hayagriva) (MIT or Apache-2.0). Their notices are in [`resources/notices`](resources/notices).
