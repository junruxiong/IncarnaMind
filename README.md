<!-- README.md and README.zh-CN.md match section for section. Change them together. -->

<p align="center"><b>English</b> · <a href="README.zh-CN.md">简体中文</a></p>

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="build/icon/svg/mac-dark.svg">
    <img src="build/icon/svg/mac-light.svg" width="96" height="96" alt="">
  </picture>
</p>

<h1 align="center">IncarnaMind</h1>

<p align="center">
  <b>Your files, your AI: write, ask, check and compare in one place.</b><br>
  Every sentence and figure checked against your own documents, with the AI model you choose. Free and open source, on your computer.
</p>

<p align="center"><a href="https://tally.so/r/rjN2jL?ref=readme"><img src="docs/images/notify-en.svg" width="278" height="40" alt="Get notified when the alpha opens"></a></p>

<p align="center">
  <a href="#download"><b>Download</b></a> ·
  <a href="#what-you-can-do">What you can do</a> ·
  <a href="#your-files-stay-yours">Privacy</a> ·
  <a href="https://github.com/IncarnaMind/IncarnaMind/discussions">Discussions</a>
  <br><br>
  <img alt="Desktop app for macOS, Windows and Linux" src="https://img.shields.io/badge/macOS%20%C2%B7%20Windows%20%C2%B7%20Linux-desktop%20app-17191C">
  <a href="LICENSE"><img alt="License: Apache-2.0" src="https://img.shields.io/github/license/IncarnaMind/IncarnaMind?color=17191C"></a>
  <!-- After the first release, add: <a href="https://github.com/IncarnaMind/IncarnaMind/releases/latest"><img alt="Latest release" src="https://img.shields.io/github/v/release/IncarnaMind/IncarnaMind?color=17191C"></a> -->
</p>

<p align="center"><img src="docs/images/demo.gif" width="880" alt="Six screens in turn: two sentences of a draft selected, with Check against Documents in the menu; the same draft checked, each sentence marked supported, unsupported or contradicted, with a count under the title; a Citation opened beside the note, its quote highlighted on the PDF page; a lender update someone else wrote, opened to verify, with Check 8 claims; its claims checked, one contradicted by the facilities agreement on page 3; and the checked draft in the Chinese interface."></p>

<p align="center"><sub>Everything shown is in the app today.</sub></p>

**IncarnaMind** is a free, open-source desktop app for macOS, Windows and Linux where you write with AI and ask questions about your own documents (PDF, Word, PowerPoint, Excel and more). It can check every sentence and figure in a draft against those documents, whoever wrote it, and every answer cites the exact page it comes from. It works with the AI you already use: your ChatGPT plan or your own API key.

## Download

**IncarnaMind isn't released yet.** Once it is, the [Releases page](https://github.com/IncarnaMind/IncarnaMind/releases) will have installers for macOS (Apple silicon and Intel), Windows and Linux. To hear when, [leave your e-mail](https://tally.so/r/rjN2jL?ref=readme) or click **Watch → Custom → Releases** at the top of this page. To try it now, run it from source with Node.js 24 or newer:

```shell
git clone https://github.com/IncarnaMind/IncarnaMind.git && cd IncarnaMind
npm install && npm run dev
```

Windows releases will be signed through the SignPath Foundation's free signing for open-source projects; until then the installer is unsigned. See the [Code signing policy](docs/code-signing-policy.md).

<details>
<summary>If your system won't open it</summary>

- **Windows** says "Windows protected your PC": the installer isn't code-signed yet. Click **More info**, then **Run anyway**.
- **macOS** says it can't verify IncarnaMind: until the app is notarized, open it once and click **Done**, then **System Settings → Privacy & Security → Open Anyway**. Or run `xattr -dr com.apple.quarantine /Applications/IncarnaMind.app`. An unsigned Mac app can't update itself: IncarnaMind tells you when a new version is out and offers its download page.
- **Linux**: run `chmod +x IncarnaMind-*.AppImage`. On Ubuntu 22.04 or later, also `sudo apt install libfuse2` (`libfuse2t64` on 24.04 and later).

</details>

## What you can do

**Check your draft before you send it.** Select sentences in a Mind and choose **Check against Documents**: each one comes back supported, unsupported or contradicted by your Documents, with the page one click away, and the line under the title counts what still needs a look. A PDF someone else wrote can be checked the same way, claim by claim: **+ → Verify a document…**.

**Write and ask in one place.** A Mind is a notebook for one topic. Beside it sits the assistant: ask it about the Mind you're in or your files, and the Answer comes back there, in a conversation, with its citations checked. Your draft stays as you wrote it until you pick **Add to note**, which puts the Answer at your cursor as text you can edit, citations and all.

**Check every answer.** Answers cite where each claim came from and carry the quote, which is checked against that place: a green check if it's there, an amber mark if it isn't, and a click opens the page with the quote highlighted. Unfold an Answer's steps to see what it searched and which sources it used.

**Take notes in a meeting.** Start recording and a live transcript runs beside your Mind, transcribed on this computer. When the call ends, save it as a Document, and your notes can cite what was said.

**Use the files you already have.** Link the folders you already use, even Zotero's storage folder: IncarnaMind reads the PDF, Word (comments included), PowerPoint, Excel, CSV, Markdown and plain-text files in them where they are, never moves or changes them, and keeps up as they change.

**Find and add sources.** Search OpenAlex, arXiv and PubMed and add open-access papers as Documents, save a web page by its address so it can be cited and checked like a file, and import your references from Zotero or any CSL JSON, BibTeX or RIS file.

**Write with AI you can check.** Tell the assistant what to change ("shorten the summary", "add today's decision to the project plan", "summarise these papers into a new note") and it changes the Mind, or makes a new one, as changes waiting for you to accept or reject, coming in paragraph by paragraph as it writes; a question still gets an answer. Or select text and choose Edit with AI, and each change says why it was made. Turn on suggestions as you write, and when you pause, grey text offers how the paragraph could go on, only where your Documents support it and always with a citation; Tab accepts.

**Keep a big library tidy.** Tags are added for you, IncarnaMind suggests a Folder for each Document for you to accept, and the choices you make yourself are kept.

**Choose your AI.** Already pay for ChatGPT Plus or Pro? Continue with ChatGPT and ask on your plan, no key needed. Or use your own key from OpenAI, Anthropic, Google, DeepSeek, Qwen, SiliconFlow or OpenRouter, and more, such as Kimi, GLM, Mistral, xAI or any OpenAI-compatible server.

**Hand it in.** Export a Mind to Word or Markdown, with its citations and bibliography in the style you pick: APA, MLA, Chicago, Harvard, IEEE, Vancouver, GB/T 7714 or IncarnaMind's numbered style. A citation that didn't pass the check is marked "[unverified]". Open a Word file as a Mind and export it back into the same file: the parts you didn't touch stay exactly as they were, and your edits can go in as tracked changes.

**Compare documents in a table.** **Compare Documents…** puts the same questions to each file you pick and fills a table, every cell a cited, checked answer.

**Connect apps and Skills that ask first.** Add Connectors (MCP servers), or bring over the ones you set up in Claude Desktop or Cursor, and Skills (standard `SKILL.md` folders), three of them built in; a tool that could change something, or a Skill's script, runs only after you say yes.

**English and Chinese.** The whole app is in both.

## Your files stay yours

You don't need an account. Your Minds and the search index stay in one data folder on your computer, and your files stay where you keep them, unchanged. A question and the passages found for it go only to the AI model you picked, or nowhere with a local model. API keys are encrypted in your system's secure storage. Usage data is collected only if you agree, as [anonymous events](docs/privacy.md#usage-data), never your files or questions, and you can turn it off any time in **Settings → Privacy**. Crash reports are off unless you turn them on, and are stripped of your content.

- **You're asked before anything leaves.** The first time IncarnaMind would send anything to an outside service, it shows what would be sent and to whom, and waits for your OK. **Settings → Privacy** lists each one and lets you take it back.
- **Providers are asked not to keep it.** Where a provider offers the choice, IncarnaMind asks it not to store your requests; every request to OpenAI and xAI says so, and requests to OpenRouter ask it to use only providers that keep and train on nothing.
- **Nothing runs without you.** Before a Connector's tool that could change something, or a Skill's script, runs, IncarnaMind asks you: allow it once, always allow it, or deny it. **Settings → Privacy** lists every rule you've set, and you can revoke any of them.

<details>
<summary><b>Questions</b></summary>

- **Is it free?** Yes: it's open source under the Apache 2.0 licence. The AI model is billed by its provider: use the ChatGPT plan you already pay for, or your own API key and pay for what you use.
- **Do I need an API key?** Not if you have ChatGPT Plus or Pro: choose Continue with ChatGPT and ask on your plan. Or paste a key from OpenAI, Anthropic, Google, DeepSeek, Qwen, SiliconFlow or OpenRouter, and more in Settings, such as Kimi, GLM, Mistral, xAI or any OpenAI-compatible server.
- **Is it an alternative to NotebookLM (now Gemini Notebook)?** Yes. IncarnaMind runs on your computer and reads your files where they are, and no account is needed. It works with the model you choose (your ChatGPT plan or your own key), every citation is checked against the page it quotes, your own sentences can be checked too, and a Mind exports to Word with its citations.
- **Can I use my ChatGPT plan?** Yes, with ChatGPT Plus or Pro. In **Settings → Models**, choose **Continue with ChatGPT** and sign in in your browser: Questions then go to OpenAI under your account and count against your plan's limits, with no key.
- **Does it work offline?** Reading your files and search do. Answers and checks need an AI model, which for most people means the internet. A local model through [Ollama](https://ollama.com) can answer offline, but more slowly and less reliably than a cloud model. Meaning-based search (embeddings) is optional and off by default.
- **Will it change my files?** No. It reads them where they are, and writes only to its own data folder and the files you export.
- **What does "Quote found" mean?** That the quoted words are in the place the citation names, not that they prove your sentence, which is why the page is one click away. "Can't check" means there's no text to compare against, as on a scanned page.
- **How do I back up?** Copy the data folder: it holds your Minds, the search index and your settings.

</details>

## Coming next

Planned, not available yet, with no dates (see the [roadmap](https://github.com/IncarnaMind/IncarnaMind/issues/74)): asking during a meeting, with answers that cite its live transcript; a reviewing pane for other people's tracked changes in a Word file; and research Tasks that run in the background.

## Contributing

Code, testing with your own documents, translations and ideas are all welcome: start with [CONTRIBUTING.md](CONTRIBUTING.md), [open an issue](https://github.com/IncarnaMind/IncarnaMind/issues/new/choose) or ask in [Discussions](https://github.com/IncarnaMind/IncarnaMind/discussions).

## History

IncarnaMind began in 2023 as a Python command-line tool for chatting with your PDFs, starred by about 800 of you, and was rebuilt in 2026 as a desktop app around one idea: an answer from your files is only useful if you can check it. The old tool lives on at the [`v1-python-cli`](https://github.com/IncarnaMind/IncarnaMind/tree/v1-python-cli) tag. Thank you for sticking around, and if IncarnaMind is useful to you, a star helps other people find it.

<details>
<summary>Citing IncarnaMind</summary>

```bibtex
@misc{IncarnaMind2023, author = {Junru Xiong}, title = {IncarnaMind}, year = {2023},
  publisher = {GitHub}, journal = {GitHub Repository},
  howpublished = {\url{https://github.com/IncarnaMind/IncarnaMind}}}
```

</details>

## Licence

[Apache License 2.0](LICENSE). The fonts that ship with the app (Source Serif 4, Source Sans 3 and JetBrains Mono) are under the [SIL Open Font License 1.1](https://openfontlicense.org). The citation styles and locales come unmodified from the [Citation Style Language](https://citationstyles.org/) project, under [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/); they are formatted by [Hayagriva](https://github.com/typst/hayagriva) (MIT or Apache-2.0). Their notices are in [`resources/notices`](resources/notices).
