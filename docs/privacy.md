# Privacy

What IncarnaMind sends from your computer, and how to change it. **Settings → Privacy** in the app lists the same, live.

- **Your content** (a Question and the Passages found for it, Document text for cloud embeddings, Tool calls to a Connector, the text you select for Edit with AI with the paragraphs it is in, and, once you turn on suggestions as you write, the paragraph you pause in with the Passages that match it) goes only to the service you picked, and only after you allow it. Edit with AI asks once for its own flow, and in local mode uses only a model on this computer. Settings → Privacy lists every flow and lets you revoke it.
- **AI agents you connect** (experimental, off by default): an agent you run yourself, such as Claude Code or Codex, can search and read the Documents and Minds in the folders you share with it, and ask IncarnaMind Questions. What it reads goes on to the AI model the agent uses, under that service's terms, not IncarnaMind's; some consumer plans may use it to train their models. You allow each agent once when you connect it, Settings → Privacy lists it, disconnecting it stops it at once, and local mode pauses every agent. How to connect each agent: [agents.md](agents.md).
- **Look up** in a Document's "Cite as" sends its DOI to Crossref, or its arXiv ID to arXiv, and nothing else from the Document, only when you click it, after you allow it once. It is off in local mode.
- **Find sources** (the Library's "Add Documents ▾", the "+" menu, or an Answer asked to find papers) sends what you search for, the words and filters such as years, to each provider it searches: OpenAlex, arXiv and PubMed, and Crossref for a DOI you type. openFDA, DailyMed and ClinicalTrials.gov, for drug labels and clinical trials, are searched only when you choose them (or type an NCT number), and are sent the drug, condition or number you typed. If you add your own openFDA key, it is kept in your keychain and sent only to openFDA, in a header. Adding a work sends its DOI, arXiv ID, PubMed ID, label set ID or NCT number to the provider that gives its copy. Nothing from your Documents or Minds is sent. Each provider asks once, before its first request; one you don't allow is never sent anything. An open-access copy on a publisher's or a repository's site is downloaded once when you add it, without cookies, as adding a web page is: only its address is requested. Find sources is off in local mode.
- **Traffic without your content**: update checks, model downloads, and remote Connectors' sign-in.
- **Web pages you add** are fetched from your computer, straight from their site, when you add one and when you choose Fetch again. Nothing else on the page is loaded: not its pictures, scripts or trackers.
- **Crash reports**: off unless you turn them on, and stripped of your content first.
- **Usage data**: only if you agree, described below.
- **The AI use record** stays on this computer and is never sent, whatever you choose about usage data. It notes each time the AI wrote into a Mind and what you did with it, as sizes, kinds, the model's name and times, with no text. Clear it in Settings → Privacy.
- **A Word file you import as a Mind** is read on this computer and sends nothing. A copy of it is kept in your data folder ("Word imports"), so export can write your changes back into it after the file has moved; it is deleted once no Mind made from it is left, Recently deleted included.
- **The cost record** stays on this computer and is never sent. For each request to a cloud model it notes the model, the tokens its provider reported and what that cost at its price, with no text, for this month's figure and the monthly cap in Settings → General.

## Usage data

Usage data is a small set of anonymous product events, such as "a Question was asked with a local model" or "an Answer took 6 seconds and has 3 Citations, each quote found". It shows which features are used and where people get stuck. It never holds your files or what you write.

### When it's sent

- **Only from a build made with an analytics project.** A build without one (from source, by default) sends nothing, asks nothing, and doesn't load the analytics code.
- **A release asks once, on the first run.** Nothing is sent unless you agree; the default is no.
- **A test build, for invited testers,** sends usage data until you turn it off, as testers agree to when they join. Its first run says so, with the switch to turn it off.
- **Local mode** ("Keep everything on this computer", in Settings → Search) turns usage data off, and it stays off.
- **You can change it any time** in Settings → Privacy, under "Usage data". Turning it off stops sending at once and drops anything still waiting to be sent.

### What's never sent

Your Documents, their text or names, Folder and Tag names, Questions, Answers, quotes, Notes, API keys, file paths, the names of Connectors or Skills you added, an account, an e-mail address, your device's serial number, your IP address or your location. Nothing is recorded automatically: there is no session replay, and no capture of clicks or typing. Every event is sent explicitly by the app's code.

### How it's sent

Events go to [PostHog](https://posthog.com), an open-source analytics service, to the project the maintainer set in the build (PostHog's EU cloud or a self-hosted server), using PostHog's open-source Node.js library in the app's main process. The window never talks to the network itself.

- Each event carries a random install ID, made the first time an event is sent. It lets one install's steps be followed, and nothing else. **Reset install ID** in Settings → Privacy gives a new one.
- No person profile is made, no location is looked up, and PostHog stores `0.0.0.0` instead of your IP address.
- Events wait in a short queue in memory (at most 100), and are sent together every 30 seconds or every 20 events. If a send fails, for example offline, those events are dropped. Nothing is written to disk.

### Every event

Every field is a value from a closed list, a whole number, or true/false: there's no field for text. The app checks each event against this list before it's sent ([`src/core/usageEvents.ts`](../src/core/usageEvents.ts)), and nothing else can be sent.

| Event | Fields |
| --- | --- |
| `app_opened` | `language`: the interface language, `en` or `zh-CN` |
| `onboarding_picked` | What the first run's "What will you use IncarnaMind for?" question was answered with, once that question exists: `papers_research`, `reports_analysis`, `contracts_legal`, `meetings_notes`, `everything`, `something_else`, `skipped`, each true or false |
| `mind_created` | No fields: you made a Mind |
| `question_asked` | `provider`: the kind of provider of the model (`openai`, `anthropic`, `google`, `deepseek`, `qwen`, `kimi`, `glm`, `siliconflow`, `mistral`, `xai`, `openrouter`, `openai-compatible`, `ollama`, `chatgpt`); `local`: whether the model runs on your computer. Never the Question or the model's prompt |
| `answer_finished` | `outcome`: `done` or `stopped`; `duration_ms`: how long it took; `citations`: how many Citations it has; `found`, `not_found`, `cant_check`: how many of them by check |
| `citation_opened` | `check`: the Citation's check, `found`, `not-found`, `unsupported`, `contradicted`, `cant-check` or `checking` |
| `documents_added` | `documents`: how many new Documents; `already_added`: files that were in already; `skipped`: files that couldn't be added; `pdf`, `docx`, `pptx`, `xlsx`, `csv`, `markdown`, `text`, `web`, `transcript`: the new Documents by format (`web`: web pages saved by their address; `transcript`: WebVTT Transcripts, a Meeting's or a file's). Never their names or addresses |
| `folder_linked` | No fields: you linked a folder. Never its name or path |
| `organize_run` | `documents`: how many Documents; `all`: whether it was all of them; `model`: the kind of model, `chat`, `jev`, `auto` or `ollama` |
| `mind_exported` | `format`: `markdown`, `docx` or `pdf`; `questions`: whether Questions were included |
| `connector_used` | `remote`: whether the Connector is remote. Sent once per Connector and Answer. Never its name: every Connector is one you added |
| `skill_used` | `skill`: a built-in Skill's name (`literature-review`, `mind-to-report`, `summarise-document`), or `own` for one you added, whatever its name; `forced`: whether the Question asked for it. Sent once per Skill and Answer |
| `error_occurred` | `area`: `answer`, `document` or `connector`; `kind`: the kind of error, such as `rate-limit`, `password-protected` or `timed-out`. Never its message |
| `settings_page_viewed` | `page`: which page of Settings, such as `privacy` |

Every event also carries:

| Field | Value |
| --- | --- |
| `app_version` | The app's version, e.g. `0.1.0` |
| `os` | `darwin`, `win32`, `linux` or `other` |
| `arch` | `arm64`, `x64` or `other` |
| `build` | `release`, or `tester` for a test build |

And PostHog's library adds the same values to every event: `$process_person_profile: false` (no person profile), `$geoip_disable: true` (no location), `$ip: "0.0.0.0"` (no IP address), and `$lib` and `$lib_version` (the library's name and version).

### For maintainers

A build sends usage data only when it's built with both of these set (electron-vite reads them from the environment or a `.env.local` file, never from the code):

- `MAIN_VITE_POSTHOG_KEY`: the PostHog project's API key (`phc_…`), which can only send events.
- `MAIN_VITE_POSTHOG_HOST`: the project's host, over https, e.g. `https://eu.i.posthog.com` or a self-hosted server.

`MAIN_VITE_TESTER_BUILD=1` makes a test build, where usage data is on until the User turns it off. The release workflow sets it only for tags with `-tester` in them (`v0.9.0-tester.1`); an `-alpha` or `-beta` tag makes a release build, which asks first. See [releasing.md](releasing.md).

In the PostHog project, turn on **Discard client IP data** too (it's the default for organisations in the EU cloud).
