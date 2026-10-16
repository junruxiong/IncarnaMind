# The agent core is general; the product stays documents-first

IncarnaMind v1 stays a documents-first notebook: an Answer is written into a Mind, with Citations checked against the Documents. The agent core underneath is shaped so that general-agent work can be added later without rewriting it. That work includes files outside Linked folders, a browser, a shell, background and scheduled Tasks, and more Connectors. The User asked for this on 2026-10-09: "I want extensibility from the very start". The design is `docs/designs/agent-extensibility.md`.

The core is built from four parts:
- **Tool providers.** Every Tool has one shape and comes from a Tool provider: Document search, Skills and each Connector now; later Files, Library, Minds, Browser, Shell and Schedules. A provider the host can't support isn't registered, so the hosted version simply lacks a local shell.
- **Effects.** Each Tool says what a call does (read, write, execute or network) and where: a folder, a host, a Connector's service, the Documents or a Mind. IncarnaMind works out the Effects itself; a Connector's annotations only narrow them, because the MCP specification calls them untrusted hints.
- **Approvals by Effect.** Approvals are decided from Effects, with the existing policies, requests and "always allow". A policy can trust a Tool, or a scope such as "write inside ~/Reports". Once a Run has read untrusted content (Passages, web pages, Connector results), only scope policies let side effects run without asking. Documents are untrusted input and the User's data is private, so an injected instruction must never meet an unbounded "always".
- **Runs.** The Tool-calling loop is a Run: instructions, Tools, a model, events, stop and steer. An Answer is one kind of Run. A Task is a later kind, which runs in the background, survives a restart from its journal in SQLite, and outputs its Effects and a log.

Every process a Tool starts goes through an `Executor` with a sandbox level. Skill scripts use level "os", the OS sandbox (Seatbelt, bubblewrap), on macOS and Linux (#65), and "none" on Windows (and on Linux without bubblewrap). A container or a remote machine come later, without changing the Tools that use them.

The loop library sits behind a `RunEngine` interface. Today AI SDK 7 implements it, with `streamText`, `stopWhen` and `prepareStep`: the loop its `ToolLoopAgent` wraps. Which library implements it is decided by measurement: the bake-off on `prototype/engine-bakeoff` (AI SDK 7, pi-coding-agent, pi-agent-core with pi-ai), run behind this same interface and recorded in ADR-0007.

Nothing above the interface imports the loop library. That covers Tools, approvals, Runs, the Executor, storage and the UI. Run histories are kept in IncarnaMind's own message form.

The model layer stays on the AI SDK whichever library runs the loop. It builds the models, asks for consent, sorts provider errors into kinds, and makes one-pass calls such as structured output.

Waiting for approval happens above the engine, in a `gate` the engine awaits before each Tool call. A Task's pending approval is written to its journal, so it survives a restart without any engine feature.

## Considered options

- **Keep the core Answer-only and add capabilities as they come** (the previous position). Rejected. Each capability would touch the loop, the approval subjects, the Tool-call record and the renderer, and safety would be decided one Tool at a time.
- **Become a general desktop agent now, like WorkBuddy.** Still rejected for the product. v1's promise is checkable Citations, and general capabilities are deferred.
- **Build the layers above the loop on a coding agent's SDK** (pi-coding-agent's sessions, built-in Tools and permission hooks; opencode's permission config). Rejected. They would replace Effects, approvals and storage that must keep the keychain rule, ADR-0003's sync rules and read-only Linked folders. They also assume a repository and an unconfined shell: pi's own security guide says it runs with the account's permissions, doesn't ask before each Tool call, and leaves isolation to a container, a managed sandbox or a micro-VM extension. As the loop alone, pi is a contestant in the bake-off.
- **Choose the loop library by argument now.** Not done. The bake-off measures the contestants behind the same interface. With the interface in place, switching later costs about the same as switching now.
- **Approve by Tool only, as Goose and v1 do.** Rejected for side effects after untrusted content. "Always allow this Tool" trusts every argument the model will ever send. A scope bounds the damage instead.
- **A shell in v1.** Rejected. None of the general tasks we expect needs one when there are narrow Tools and sandboxed scripts, and an unsandboxed shell plus untrusted Documents plus "always allow" is the injection path this ADR closes.

## Consequences

- Five refactors land before more agent features build on today's shapes: one Tool shape; Effects with one approval decision; the Run engine interface with a contract suite every engine must pass; taint recorded; and the execution seam. Each keeps v1's behaviour and stored data.
- v1's existing "always" policies keep working; taint is only recorded for them for now. When the OS sandbox lands, an unsandboxed Skill script's "always run" asks again after untrusted content; sandboxed scripts keep running without asking. Connector "always allow" stays until there is evidence. New side-effecting providers follow the taint rule from the start.
- The first general capability, when one is scheduled, is Files (a granted read folder and a granted output folder) with Library `add_documents` and Schedules. A Shell comes only after the OS sandbox, and never with Tool-level "always".
- The sandbox follows the industry pattern and stays simple (User, 2026-10-09: "do it the industry-standard way, simple and clean, no over-engineering"; research in `docs/research/sandboxing.md`). Skill scripts run at the "os" level through `@anthropic-ai/sandbox-runtime`, pinned to 0.0.79, on macOS and Linux (#65): Seatbelt or bubblewrap, no network, secret-looking environment variables removed. stdio Connectors stay unsandboxed, as in Claude Code, Cursor, Codex and Muse Code. No extra Connector hardening (environment allowlists, version pinning, description hashing) is built.
- A later Task's output goes to a Tasks view with its log, and its deliverable to where the User chose (a Mind, a folder or the Library), not all into a Mind.
- Effect policies on folders belong to one device: they get a `device_id`, unlike today's per-User rows (ADR-0003).
- Task tables (`runs`, `run_steps`, `schedules`) follow ADR-0003's rules and record the device a Run executes on. "Run" and "Task" enter CONTEXT.md when Tasks are built; "Effect" is in it now.
- A Files provider never changes a file in a Linked folder, and adds a new one there only with that folder's permission (ADR-0010, amended 2026-10-10 by #124).
- On macOS, the OS sandbox relies on `sandbox-exec` and Seatbelt profiles, which Apple deprecates and doesn't document for third parties; Claude Code and Codex rely on them too. Windows stays at sandbox level "none" and asks every time.

## Addendum, 2026-10-11: the User's own agent

The owner asked why we write our own harness rather than use Claude Code's, Codex's, pi's or OpenClaw's, and suggested Users connect their own. We keep our own loop. A User's agent can use IncarnaMind in two ways, both outside that loop. Research: `docs/research/bring-your-own-agent.md`.

**We keep our loop.** The loop is already the AI SDK's. What we write is the Citation check, consent, Proposals, Effects and read-only Linked folders. An outside harness as our engine would drop them: Claude's ACP adapter uses its own file and shell Tools whatever the client allows. Most Users won't have a coding agent, and many use a local model.

**IncarnaMind is an MCP server.** A User can connect their own agent, such as Claude Code, Codex, Cursor or Claude Desktop, to IncarnaMind.
- It runs as a stdio command that talks to the running app over a local socket. There is no HTTP port until a client needs one, because stdio and a socket are what the MCP security guide recommends for local servers.
- Each agent is a pairing the User makes in Settings, with its own Search scope, its own consent, a token only it holds, and a list of what it did.
- Its Tools read the Documents and the Minds and mint Citations. Its only write is a Proposal, which waits for the User in IncarnaMind.
- Only IncarnaMind mints a Citation. Every Citation an agent sends back is checked again, and a sentence without one is marked.
- Off by default, and offered as an experimental setting first. A new pairing sees no Documents until the User picks its Folders. Local mode pauses every outside agent, because IncarnaMind can't tell which model the agent sends text to.

**IncarnaMind can host an outside agent over ACP, later and experimentally.** A hosted agent is a new kind of Run whose loop runs in the agent's process. It isn't a Run engine, because the model isn't ours.
- The agent gets IncarnaMind's MCP server at session start, and its text streams into an Answer Block with re-checked Citations.
- Its permission requests use our approval cards. After our Tools return Passages to it, we stop offering "always allow".
- We start it in its most restrictive mode, and never in one that skips permissions.
- Its own Tools are outside our Effects, and we say so in its consent.

**Vendor terms decide which agents.**
- We never show a vendor's sign-in, read its tokens, or bundle its proprietary binary.
- Codex, Goose, OpenCode, and Gemini CLI on a paid key are hosted first.
- Claude is not hosted for now. Claude Code and Claude Desktop connect through the MCP server above, on the User's own sign-in. Hosting it can be reconsidered if Anthropic confirms subscription use in a third-party app.

**Considered options.**
- **Run the User's harness as our engine.** Rejected for the reasons above.
- **A localhost HTTP server with a token.** Kept for later. It opens a port that web pages can try to reach, and no client we target needs it today.
- **Let agents write to Minds directly.** Rejected. A change from outside waits as a Proposal, as an AI change from inside does.
- **Trust ACP's `fs` and `terminal` capabilities to confine the agent.** Rejected. Agents use their own Tools when the client declines, and ACP v2 removes them.

**Consequences.**
- A new data flow, "agents", one consent per pairing, listed on the Privacy page.
- A pairings table, following ADR-0003's rules, with each token kept by `secrets.ts`.
- The socket listener lives in the main process, and the core never imports Electron.
- "Outside agent" enters CONTEXT.md when Phase 1 ships.
- `@agentclientprotocol/sdk` (Apache-2.0) is added only with the ACP host.

## Amendment, 2026-10-11: third-party text asks before "always" (#78)

Web pages and documents other people wrote are now main inputs, so the consequence above that Connector "always allow" stays until there is evidence no longer holds after reading them. **Third-party text** (CONTEXT.md) is what someone other than the User wrote: a saved web page, the document being checked, and text in a Mind or a conversation that came from them (imported from someone else's .docx, a quote added from a web page, an Answer written by a Run that had read some). Text pasted from other apps can't be traced, and counts as the User's.

- **The rule.** Once a Run has read third-party text, a Tool-level "always" (a Connector Tool's "always allow", a Skill's "always run") no longer covers a call that writes or sends data. The call asks, the approval card says why ("This Answer read a web page", "This Mind contains imported text"), and it offers no "always". A call that runs by default still runs (a Connector Tool marked read-only), and so does an "always run" script in the OS sandbox, which can neither write outside its working folder nor go online.
- **Unchanged.** In a Run that read only the User's own Documents, "always" still counts, as above. New side-effecting Tools keep the day-one rule: they ask unless a scope grant covers them, and until their provider can ask, the Answer's gate refuses their side effects.
- **When a Run is third-party.** From its start, when its Question context holds third-party text, or when it checks a document (#97); and from the moment one of its Tools reads some, such as a search that finds a web page's Passages. Calls from an outside agent, the User's own agent connected to IncarnaMind (#351), are untrusted input of the same kind: an Answer one asks for is offered no Tool that writes or sends, so no "always" can be steered by it.
- **Origin is kept with the text.** A Mind records, beside its Blocks, which Blocks hold third-party text and where it came from (ADR-0003, amended); a conversation's Answer records it on its row. An Answer written by a third-party Run is recorded too, so a later Question that has it in its context counts as third-party as well.
- **Check Runs get two Tools.** Find support, verify, comparison tables' cells and contradiction watch are offered only Document search and `cite`, whatever else is set up, so a document being checked can't steer a write or a send. Ordinary Answers keep every Tool.

## Amendment, 2026-10-12: where an Answer is written (ADR-0017)

The opening says an Answer is written into a Mind. With ADR-0017 and #201, an Answer is written in the assistant's conversation instead. A Mind gets it only when the User picks Add to note, or as Proposals the User accepts (#202). The product stays documents-first: wherever an Answer is shown, its Citations are checked against the Documents.

## Amendment, 2026-10-12: web search's Tools (#128)

Web search gives an Answer two Tools of IncarnaMind's own, offered only when the User has it on in Settings → Search, Local mode is off, the Answer's model's provider has a web search of its own, and the Question asks for the web. `search_web` searches through that provider; `read_web_page` saves a page the search found as a Web page Document and gives its Passages to cite.

- **Neither asks the User.** A search sends to a service the User allowed (consent asks before the first). Reading a page is a read from a web host: it fetches only an address a search of the same Run returned, so nothing the model writes, and nothing of the User's, goes with it. An address the model made up is refused with a reason. "Read" from a web host therefore runs by default (`runsByDefault`); a later Tool that fetches an address the model chose sends data, and declares "network". The Research Task's design board (Final-42-Research-Running) shows `fetch_page` asking after a web page was read; with the allowlist there is nothing for that ask to protect, so #130 should revisit it rather than add it.
- **Both read third-party text.** Search results (titles written by other people) and a page's Passages make the Run third-party (#78): from then on, writes and sends a Tool-level "always" would let through ask.

## Amendment, 2026-10-16: edits the User asked for apply at once (#880)

An edit the User asks for on the text (Edit with AI, the selection menu's AI actions, a change asked for in the composer with text selected) applies at once, with one Undo and a line over it saying what changed. Changes the User didn't ask for stay Proposals, and so does an edit the guards hold back: one that would remove most of the selection without mapping onto it. The core still writes every AI change as Proposals; the window that asked accepts them, and they stay Proposals otherwise.
