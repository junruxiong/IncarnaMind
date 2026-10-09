# The agent core is general; the product stays documents-first

IncarnaMind v1 stays a documents-first notebook: an Answer is written into a Mind, with Citations checked against the Documents. The agent core underneath is shaped so that general-agent work can be added later without rewriting it. That work includes files outside Linked folders, a browser, a shell, background and scheduled Tasks, and more Connectors. The User asked for this on 2026-10-09: "I want extensibility from the very start". The design is `docs/designs/agent-extensibility.md`.

The core is built from four parts:
- **Tool providers.** Every Tool has one shape and comes from a Tool provider: Document search, Skills and each Connector now; later Files, Library, Minds, Browser, Shell and Schedules. A provider the host can't support isn't registered, so the hosted version simply lacks a local shell.
- **Effects.** Each Tool says what a call does (read, write, execute or network) and where: a folder, a host, a Connector's service, the Documents or a Mind. IncarnaMind works out the Effects itself; a Connector's annotations only narrow them, because the MCP specification calls them untrusted hints.
- **Approvals by Effect.** Approvals are decided from Effects, with the existing policies, requests and "always allow". A policy can trust a Tool, or a scope such as "write inside ~/Reports". Once a Run has read untrusted content (Passages, web pages, Connector results), only scope policies let side effects run without asking. Documents are untrusted input and the User's data is private, so an injected instruction must never meet an unbounded "always".
- **Runs.** The Tool-calling loop is a Run: instructions, Tools, a model, events, stop and steer. An Answer is one kind of Run. A Task is a later kind, which runs in the background, survives a restart from its journal in SQLite, and outputs its Effects and a log.

Every process a Tool starts goes through an `Executor` with a sandbox level. Skill scripts use level "none" today. The OS sandbox (Seatbelt, bubblewrap), a container, or a remote machine come later, without changing the Tools that use them.

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
- A Files provider never writes into a Linked folder (ADR-0010).
- On macOS, the OS sandbox relies on `sandbox-exec` and Seatbelt profiles, which Apple deprecates and doesn't document for third parties; Claude Code and Codex rely on them too. Windows stays at sandbox level "none" and asks every time.
