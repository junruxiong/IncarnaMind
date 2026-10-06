# The desktop shell is Electron

We use Electron, not Tauri, for the desktop app (ADR-0004). Tauri's installers are much smaller (roughly 10–15 MB against 100 MB or more), which fits the "lightweight" goal. But IncarnaMind connects to MCP servers and runs skills (ADR-0007). Most MCP servers are local Node or Python processes that the app has to start and talk to over stdin/stdout, and the official MCP SDK targets Node. Electron runs Node inside the app, so this works directly. With Tauri we would have to write and maintain a bridge.

The core is written without Electron APIs, so the hosted version (ADR-0002) can reuse it.

## Considered options

**Tauri 2** was rejected for the reasons above. Revisit only if installer size becomes a real complaint from users.
