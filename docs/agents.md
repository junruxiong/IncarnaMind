# Connect your AI agent to IncarnaMind

*Experimental.* [中文版](agents.zh-CN.md)

If you already use an AI agent, such as Claude, Codex, Cursor or Kimi, you can let it search the folders you choose in IncarnaMind and quote them. Every quote it gets from IncarnaMind is checked against your Document, and you can open any of them in IncarnaMind to see it in place.

## Before you start

1. Open IncarnaMind, then **Settings → Agents**, and turn on **Connect AI agents**.
2. Find your agent in the list (agents on this computer come first) and click **Connect**.
3. Choose the folders it may read, then click **Allow and connect**. It can't read anything else.
4. Follow the one step for your agent below.

Keep IncarnaMind open while you use your agent: the agent talks to it.

**What leaves your computer.** What the agent reads goes to the AI model the agent uses, under that service's terms, not IncarnaMind's. Some plans may use it to train their models; check your agent's privacy settings. While **Keep everything on this computer** is on (Settings → Search), every agent is paused.

**To stop.** Settings → Agents → **Disconnect**. The agent can't read anything from then on. Each agent's **Activity** lists everything it did.

## Ask in your agent

Once it's connected, ask your agent something like:

> Using IncarnaMind, what do my documents say about the termination clause? Cite your sources.

The answer carries IncarnaMind's citations, written like `[cite:k3x9a2b7]`. To see one, ask:

> Show me that citation in IncarnaMind.

## Your agent, step by step

Menu names can differ a little between versions of each app.

<a id="claude-desktop"></a>
### Claude (the desktop app)

Click **Connect**. Claude opens and shows IncarnaMind: click **Install**. Then start a new chat.

<a id="claude-code"></a>
### Claude Code

Click **Connect**: IncarnaMind copies one line. Paste it into Terminal (Windows: PowerShell) and press Return. Start a new Claude Code session; `/mcp` lists IncarnaMind.

<a id="codex"></a>
### Codex, and the ChatGPT desktop app

Click **Connect**: IncarnaMind adds itself to Codex's settings file, and shows you first what it adds. Start a new Codex session. The ChatGPT desktop app uses the same settings as Codex on your computer. Right after connecting, **Undo** takes the change back.

<a id="cursor"></a>
### Cursor

Click **Connect**. Cursor opens and asks to install IncarnaMind: click **Install**.

<a id="vscode"></a>
### VS Code

Click **Connect**. VS Code opens and asks to install IncarnaMind: click **Install**. Use it from Copilot Chat in agent mode.

<a id="gemini-cli"></a>
### Gemini CLI

Click **Connect**: IncarnaMind adds itself to Gemini CLI's settings. Start Gemini CLI again.

<a id="kimi-code"></a>
### Kimi Code CLI

Click **Connect**: IncarnaMind adds itself to Kimi Code's settings. Start Kimi Code again.

<a id="qwen-code"></a>
### Qwen Code

Click **Connect**: IncarnaMind adds itself to Qwen Code's settings. Start Qwen Code again.

<a id="codebuddy"></a>
### CodeBuddy

Click **Connect**: IncarnaMind adds itself to CodeBuddy's settings. Start CodeBuddy again.

<a id="minimax-code"></a>
### MiniMax Code

Click **Connect**: IncarnaMind adds itself to MiniMax Code's settings. Start MiniMax Code again.

<a id="trae"></a>
### Trae

Click **Connect**: IncarnaMind copies its settings. In Trae, open **Settings → MCP → Add → Add manually**, paste, and save.

<a id="qoder"></a>
### Qoder

Click **Connect**: IncarnaMind copies its settings. In Qoder, open its MCP settings, add a server, paste, and save.

<a id="comate"></a>
### Comate

Click **Connect**: IncarnaMind copies its settings. In Comate, open the MCP settings, click **+ Configure** to open its settings file, paste IncarnaMind's entry, and save.

<a id="cherry-studio"></a>
### Cherry Studio

Click **Connect**: IncarnaMind copies its settings. In Cherry Studio, open **Settings → MCP → MCP Servers → Add**, choose to import from JSON (or fill in the command, arguments and environment from the settings you copied), save, then turn it on for the assistant you chat with.

<a id="chatbox"></a>
### Chatbox

Click **Connect**: IncarnaMind copies its settings. In Chatbox, open **Settings → MCP → Add Server**, and fill in the command, the arguments and the environment from the settings you copied.

<a id="deepchat"></a>
### DeepChat

Click **Connect**: IncarnaMind copies its settings. In DeepChat, add an MCP server in its settings and paste them.

<a id="goose"></a>
### Goose

Click **Connect**: IncarnaMind copies its settings. In Goose, open **Extensions → Add custom extension**, choose STDIO, and fill in the command, the arguments and the two environment values from the settings you copied.

<a id="opencode"></a>
### OpenCode

Click **Connect**: IncarnaMind copies its settings. In `opencode.json`, add under `"mcp"` an entry `"incarnamind"` with `"type": "local"`, `"command"` set to the command followed by its arguments, and `"environment"` set to the environment from the settings you copied.

<a id="cline"></a>
### Cline

Click **Connect**: IncarnaMind copies its settings. In Cline, open **MCP Servers → Configure**, paste IncarnaMind's entry into `mcpServers`, and save.

<a id="kilo"></a>
### Kilo Code

Click **Connect**: IncarnaMind copies its settings. Add them to Kilo Code's MCP settings as for OpenCode above.

<a id="openclaw"></a>
### OpenClaw

Click **Connect**: IncarnaMind copies its settings. Add IncarnaMind as an MCP server in OpenClaw with the command, arguments and environment you copied.

<a id="pi"></a>
### pi

Click **Connect**: IncarnaMind copies its settings. Add IncarnaMind as an MCP server in pi with the command, arguments and environment you copied.

<a id="other"></a>
### Another agent

Click **Connect**: IncarnaMind copies its settings, in the `mcpServers` form most agents take. Paste them where your agent adds an MCP server, then restart it. Any agent that can start a local MCP server works.

## If something goes wrong

Your agent passes on what IncarnaMind says, in plain words.

- **"IncarnaMind isn't open"**: open IncarnaMind, check that **Connect AI agents** is on, then restart the agent or reconnect it to IncarnaMind.
- **"This agent can't read any Documents yet"**: choose its folders in Settings → Agents → **Folders…**.
- **"Connected agents are paused"**: turn off **Keep everything on this computer** in Settings → Search.
- **"This agent isn't connected to IncarnaMind any more"**: it was disconnected. Click **Connect** again.
- **An answer from IncarnaMind takes long**: asking IncarnaMind writes a whole Answer on your own chat model. Some agents stop waiting after a minute; Codex is given five minutes when IncarnaMind sets it up.

## For technical users

IncarnaMind is an MCP server over stdio. The command is IncarnaMind's own app, run as Node, with a small script that passes everything on to the running app over a local socket; the environment holds where that socket is and the agent's key. It serves the current MCP version (2026-07-28) and the one before it. Each connected agent's **Details for technical users** copies its settings again.

The Tools: `search`, `read_passages`, `cite`, `check_citations`, `show_citation`, `list_minds`, `read_mind` and `ask`. All are read-only: an agent can't change your Documents or Minds.
