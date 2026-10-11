# 把你的 AI 智能体连接到 IncarnaMind

*实验功能。* [English](agents.md)

如果你已经在用 AI 智能体，例如 Claude、Codex、Cursor 或 Kimi，可以让它搜索你在 IncarnaMind 中选择的文件夹并引用其中内容。它从 IncarnaMind 得到的每条引文都会与你的文档核对，你也可以在 IncarnaMind 中打开任意一条，查看它在文档中的位置。

## 开始之前

1. 打开 IncarnaMind，进入“设置 → 智能体”，开启“连接 AI 智能体”。
2. 在列表中找到你的智能体（这台电脑上已有的排在前面），点击“连接”。
3. 选择它可以读取的文件夹，然后点击“允许并连接”。其他内容它都无法读取。
4. 按下面对应智能体的一步操作完成连接。

使用智能体时请保持 IncarnaMind 打开：智能体需要与它通信。

**哪些内容会离开你的电脑。** 智能体读到的内容会发送给它所用的 AI 模型，适用该服务的条款，而非 IncarnaMind 的条款。某些套餐可能会用这些内容训练模型，请查看你的智能体的隐私设置。开启“所有内容都留在这台电脑上”（在“设置 → 搜索”中）时，所有智能体都会暂停。

**停止使用。** 在“设置 → 智能体”中点击“断开连接”，此后它将无法读取任何内容。每个智能体的“活动”会列出它做过的每一件事。

## 在智能体中提问

连接后，可以这样问你的智能体：

> 用 IncarnaMind 查一下，我的文档中关于终止条款是怎么说的？请注明出处。

回答中会带有 IncarnaMind 的引用，形如 `[cite:k3x9a2b7]`。想查看某一条，可以说：

> 在 IncarnaMind 中打开这条引用。

## 各智能体的设置步骤

各应用不同版本的菜单名称可能略有差异。

<a id="claude-desktop"></a>
### Claude（桌面应用）

点击“连接”。Claude 会打开并显示 IncarnaMind：点击“安装”，然后新建一个对话。

<a id="claude-code"></a>
### Claude Code

点击“连接”：IncarnaMind 会复制一行命令。把它粘贴到终端（Windows 中为 PowerShell）中运行。新开一个 Claude Code 会话，输入 `/mcp` 即可看到 IncarnaMind。

<a id="codex"></a>
### Codex，以及 ChatGPT 桌面应用

点击“连接”：IncarnaMind 会把自己添加到 Codex 的设置文件中，并先让你看到要添加的内容。新开一个 Codex 会话即可使用。ChatGPT 桌面应用与这台电脑上的 Codex 使用同一份设置。连接后可立即点击“撤销”恢复原样。

<a id="cursor"></a>
### Cursor

点击“连接”。Cursor 会打开并询问是否安装 IncarnaMind：点击“安装”。

<a id="vscode"></a>
### VS Code

点击“连接”。VS Code 会打开并询问是否安装 IncarnaMind：点击“安装”。在 Copilot Chat 的智能体模式中使用。

<a id="gemini-cli"></a>
### Gemini CLI

点击“连接”：IncarnaMind 会把自己添加到 Gemini CLI 的设置中。重新启动 Gemini CLI 即可。

<a id="kimi-code"></a>
### Kimi Code CLI

点击“连接”：IncarnaMind 会把自己添加到 Kimi Code 的设置中。重新启动 Kimi Code 即可。

<a id="qwen-code"></a>
### Qwen Code

点击“连接”：IncarnaMind 会把自己添加到 Qwen Code 的设置中。重新启动 Qwen Code 即可。

<a id="codebuddy"></a>
### CodeBuddy

点击“连接”：IncarnaMind 会把自己添加到 CodeBuddy 的设置中。重新启动 CodeBuddy 即可。

<a id="minimax-code"></a>
### MiniMax Code

点击“连接”：IncarnaMind 会把自己添加到 MiniMax Code 的设置中。重新启动 MiniMax Code 即可。

<a id="trae"></a>
### Trae

点击“连接”：IncarnaMind 会复制它的设置。在 Trae 中打开“设置 → MCP → 添加 → 手动添加”，粘贴后保存。

<a id="qoder"></a>
### Qoder

点击“连接”：IncarnaMind 会复制它的设置。在 Qoder 中打开 MCP 设置，添加服务器，粘贴后保存。

<a id="comate"></a>
### 百度 Comate

点击“连接”：IncarnaMind 会复制它的设置。在 Comate 的 MCP 设置中点击“+ 配置”打开设置文件，粘贴 IncarnaMind 这一项后保存。

<a id="cherry-studio"></a>
### Cherry Studio

点击“连接”：IncarnaMind 会复制它的设置。在 Cherry Studio 中打开“设置 → MCP → MCP 服务器 → 添加”，选择从 JSON 导入（或按复制的设置填入命令、参数和环境变量）并保存，再到你对话所用的助手中启用它。

<a id="chatbox"></a>
### Chatbox

点击“连接”：IncarnaMind 会复制它的设置。在 Chatbox 中打开“设置 → MCP → 添加服务器”，按复制的设置填入命令、参数和环境变量。

<a id="deepchat"></a>
### DeepChat

点击“连接”：IncarnaMind 会复制它的设置。在 DeepChat 的设置中添加 MCP 服务器并粘贴。

<a id="goose"></a>
### Goose

点击“连接”：IncarnaMind 会复制它的设置。在 Goose 中打开“Extensions → Add custom extension”，选择 STDIO，按复制的设置填入命令、参数和两个环境变量。

<a id="opencode"></a>
### OpenCode

点击“连接”：IncarnaMind 会复制它的设置。在 `opencode.json` 的 `"mcp"` 下添加一项 `"incarnamind"`：`"type"` 设为 `"local"`，`"command"` 设为命令加上它的参数，`"environment"` 设为复制的设置中的环境变量。

<a id="cline"></a>
### Cline

点击“连接”：IncarnaMind 会复制它的设置。在 Cline 中打开“MCP Servers → Configure”，把 IncarnaMind 这一项粘贴到 `mcpServers` 中并保存。

<a id="kilo"></a>
### Kilo Code

点击“连接”：IncarnaMind 会复制它的设置。按上面 OpenCode 的方式添加到 Kilo Code 的 MCP 设置中。

<a id="openclaw"></a>
### OpenClaw

点击“连接”：IncarnaMind 会复制它的设置。在 OpenClaw 中用复制的命令、参数和环境变量添加一个 MCP 服务器。

<a id="pi"></a>
### pi

点击“连接”：IncarnaMind 会复制它的设置。在 pi 中用复制的命令、参数和环境变量添加一个 MCP 服务器。

<a id="other"></a>
### 其他智能体

点击“连接”：IncarnaMind 会以大多数智能体接受的 `mcpServers` 格式复制它的设置。在你的智能体添加 MCP 服务器的地方粘贴，然后重新启动它。任何能启动本地 MCP 服务器的智能体都可以使用。

## 遇到问题时

你的智能体会把 IncarnaMind 的提示原样告诉你。

- **“IncarnaMind 没有打开”**：打开 IncarnaMind，确认“连接 AI 智能体”已开启，然后重新启动智能体，或让它重新连接 IncarnaMind。
- **“这个智能体还不能读取任何文档”**：在“设置 → 智能体 → 文件夹…”中为它选择文件夹。
- **“已连接的智能体暂停使用”**：在“设置 → 搜索”中关闭“所有内容都留在这台电脑上”。
- **“这个智能体已不再连接 IncarnaMind”**：它已断开连接，重新点击“连接”即可。
- **IncarnaMind 的回答需要较长时间**：向 IncarnaMind 提问时，它会用你自己的对话模型生成完整的回答。有些智能体只等待一分钟；由 IncarnaMind 设置的 Codex 会等待五分钟。

## 技术细节

IncarnaMind 是一个通过 stdio 运行的 MCP 服务器。命令是以 Node 方式运行的 IncarnaMind 应用本身，加上一个小脚本，它通过本机套接字把所有请求转给正在运行的应用；环境变量中是套接字的位置和这个智能体的密钥。它同时支持当前的 MCP 版本（2026-07-28）和上一版本。每个已连接智能体的“技术细节”可以再次复制它的设置。

工具包括：`search`、`read_passages`、`cite`、`check_citations`、`show_citation`、`list_minds`、`read_mind` 和 `ask`，都是只读的：智能体无法更改你的文档或 Mind。
