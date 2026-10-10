<!-- README.md 与 README.zh-CN.md 逐节对应，修改时请两份一起改。 -->

<p align="center"><a href="README.md">English</a> · <b>简体中文</b></p>

<h1 align="center">IncarnaMind</h1>

<p align="center">
  <b>会自己核对引文的 AI 笔记本。</b><br>
  写作和提问在同一个地方。回答只引用你自己的文件，标明出处，并逐条核对引文确实在原文的那个位置。免费开源，在你自己的电脑上运行。
</p>

<p align="center">
  <a href="https://github.com/junruxiong/IncarnaMind/releases"><b>发布页</b></a> ·
  <a href="#怎么用">怎么用</a> ·
  <a href="#你的文件始终归你">隐私</a> ·
  <a href="#常见问题">常见问题</a> ·
  <a href="https://github.com/junruxiong/IncarnaMind/discussions">讨论区</a>
</p>

<p align="center">
  <img alt="桌面应用，支持 macOS、Windows 和 Linux" src="https://img.shields.io/badge/macOS%20%C2%B7%20Windows%20%C2%B7%20Linux-desktop%20app-17191C">
  <a href="LICENSE"><img alt="许可证：Apache-2.0" src="https://img.shields.io/github/license/junruxiong/IncarnaMind?color=17191C"></a>
  <!-- 第一个版本发布后，加上：<a href="https://github.com/junruxiong/IncarnaMind/releases/latest"><img alt="最新版本" src="https://img.shields.io/github/v/release/junruxiong/IncarnaMind?color=17191C"></a> -->
</p>

<p align="center">
  <img src="docs/images/hero-zh.png" width="880" alt="中文界面的笔记本，里面有两个问题和它们的回答，旁边是一份 PDF。第一个回答带编号引用和绿色勾号，引用的引文在 PDF 页面上高亮显示。第二个回答的引用标成橙色，展开的卡片写着“未在第 1 页找到引文”。">
</p>

<p align="center"><sub>第二条引文是特意改动过的，用来演示引文不在所引页面上时会怎样。</sub></p>

> [!NOTE]
> **IncarnaMind 还没有正式发布。** 桌面版正在开发。想在第一个版本发布时收到通知，可以点本页右上角的 **Watch → Custom → Releases**。在找 2023 年的命令行版本？它在 [`v1-python-cli`](https://github.com/junruxiong/IncarnaMind/tree/v1-python-cli) 标签下。

## 为什么选 IncarnaMind

适合所有要从资料里写东西的人：论文、报告、合同、笔记。

- **回答可以核对。** 每个回答都标出它用到的出处，并附上原文引文。回答写完后，IncarnaMind 会到那个位置去找这段引文：找到了打绿勾，没找到就标成橙色。点一下引用，文件会直接打开到那里，引文高亮显示。
- **是笔记本，不是聊天框。** 你在 Mind 里写作，一个 Mind 就是为一个主题或项目准备的笔记本。在任何位置都能提问，回答直接写进你的草稿，可以像普通文字一样修改。
- **文件留在原处。** 关联你平时用的文件夹，比如放论文或合同的文件夹，或者 Zotero 的存储文件夹。IncarnaMind 就地读取，从不移动、重命名或修改文件；文件有增删改，它也会跟着更新。
- **常用格式都支持。** PDF、Word、PowerPoint、Excel、CSV、Markdown 和纯文本。
- **模型由你来选。** 可以填自己的 OpenAI、Anthropic、Google 密钥，或接入任何兼容 OpenAI 接口的服务；也可以通过 Ollama 使用免费的本地模型，内容不出你的电脑。
- **写完就能交。** 导出为 Word，引用自动变成脚注；也可以导出为 Markdown。没通过核对的引用会标上“[未核实]”。

## 怎么用

1. **添加文件。** 关联一个文件夹，或者单独添加文件。文件读完就能搜索，不会上传到任何地方。
2. **边写边问。** 新建一个 Mind，照常写作。需要从文件里找东西时，就在当前位置提一个问题。输入 `@` 可以只搜索某些文件夹、标签或文档。
3. **先核对，再使用。** 点一下引用，就能看到引文和核对结果，还能打开原文那一页。有用的留下，其余的直接改。

想先看看效果？第一次运行时，IncarnaMind 会提供一个示例 Mind《茶从哪里来》，不用设置任何模型就能体验。

### 引用会指到哪里

| 文件 | 引用指向 |
| --- | --- |
| PDF | 具体哪一页 |
| PowerPoint | 具体哪张幻灯片（包括演讲者备注） |
| Excel、CSV | 哪个工作表的哪几行 |
| Word、Markdown | 引文所在的章节 |
| 纯文本 | 具体哪几行 |

“找到引文”表示这段文字确实在那个位置，但不能证明你写的那句话就是对的，所以原文永远只差一次点击。“无法核对”表示那里没有可以比对的文字，比如扫描页。

## 你的文件始终归你

不用注册账号，也没有 IncarnaMind 服务器。

| 内容 | 去向 |
| --- | --- |
| 你的 Mind 和搜索索引 | 你电脑上的一个数据文件夹 |
| 你的文件 | 哪儿也不去，原样留在原处 |
| 你的问题，以及为它找到的段落 | 发给你选的模型；用本地模型时哪儿也不去 |
| API 密钥 | 用系统自带的安全存储加密保存 |
| 使用数据 | 不收集 |
| 崩溃报告 | 默认关闭；开启后也会先去掉你的内容 |

IncarnaMind 第一次要把内容发给外部服务之前，会先告诉你要发什么、发给谁，等你同意后才发送。**设置 → 隐私**里列出了每一项，随时可以撤回。服务商提供选项的，IncarnaMind 都会要求它不要保存你的请求；发给 OpenAI 的每个请求都带着这个要求。

**运行什么，由你说了算。** 连接器里可能改动内容的工具，以及技能里的脚本，在运行之前 IncarnaMind 都会先问你：允许这一次、始终允许，或者拒绝。**设置 → 工具**里列出了你设过的每条规则，随时可以撤销。

## 更多功能

- **轻松整理。** 用文件夹和标签管好大量文档。标签可以自动添加，你自己设的标签永远不会被改动。
- **连接你的应用。** 添加连接器（MCP 服务器），或者直接导入你在 Claude Desktop、Cursor 里配置好的。任何可能改动内容的操作，都会先问你。
- **按你的方式做事。** 添加技能（标准的 `SKILL.md` 文件夹），让回答照你的步骤来。内置三个技能：总结一篇文档、跨文档写文献综述、把 Mind 整理成报告。技能里的脚本要你批准后才会运行。
- **中英双语。** 整个界面都有中文和英文。

## 即将推出

下面是计划中的功能，现在还不能用。计划见[路线图 issue](https://github.com/junruxiong/IncarnaMind/issues/74)，没有具体日期。

- **会议。** 在你的电脑上实时转写，也可以用你自己的密钥在云端转写；区分不同发言人，并给出引用转写内容的摘要。
- **Word 双向支持。** 打开 Word 文件，再导出回去，你的修改会保留为修订。
- **核对草稿。** 对照你的文件，核对你自己写的句子，以及别人给你的文档。
- **任务。** 研究类任务在后台运行，你可以接着做别的事。

## 下载

IncarnaMind 还没有正式发布，暂时没有可下载的安装包。想现在试用，可以[从源码运行](#参与贡献)。第一个版本发布后，[发布页](https://github.com/junruxiong/IncarnaMind/releases)会提供各系统的安装包：

| 你的电脑 | 下载 |
| --- | --- |
| Apple 芯片的 Mac（M1 及以后） | `IncarnaMind-<版本>-mac-arm64.dmg` |
| Intel 芯片的 Mac | `IncarnaMind-<版本>-mac-x64.dmg` |
| Windows（64 位） | `IncarnaMind-<版本>-win-x64.exe` |
| Linux（64 位） | `IncarnaMind-<版本>-linux-x86_64.AppImage` |

IncarnaMind 每次启动都会检查新版本。在 Windows、Linux 的 AppImage 和已签名的 Mac 版上，它会在后台下载更新，退出时安装；选择**立即重启**则马上安装。

<details>
<summary><b>Windows 提示“Windows 已保护你的电脑”</b></summary>

安装程序暂时还没有代码签名，所以第一次运行时会被 Microsoft Defender SmartScreen 拦下。点**更多信息**，再点**仍要运行**。如果浏览器拦截了下载，选择**保留**。

</details>

<details>
<summary><b>macOS 提示无法验证 IncarnaMind</b></summary>

Mac 版完成签名和公证之前，macOS 会拒绝打开它，提示“Apple 无法验证“IncarnaMind”是否包含恶意软件”。想照样打开：

1. 先打开一次 IncarnaMind，在警告里点**完成**。
2. 打开**系统设置 → 隐私与安全性**，往下滚到**安全性**，点 IncarnaMind 那条提示旁边的**仍要打开**。
3. 输入密码或用触控 ID 确认，再点一次**仍要打开**。

macOS 会记住你的选择。也可以在“终端”里运行：`xattr -dr com.apple.quarantine /Applications/IncarnaMind.app`。

未签名的 Mac 版不能自动更新：有新版本时，IncarnaMind 会通知你，并打开下载页面。照同样的方法安装，替换旧版本即可。

</details>

<details>
<summary><b>Linux 上的 AppImage 打不开</b></summary>

先给它执行权限：`chmod +x IncarnaMind-*.AppImage`。Ubuntu 22.04 及以后的版本还需要安装 FUSE 2：`sudo apt install libfuse2`（24.04 及以后是 `libfuse2t64`）。

</details>

## 常见问题

**要花钱吗？**
不用。IncarnaMind 开源免费（Apache 2.0 许可证）。如果用云端模型，按用量付费给模型服务商；用本地模型则完全免费。

**一定要有 API 密钥吗？**
不一定。只要 [Ollama](https://ollama.com) 在运行，IncarnaMind 一键就能下载本地模型。也可以填 OpenAI、Anthropic、Google 的密钥，或接入任何兼容 OpenAI 接口的服务。

**能离线用吗？**
搜索文件可以离线：搜索靠文件里的字词，按含义搜索（嵌入）是可选功能，默认关闭。回答需要模型，所以用本地模型时，全部功能都能离线。

**会改动我的文件吗？**
不会。它只在原处读取文件，只往自己的数据文件夹和你导出的文件里写内容。

**“找到引文”到底是什么意思？**
意思是这段引文确实出现在引用所指的位置，但不代表它能证明那句话。“无法核对”表示那里没有可以比对的文字，比如扫描页。

**和直接问聊天机器人有什么不同？**
你在同一份文档里写作和提问，而不是在聊天窗口里；每段引文都会和它所指的原文比对；文件留在你的电脑上，愿意的话还能全程用本地模型。

**怎么备份？**
复制数据文件夹就行，里面有你的 Mind、搜索索引和设置。文件本来就是你自己的，照原来的方式备份即可。

## 参与贡献

IncarnaMind 公开开发，欢迎各种帮助：写代码、用你自己的文档测试、翻译、提想法。

- 发现问题或想要新功能：[提交 issue](https://github.com/junruxiong/IncarnaMind/issues/new/choose)
- 提问和交流想法：[讨论区](https://github.com/junruxiong/IncarnaMind/discussions)（中文、英文都可以）
- 第一次参与代码？可以从标着 [good first issue](https://github.com/junruxiong/IncarnaMind/labels/good%20first%20issue) 的 issue 开始，并先读一下 [CONTRIBUTING.md](CONTRIBUTING.md)（英文）。

从源码运行需要 Node.js 24 或更高版本：

```shell
git clone https://github.com/junruxiong/IncarnaMind.git
cd IncarnaMind
npm install
npm run dev
```

测试、评测和代码结构见 [docs/development.md](docs/development.md)（英文），打包和发布见 [docs/releasing.md](docs/releasing.md)（英文）。代码里用到的术语见 [CONTEXT.md](CONTEXT.md)，设计决策记录在 [docs/adr](docs/adr)。

## 项目由来

IncarnaMind 最早是 2023 年的一个 Python 命令行工具，用来和 PDF 对话，当时收获了大约 800 个 star。2026 年，IncarnaMind 从头重写成了桌面应用，核心只有一个想法：从文件里得到的回答，能核对才真正有用。旧版命令行工具保留在 [`v1-python-cli`](https://github.com/junruxiong/IncarnaMind/tree/v1-python-cli) 标签下。谢谢一直关注它的你。

如果 IncarnaMind 对你有用，点个 star，能让更多人看到它。

<details>
<summary>引用 IncarnaMind</summary>

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

</details>

## 许可证

[Apache License 2.0](LICENSE)。应用内置的字体（Source Serif 4、Source Sans 3 和 JetBrains Mono）采用 [SIL Open Font License 1.1](https://openfontlicense.org)。
