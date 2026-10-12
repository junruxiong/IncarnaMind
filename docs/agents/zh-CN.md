# Writing Chinese strings

Read this before adding or changing a string in `src/shared/i18n/zh-CN.ts`, whether you are a person or a coding agent. It sits beside the words in [`CONTEXT.md`](../../CONTEXT.md) and the interaction rules in [`interaction.md`](interaction.md).

**The bar:** the Chinese interface reads like mainstream Chinese software, in the words Chinese Users already know from Feishu, Yuque, Tencent Docs, WPS, Kimi, Doubao, Yuanbao, ChatGPT, Notion, Obsidian, Zotero, macOS and Windows. Write it as a Chinese product would say it, not as a translation of the English. When a term isn't in the glossary below, use the word those apps show for the same thing, and add it here.

## The rules

1. **你, never 您.** Ant Design, Arco, Semi and Microsoft's Chinese style guide all say so; macOS uses 你 throughout.
2. **A space between Chinese and Latin letters or numbers.** "新建 Mind", "135 MB", "第 3 页", "{count} 个文档". No space next to full-width punctuation ("“Lease”的文件"), and none before % or °.
3. **Full-width punctuation in Chinese text:** ，。：；？！（）. Separate list items with 、 ("PDF、Word 和 Excel"). Parentheses stay full-width around Latin text: "Word（.docx）".
4. **Quotation marks are “ ”**, with ‘ ’ inside. Quote the names of UI items, files, Documents, Folders, Tags and anything the User typed: "移动到“财务”", "已保存“Tides 2.docx”". Keep 《》 for the titles of published works. Never use 「」.
5. **One "…" (U+2026)** after a command that asks for more before it acts ("导出…", "移动到…") and on progress ("正在整理…"). Use "……" only in running prose.
6. **Buttons and menu items are a verb, or a verb and its object:** 保存, 添加文档, 移动到…. No 请, no 点击, no 。. Cancel is 取消, Done is 完成, Retry is 重试.
7. **Errors say what happened, then what to do.** Prefer 无法… or 未能… to a bare 失败 or 没有成功: "无法调用 Notion：search", "未能登录。", "无法读取此文件". Don't blame the User.
8. **Empty states say "还没有…" and the next step:** "还没有文档。添加一个同步文件夹，或单独添加文件。" Not "暂无数据".
9. **Counts use 个** for 文档, 文件, 文件夹, 标签, 引用, 段落 and 工具, and **项** for a mixed selection ("还有 3 项"). 篇 belongs only in prose ("一篇文档"). Mind takes no plural: "Mind（3）", "收起 Mind".
10. **Status:** progress is 正在… ("正在整理 3 个文档…"), done is 已… ("已保存"), not yet is 尚未… or 未….
11. **Use the verbs Chinese apps use:** 生成 an Answer (not 写), 加载 (not 载入), 调用 a Tool (not 询问), 搜索 in a search field (not 查找), 新建 (not 创建), 重命名 (not 改名), 撤销 (not 撤消).
12. **Avoid translationese.**
    - No 被 where an active sentence works ("已删除", not "已被删除").
    - No 进行 + verb ("根据描述来判断", not "依据描述进行判断").
    - No long 的 chain before a noun: split the sentence ("添加一个问题，只搜索此文件夹", not "添加一个只搜索此文件夹的问题").
    - Drop 它/它的 and 你的 when they add nothing.
    - Put the condition first ("开启后，…").
13. **One word for one thing.** Use the glossary everywhere and never a second word for the same concept: 不在文件夹中, not 未分类 or 未分组; 对话模型, not 聊天模型; 提供商, not 服务商.
14. **Platform words go in their own keys.** Where macOS and Windows name a thing differently, add a key per platform and choose it in code, as `linkedFolders.reveal.mac` / `.windows` / `.other` and `menu.copy` / `menu.copy.other` do. Don't put one platform's word on the other.
15. **Names stay as they are:** IncarnaMind, Mind, ChatGPT, Ollama, OpenAI, PDF, Word, Markdown. Write file types in capitals (ZIP, CSV), and keys as they are printed (Enter, Esc, Backspace, ⌘J).

Before you open the pull request, read every new Chinese string aloud. If it sounds translated, rewrite it. Check it on screen at the narrowest window, too (interaction rule 11).

## Glossary

The words from `CONTEXT.md`, then the ones the interaction rules bring in. Where Chinese apps disagree, the choice follows the one most of them show.

### IncarnaMind's own words

| English | Chinese | Notes |
|---|---|---|
| Mind | Mind | A product name, like Notion: no translation, no plural, no measure word in labels. "新建 Mind", "Mind（3）", "这个 Mind" in prose. |
| Assistant | 助手 | The one per window. "显示助手", "收起助手". As Kimi, Doubao and Yuanbao call themselves (智能助手, AI 助手). |
| Conversation | 对话 | New chat is "新对话", History "历史对话", counted with 个: "3 个对话". As Doubao, Yuanbao, DeepSeek, Qwen, Feishu, Cherry Studio and Open WebUI show it. |
| Note | 笔记 | |
| Block | 块 | "块选项". As Feishu does. |
| Question | 问题 | Ask: 提问. "就此文档提问". |
| Answer | 回答 | Generate: 生成; Regenerate: 重新生成; Stop: 停止. |
| Question context | 上下文 | |
| Search scope | 搜索范围 | "仅搜索", "缩小范围". |
| Document | 文档 | Counted with 个. |
| Missing Document | (status) 文件缺失 | Long form: "找不到文件：它已从所在文件夹中移除。" |
| Unavailable Document | (status) 无法访问 | |
| Linked folder | 同步文件夹 | "添加同步文件夹…", "移除同步文件夹…". As Tencent Docs does. Say IncarnaMind only reads it wherever a Linked folder is added. |
| Other Documents | 其他文档 | |
| Web page | 网页 | A Document saved from a web address (网址). Its snapshot is 快照, as search engines' 网页快照: "打开快照". Fetching it is 抓取: "重新抓取", "无法抓取：网页需要登录". The live page: 原网页 ("打开原网页"). |
| Passage | 段落 | |
| Citation | 引用 | The numbered reference in an Answer, as Kimi, Doubao, Qwen, Cherry Studio and Open WebUI show it. |
| quote (of a Citation) | 引文 | "已在第 3 页找到引文", "未找到引文". |
| Quote (block) | 引述 | The "/" menu's block for quoted text; not a Citation (引用) or its quote (引文). |
| Location | 位置 | "第 3 页", "第 2 张幻灯片", "第 4–6 行", "§ 标题", a web page's "§ 历史，第 3 段". |
| Footnote | 脚注 | As Word, WPS and Feishu. "插入脚注", "脚注 3". |
| Reference record | 引用信息 | The "Cite as" panel's title too. Its fields: 类型、标题、作者、日期、期刊（出处）、卷、期、页码、出版者、DOI、网址、语言. |
| Needs details (a reference record) | 信息待补充 | |
| Look up (a reference record) | 查询 | "按 DOI 从 Crossref 补全空白字段". |
| Find sources | 查找资料 | "查找资料…". 资料 matches 资料库 (Library): what is found is added to it. |
| Source provider | 资料来源 | "资料来源：OpenAlex arXiv PubMed". Not 提供商, which is an AI model's provider. |
| Open access | 开放获取 | As CNKI and publishers say it. "仅开放获取", "无开放全文". |
| Preprint / Accepted manuscript | 预印本 / 录用稿 | The version an open copy is. |
| Cited by {count} | 被引 {count} 次 | As CNKI and Google Scholar show it. |
| Import references | 导入引用信息… | From a CSL JSON, BibTeX or RIS file. Counted with 条: "导入了 3 条". |
| Reference without a Document | 还没有文档 | Its list: "添加 PDF…" links it. |
| Citation style | 引用格式 | "引用格式…", "引用格式已改为 APA 第 7 版". A style keeps its own name; its variant goes in full-width brackets, in the standard's words: "GB/T 7714—2015（顺序编码制）", "（著者-出版年制）". |
| Bibliography | 参考文献 | Its heading in the Mind and in Chinese styles' exports. |
| Folder | 文件夹 | |
| Source location | 文件位置 | The sidebar view of the files as they are on disk. |
| Tag | 标签 | Tagging is 打标签; automatic tagging is 自动打标签. |
| Library | 资料库 | Where the User's files live in ChatGPT (资料库), Yuanbao (资料库, 加入资料库) and LobeChat. Not 知识库, which in Feishu and Yuque is a wiki and in AI clients a collection built for one assistant. |
| Organize | 整理 | "整理此文件夹". As Tencent Docs (批量整理文档) and Apple Notes (整理备忘录). |
| Not in a Folder | 不在文件夹中 | One place for Documents and Minds in no Folder. |
| Tab | 标签页 | As Chrome, Edge and VS Code. A preview tab is kept with 保持打开. |
| Group (of tabs) | 栏 | "第 1 栏", "向右拆分", "各栏等宽", "移动到右侧一栏". |
| Tool | 工具 | Calling one: 调用. |
| Connector | 连接器 | As Doubao, Feishu, LobeChat and Microsoft. |
| Skill | 技能 | As Kimi, Doubao, Yuanbao, Feishu, Notion and Cherry Studio. |
| Skill script | 技能脚本 | |
| Built-in Skill | 内置技能 | |
| Outside agent | 智能体 | The User's own AI agent: "连接智能体", "设置 → 智能体". As Kimi, Doubao, Yuanbao and Coze call an agent. |
| Proposal | 修订 | Accept: 接受; Reject: 拒绝; Accept all: 全部接受; Reject all: 全部拒绝. As WPS and Feishu (接受修订, 拒绝修订). |
| Checkpoint | 检查点 | |
| Citation check | 核对 | "正在核对引文…", "无法核对". |
| Quote not found | 未找到引文 | |
| Unsupported | 无依据 | |
| Contradicted | 与文档矛盾 | |
| Needs re-check | 需重新核对 | |
| Mark | 标记 | |
| Run | 运行 | |
| Task | 任务 | |
| Routine | 定时任务 | As Kimi, Doubao, Yuanbao, Cherry Studio and LobeChat. |
| Comparison table | 对比表 | |
| Chart | 图表 | Its kinds: 柱状图、折线图、饼图、散点图. "以表格显示" / "以图表显示"; "用表格生成图表". |
| Columns (a chart's series) | 数据列 | The columns of a table a chart draws. A scatter chart's two: 横轴 and 纵轴. |
| Source (of a chart's values) | 数据来源 | Under a chart in an export, as Chinese reports caption theirs: "数据来源：". |
| Memory | 记忆 | |
| Meeting | 会议 | |
| Transcript | 转写稿 | |
| Speaker | 发言人 | |
| Granted folder | 授权文件夹 | |
| Output location | 保存位置 | |
| Usage data | 使用数据 | |
| Scanned page, scan | 扫描页、扫描件 | A PDF page that is a picture: counted with 页 ("12 页扫描件"). |
| Text recognition (OCR) | 文字识别 | Reading a scan: 识别 ("正在识别 12 页扫描件…", "第 3 页扫描件无法识别"). The model is 文字识别模型. |
| Text read from a scan | 从扫描件识别的文字 | On a Citation's card. A word it may have misread: "可能识别有误". |

### Figures

A figure in an Answer is written as reports write it, with no space between the number and its scale or unit: "3,045.74万元", "2.3个百分点", "约1.2亿元".

| English | Chinese | Notes |
|---|---|---|
| percentage points | 个百分点 | A difference of two percents: "2.3个百分点", never "2.3%". |
| about (an approximate figure) | 约 | "约1.2亿元". |
| times (a multiple) | 倍 | "4.00倍". |
| yes / no (what a comparison gives) | 是 / 否 | |

### Models and settings

| English | Chinese | Notes |
|---|---|---|
| chat model | 对话模型 | |
| embedding model | 嵌入模型 | |
| reranking | 重新排序; the model: 重排序模型 | |
| provider | 提供商 | As Open WebUI, Chatbox, AnythingLLM, Dify and Microsoft. |
| API key | API 密钥 | |
| Server URL (base URL) | API 地址 | As Cherry Studio and Open WebUI. |
| local / cloud | 本地 / 云端 | "本地模型", "云端模型", "在本机运行". |
| sign in / sign out | 登录 / 退出登录 | |
| ChatGPT plan | ChatGPT 订阅 | |
| Settings | 设置 | |

### File and editing words

| English | Chinese | Notes |
|---|---|---|
| Recently deleted | 回收站 | As Feishu, Yuque, Tencent Docs, WPS, DingTalk, Obsidian and Zotero. Restore: 恢复. Delete permanently: 彻底删除. |
| Undo | 撤销 | The undo line: "已将 3 个文档移动到“财务” · 撤销". |
| Rename | 重命名 | |
| Move to… | 移动到… | |
| Pin / Unpin / Pinned | 置顶 / 取消置顶 / 已置顶 | |
| Recent | 最近 | |
| Peek (Space) | 预览 | macOS calls its Quick Look 快速查看; Chinese apps say 预览. |
| Get info | 文档信息 | As Feishu (文档信息) and Tencent Docs (查看文档信息). The same on every platform. |
| To review | 待确认 | |
| Suggested Folder | 推荐的文件夹 | The chip says 推荐. |
| Show in Finder | macOS: 在访达中显示; Windows: 在文件资源管理器中显示; Linux: 在文件管理器中显示 | Per-platform keys. |
| Copy path | 复制路径 | As VS Code and Obsidian, on every platform. |
| Open in default app | 用默认应用打开 | |
| Copy (Edit menu) | macOS: 拷贝; Windows and Linux: 复制 | Per-platform keys. |
| View (menu) | macOS: 显示; Windows and Linux: 查看 | Per-platform keys. |
| Sidebar | 侧边栏 | One word on every platform, so the button and the menu agree. |
| Import / Export | 导入 / 导出 | |
| Filter | 筛选; Clear filters: 清除筛选 | |
