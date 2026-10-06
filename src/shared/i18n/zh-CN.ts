import type { Dictionary } from "./types";

/** Simplified Chinese. `satisfies Dictionary` fails the type-check on a missing or unknown key. */
export const zhCN = {
  "app.name": "IncarnaMind",

  "sidebar.label": "侧边栏",
  "sidebar.newMind": "新建 Mind",
  "sidebar.minds": "Minds",
  "sidebar.noMinds": "还没有 Mind",
  "sidebar.settings": "设置",
  "sidebar.github": "GitHub",
  "sidebar.resize": "调整侧边栏宽度",

  "mind.untitled": "未命名",
  "mind.noneOpen.title": "开始一个 Mind",
  "mind.noneOpen.body": "Mind 是为一个主题或项目准备的笔记本。",

  "viewer.label": "文档查看器",
  "viewer.close": "关闭文档查看器",
  "viewer.resize": "调整文档查看器宽度",

  "settings.title": "设置",
  "settings.language": "界面语言",
  "settings.language.system": "跟随系统",
  "settings.language.en": "English",
  "settings.language.zh-CN": "简体中文",
  "settings.done": "完成",

  "error.load": "IncarnaMind 无法载入你的数据：{message}",
  "error.action": "操作没有成功：{message}",
  "error.dismiss": "关闭",
  "error.startup.title": "IncarnaMind 无法启动",
  "error.startup.body": "无法打开你的数据文件夹。\n\n{message}",
} satisfies Dictionary;
