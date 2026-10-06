/**
 * English: the reference dictionary. Its keys define `MessageKey`; every other
 * language must provide exactly these keys, or the type-check fails.
 * Placeholders look like `{name}`.
 */
export const en = {
  "app.name": "IncarnaMind",

  "sidebar.label": "Sidebar",
  "sidebar.newMind": "New Mind",
  "sidebar.minds": "Minds",
  "sidebar.noMinds": "No Minds yet",
  "sidebar.settings": "Settings",
  "sidebar.github": "GitHub",
  "sidebar.resize": "Resize the sidebar",

  "mind.untitled": "Untitled",
  "mind.noneOpen.title": "Start a Mind",
  "mind.noneOpen.body": "A Mind is a notebook for a topic or project.",
  "mind.title.label": "Mind title",
  "mind.editor.label": "Mind content",
  "mind.editor.placeholder": "Start writing…",
  "mind.delete": "Delete Mind",
  "mind.delete.body": "“{title}” will be removed from your Minds.",
  "mind.delete.confirm": "Delete",
  "mind.delete.cancel": "Cancel",

  "editor.placeholder": "Type / for headings, code and math",
  "editor.slash.label": "Insert",
  "editor.slash.empty": "No matches",
  "editor.slash.text": "Text",
  "editor.slash.heading1": "Heading 1",
  "editor.slash.heading2": "Heading 2",
  "editor.slash.heading3": "Heading 3",
  "editor.slash.codeBlock": "Code block",
  "editor.slash.math": "Math block",
  "editor.slash.inlineMath": "Inline math",
  "editor.block.handle": "Drag to move, click for options",
  "editor.block.menu": "Block options",
  "editor.block.delete": "Delete",
  "editor.math.label": "LaTeX formula",
  "editor.math.placeholder": "e.g. E = mc^2",
  "editor.math.hint": "Enter to save · Shift+Enter for a new line · Esc to cancel",
  "editor.math.empty": "New formula",
  "editor.code.language": "Code language",
  "editor.code.auto": "Auto",
  "editor.format.label": "Formatting",
  "editor.format.bold": "Bold",
  "editor.format.italic": "Italic",
  "editor.format.strike": "Strike",

  "viewer.label": "Document viewer",
  "viewer.close": "Close the Document viewer",
  "viewer.resize": "Resize the Document viewer",

  "settings.title": "Settings",
  "settings.language": "Interface language",
  "settings.language.system": "Same as system",
  "settings.language.en": "English",
  "settings.language.zh-CN": "简体中文",
  "settings.done": "Done",

  "error.load": "IncarnaMind couldn't load your data: {message}",
  "error.action": "That didn't work: {message}",
  "error.dismiss": "Dismiss",
  "error.startup.title": "IncarnaMind couldn't start",
  "error.startup.body": "Your data folder couldn't be opened.\n\n{message}",

  "update.available.message": "IncarnaMind {version} is available",
  "update.available.detail":
    "This copy of IncarnaMind can't update itself. Download the new version and use it to replace this one.",
  "update.available.confirm": "Download",
  "update.ready.message": "IncarnaMind {version} is ready to install",
  "update.ready.detail":
    "It installs the next time you quit IncarnaMind. Restart now to install it straight away.",
  "update.ready.confirm": "Restart now",
  "update.later": "Later",
} as const satisfies Record<string, string>;
