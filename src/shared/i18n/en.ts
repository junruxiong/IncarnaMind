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
