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

  "viewer.label": "Document viewer",
  "viewer.close": "Close the Document viewer",
  "viewer.resize": "Resize the Document viewer",

  "documents.title": "Documents",
  "documents.add": "Add Documents",
  "documents.none": "No Documents yet. Drop PDF, TXT or Markdown files here.",
  "documents.drop": "Drop PDF, TXT or Markdown files to add them",
  "documents.status.queued": "Queued",
  "documents.status.extracting": "Extracting text…",
  "documents.status.ready": "Ready",
  "documents.status.failed": "Failed: {reason}",
  "documents.status.noText": "No text found",
  "documents.failure.unreadable": "the file couldn't be read",
  "documents.failure.passwordProtected": "the PDF needs a password",
  "documents.failure.fileMissing": "its copy in the data folder is missing",
  "documents.failure.processingError": "something went wrong while processing it",
  "documents.rename": "Rename {name}",
  "documents.renameLabel": "New name for {name}",
  "documents.delete": "Delete {name}",
  "documents.delete.title": "Delete this Document?",
  "documents.delete.body":
    "“{name}” will be removed from IncarnaMind, along with its search index. Your original file isn't touched.",
  "documents.delete.cancel": "Cancel",
  "documents.delete.confirm": "Delete",
  "documents.skipped": "Only PDF, TXT and Markdown files can be added. Skipped: {names}",

  "folders.label": "Folders",
  "folders.all": "All Documents",
  "folders.new": "New Folder",
  "folders.newInside": "New Folder in {name}",
  "folders.nameLabel": "Name of the new Folder",
  "folders.expand": "Expand {name}",
  "folders.collapse": "Collapse {name}",
  "folders.rename": "Rename {name}",
  "folders.renameLabel": "New name for {name}",
  "folders.delete": "Delete {name}",
  "folders.delete.title": "Delete this Folder?",
  "folders.delete.body":
    "“{name}” and any Folders inside it will be deleted. The Documents in them are kept: they become unfiled.",
  "folders.delete.cancel": "Cancel",
  "folders.delete.confirm": "Delete Folder",
  "folders.empty": "No Documents in this Folder yet. Drag Documents here, or use “Move to…”.",
  "folders.moveTo": "Move {name} to…",
  "folders.moveTo.title": "Move to",
  "folders.unfiled": "Unfiled",

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
