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
} as const satisfies Record<string, string>;
