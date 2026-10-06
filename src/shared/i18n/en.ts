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

  "providers.setup.title": "Choose how Questions are answered",
  "providers.setup.body":
    "Answers come from an AI model you choose. Notes and Documents work without one, so you can also do this later in Settings.",
  "providers.setup.orApiKey": "Or use an API key",
  "providers.setup.later": "Set up later",

  "providers.kind.openai": "OpenAI",
  "providers.kind.anthropic": "Anthropic",
  "providers.kind.google": "Google",
  "providers.kind.openai-compatible": "OpenAI-compatible server",
  "providers.kind.ollama": "Ollama",
  "providers.kind.chatgpt": "Sign in with ChatGPT",
  "providers.chatgpt.unavailable": "Not available yet",

  "providers.form.label": "Chat provider",
  "providers.form.apiKey": "API key",
  "providers.form.apiKeyOptional": "API key (optional)",
  "providers.form.apiKeySaved": "A key is saved. Leave this empty to keep it.",
  "providers.form.baseUrl": "Server URL",
  "providers.form.baseUrlHint":
    "For example https://api.deepseek.com/v1 or http://localhost:1234/v1",
  "providers.form.model": "Model",
  "providers.form.test": "Test connection",
  "providers.form.testing": "Testing…",
  "providers.form.save": "Use this provider",
  "providers.form.saving": "Saving…",

  "providers.test.ok": "Connected: the provider answered.",
  "providers.test.auth": "The key was refused. Check that it is correct and still active.",
  "providers.test.model": "The provider doesn't know this model. Check its name.",
  "providers.test.rate-limit": "The provider is limiting requests for this key. Try again shortly.",
  "providers.test.network": "Couldn't reach the server. Check its URL and your connection.",
  "providers.test.provider": "The provider returned an error.",
  "providers.test.consent-declined":
    "Nothing was sent, because you chose not to send data to this service.",
  "providers.test.unknown": "The test didn't work.",

  "providers.secrets.plainText.title": "No keyring is running",
  "providers.secrets.plainText.body":
    "IncarnaMind encrypts API keys with your system's keyring, and none is running, so a key would be stored as plain text in your data folder. Start GNOME Keyring or KWallet (for example, install gnome-keyring, then sign out and back in), and restart IncarnaMind.",
  "providers.secrets.plainText.accept":
    "Store keys without a keyring anyway. Anyone who can read my files could read them.",
  "providers.secrets.unavailable":
    "This system can't encrypt API keys, so they can't be saved. You can still use Ollama, or a server that needs no key.",

  "providers.ollama.title": "Local models with Ollama",
  "providers.ollama.checking": "Looking for Ollama on this computer…",
  "providers.ollama.detected":
    "Ollama is running on this computer. With a local model, your Questions and Documents stay on it.",
  "providers.ollama.notDetected":
    "To run models on this computer, install Ollama from ollama.com and start it.",
  "providers.ollama.use": "Use local models ({model})",
  "providers.ollama.pulling": "Downloading {model}… This can take a while.",
  "providers.ollama.checkAgain": "Check again",

  "providers.readiness.no-provider":
    "Asking Questions needs a chat model. Notes and Documents work without one.",
  "providers.readiness.missing-api-key":
    "The API key for {service} isn't saved on this device. Add it in Settings to ask Questions.",
  "providers.readiness.consent-declined":
    "You chose not to send data to {service}, so Questions are off. Choose another provider or a local model, or allow it in Settings.",
  "providers.readiness.setUp": "Set up",

  "providers.settings.title": "Chat model",
  "providers.settings.none": "No chat model is set up yet.",
  "providers.settings.provider": "Provider: ",
  "providers.settings.local": "on this computer",
  "providers.settings.defaultModel": "Default model",
  "providers.settings.saveModel": "Save",
  "providers.settings.change": "Change provider",
  "providers.settings.cancel": "Cancel",
  "providers.settings.remove": "Remove",

  "consent.dialog.title": "Send data to {service}?",
  "consent.dialog.body": "To {purpose}, IncarnaMind sends this to {service}:",
  "consent.dialog.bodyMore": "To {purpose}, IncarnaMind will now also send this to {service}:",
  "consent.dialog.note":
    "Nothing is sent until you allow it. You can change your mind in Settings.",
  "consent.dialog.allow": "Allow",
  "consent.dialog.decline": "Don't allow",

  "consent.flow.chat": "Chat",
  "consent.flow.chat.purpose": "answer your Questions",
  "consent.data.blocks":
    "Your draft: the Blocks above the Question, with your Notes and earlier Answers",
  "consent.data.passages": "Passages from your Documents that match the Question",
  "consent.data.tool-results": "Results of the Tools an Answer uses, such as Connectors",

  "consent.settings.title": "Data sent to other services",
  "consent.settings.empty": "Nothing is sent to other services.",
  "consent.settings.status.accepted": "Allowed",
  "consent.settings.status.declined": "Not allowed",
  "consent.settings.status.not-asked": "Not asked yet",
  "consent.settings.revoke": "Revoke",
  "consent.settings.askAgain": "Ask again",

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
