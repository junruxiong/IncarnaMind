# Documents are indexed in place, not copied

Documents stay where the User keeps them. IncarnaMind indexes each file at its own path and never copies it into the data folder. The User adds Linked folders (a papers folder, Zotero's storage folder) and IncarnaMind keeps them in sync: new files become Documents, changed files are indexed again, and removed files are marked missing. A single file added on its own is indexed at its path too. Each folder inside a Linked folder is a Folder in the sidebar; in-app Folders that the User filed Documents into by hand are gone, and Tags remain for grouping across folders.

What the index keeps is enough for Citations without the file:
- Each Document's page text and Passages live in SQLite, and the Citation check compares quotes against that stored text, not the file.
- Each version of a file is identified by the SHA-256 of its content.
- When a file changes, the new version is indexed and becomes what search and new Answers use. The old version's pages are kept for as long as a Citation points at them, so the Citation is still checked against the text it quoted, and it says the Document changed after it was cited.
- A file that moves or is renamed is found again by its content fingerprint, so its Citations and Search scopes follow it.
- A file that disappears leaves a missing Document whose text and Citations remain. Only the rendered page, which needs the file, is lost.

This replaces the spec's original choice (issue #20, story 41: "copied into the app's data folder, so that moving or deleting the originals breaks nothing") and narrows ADR-0008: the data folder no longer holds Document files.

## Considered options

- **Copy every added file into the data folder** (the previous design): Citations and backups never depend on the User's own files, but a research library is duplicated (often gigabytes), files edited in place go stale, and a folder can't stay in sync.
- **Copy single files, index Linked folders in place**: protects files dragged in from Downloads, but two storage models are harder to explain and to build.
- **Keep in-app Folders alongside Linked folders**: more flexible, but two kinds of Folder in one tree confuse; Tags already cover grouping across folders.

## Consequences

- Backing up the data folder backs up the Minds, the index and settings, but not the Documents. The User's files are theirs to back up.
- IncarnaMind watches Linked folders while it runs and reconciles them at start-up (size and modified time first, then the content hash), because files change while it is closed.
- Opening a Document in another app opens the original; no temporary copies.
- Linked folders on cloud drives can hold placeholder files that aren't downloaded; those wait until they're readable.
