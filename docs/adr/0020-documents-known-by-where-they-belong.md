# Documents are known by where they belong, not by their path on one computer

Since Documents were indexed in place (ADR-0010), a Document's unique key was its file's absolute path on this computer. That path is different on every computer, and changes when the User moves or renames a Linked folder, so nothing could tell that a Document on one computer is the same as one on another, and a moved Linked folder lost every Document to a copy that started over. A User's Minds will sync between their computers, and later with a phone, so before alpha Users build up data, a Document is known by where its file **belongs**, which is the same everywhere, and its path is only where that place is on this computer.

## The identity rules

- **A Document is its id.** A random UUID, made once. Everything that points at a Document points at its id: Citations (with the version they quote), Tags, Folders, Search scopes, check results and every retained version (ADR-0010's amendment). Its id never changes when its file moves.
- **Where its file belongs (`documents.source_kind`).** One of:
  - **In a Linked folder:** the Linked folder's id and the file's path inside it (`relative_path`), with "/" between names on every system. At most one live Document has a Linked folder and relative path (`documents_by_place`). A Linked folder's id is random, made when it is linked, and its Folders' ids are made from that id and their relative paths, so they too are the same wherever the folder is.
  - **In the data folder:** a file IncarnaMind keeps, a web page's snapshot, saved text, or a copy the old layout made: its path inside the data folder.
  - **A file added on its own:** nothing but its id. Where it is is this computer's alone, as a picked file is.
- **Its version is its content hash.** The SHA-256 of the file's bytes identifies a version, as before. A Citation names its Document and the version it quotes, and the first 16 digits of that hash are the version's fingerprint an outside agent's Citation carries (#406). The same content in two places is still two Documents, and the same file is still found again by its content when it moves inside or between the folders IncarnaMind watches.
- **What belongs to this computer.** A Linked folder's `path`, a Document's `path`, `file_mtime_ms` and `file_status` are what this computer has seen and where it keeps things: derived from the identity (a Linked folder's path and the relative path, or the data folder and the relative path), or seen here (a file added on its own). A later sync never carries them: a missing file on one computer can be there on another.
- **The workspace.** The data folder is one workspace, known by a random id made at its first start (the `workspace` table, one live row), with no account behind it and nothing sent. A copy of the data folder is the same workspace. A User's computers that sync will share one workspace id.
- **Minds' content structure.** Each Mind records the content schema version it was last written with (ADR-0003's amendment), so a later sync can tell old content from new.

## When a Linked folder is somewhere else

A Linked folder whose folder can't be reached is unavailable (ADR-0010): its Documents stay searched from their stored text. **Locate folder…**, in the Linked folder's menu, points the same Linked folder at where it is now, on this computer or after the data folder was opened on another one. Each of its Documents is then looked for at its relative path there, in one pass: a file whose size and modified time match isn't read; one that doesn't is hashed, and the same content is the same version, so nothing is processed again; a new version is processed as a new version; a file that isn't there is missing; a new file is a new Document. Every Document keeps its id, versions, Tags, Folder and the Citations that quote it. The Linked folder's own Folder takes the folder's new name.

It changes nothing, and says why, for a folder that holds none of the Linked folder's files (a few are looked for, so a wrong folder can't make every Document missing), one inside or around another Linked folder, or one where a file of it was already added on its own. IncarnaMind doesn't search the disk for a folder that moved: it could be anywhere, on a drive that isn't plugged in, and looking through the User's folders is theirs to allow.

The data folder's own files follow the data folder: at each start, a Document kept in the data folder is looked for at its relative path in the data folder as it is now.

## How this relates to keeping files in place

Files stay where the User keeps them, and IncarnaMind still only reads them (ADR-0010). Only the key changes: ADR-0010's "a Document is a file at a path" becomes "a file at a place: its Linked folder and its path inside it, the data folder, or a file of its own". ADR-0003's first sync rule, "Documents are identified by a hash of their content", which ADR-0010 had already narrowed, is replaced by these rules: the hash identifies a version, not a Document.

## Later kinds

- **A file imported on a phone** is a copy in the app's own container, as a web page's snapshot is in the data folder, with the picker's bookmark kept as a hint to the original, never as its identity. It will be one more `source_kind`. A kind this version doesn't know is left as it is.
- **A file moved and edited while IncarnaMind didn't see it** (#389) is recognised by a second clue: the file's identity on its disk (device, file number and creation time from `stat`), or one clear candidate. Like `path`, that clue belongs to this computer and is stored beside it; it matches a file to its Document, and never becomes part of what the Document is.

## Consequences

- **Migration 55** adds `source_kind` and `relative_path` to `documents`, fills them in for every Document of a Linked folder from its path (on macOS, Linux and Windows), adds `documents_by_place`, and the `workspace` table. The core marks the files in the data folder at its next start, since only it knows where the data folder is. No file is read and no Document is processed again, and Citations stay found.
- **A Document the User removed from a Linked folder** stays out by its relative path, so it stays out wherever the folder is located.
- **The core's interface** gains `locateLinkedFolder`. `Document.path` and `LinkedFolder.path` are where things are on this computer.
- **Not built here:** sync itself; locating a file added on its own that is unavailable on another computer (Locate file… in the viewer finds a missing one, as before); Linked folders made inside the data folder, such as the example Documents', which have to be located like any other once the data folder moves.

## Considered options

- **The content hash as the identity** (ADR-0003's first rule): an edit would make another Document, and two copies of a file would be one. ADR-0010 had already left it.
- **The absolute path, with paths rewritten when a Linked folder moves:** nothing to migrate, but two computers would still disagree about every key, and a sync would have nothing to match on.
- **This computer's paths in a table of their own now:** the purest split, but every query that reads a Document's path would join it, for no gain until sync is built; the columns are named in this ADR as this computer's, and a sync can carry the others.
- **Finding a moved Linked folder by itself,** by searching the disk for its files: slow, often wrong, and it reads folders the User never chose.
- **A file added on its own known by its path inside the User's home folder:** it breaks as soon as the file moves, and home folders differ between systems.
