# Each Mind is stored as one CRDT document (Yjs)

Each Mind's content is a single Yjs document that the editor edits directly. It is not stored as one database row per Block with a numeric order. We chose this because the hosted version must sync Minds across devices (ADR-0002). Yjs merges edits made offline on different devices without conflicts, so adding sync later means relaying updates, not migrating data. It also removes the job of translating editor changes into insert/reorder/delete calls, which the old frontend never got working. The cost is that finding Blocks in a Mind, such as the ones above a Question, means reading the document instead of running a query.

## Consequences

Version 1 has no sync, but it follows these rules so sync can be added without migrating data:

- Documents are identified by a hash of their content, so the same file on two devices is recognised as one Document.
- Every ID is a random UUID generated on the device, never an auto-increment number, so two devices can't produce the same ID.
- Deleting something marks it deleted rather than removing the row, so a later sync can tell other devices about the deletion.
- Secrets (API keys, OAuth tokens) are kept only on the device and never go in data that will be synced.
- Settings are split into per-device (window size, file paths) and per-User (default model, interface language). Only per-User settings will sync.

## Amendment: kinds, a content schema version, one reference shape, and Citations

Chats now, and sheets, decks and boards after the alpha, are stored the way Minds are, so the storage leaves room for them and stays safe once Minds sync between computers running different versions.

- **Kind.** Each Mind has a `kind` on its row, `mind` by default (`chat` for a chat). A Mind of a kind this version doesn't know is listed, but it opens as "Update IncarnaMind to open this", never in the editor, and the core refuses to write to it.
- **Content schema version.** A Mind's Yjs document records, in its settings map, the content schema version it was last written with (`CONTENT_SCHEMA_VERSION` is the newest this version understands, 1 today). A Mind with none reads as version 1. The core records its version on every write and never lowers one. A Mind written by a newer version opens read-only with a notice to update: the editor's Yjs binding deletes node types and marks, and strips attributes, that its schema doesn't know, which would silently destroy another computer's content once sync exists. For such a Mind the core neither compacts nor accepts any update, so its Yjs state stays byte for byte as it was. Raise the version whenever a change would make an older version lose or misread content (a new node type, mark or attribute).
- **One reference shape.** A reference from one artifact to another is `{ artifactId, anchor }`. The anchor is a Block's stable ID for now (`{ kind: "block", blockId }`), never a position, so the reference survives edits and sync. It is used for a Block's origin and for the link back from a Note to the chat it came from.
- **Citations stay editor-independent.** A Citation's target is always a Document location (a Document and where in it), independent of the editor and of the artifact that holds the Citation. Citations are checked against Documents, never against Blocks or any artifact's structure, so a Citation means the same in a Mind, a chat or a later kind of artifact.
