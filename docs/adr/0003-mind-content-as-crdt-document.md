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
- **Content schema version.** A Mind's Yjs document records, in its settings map, the content schema version it was last written with (`CONTENT_SCHEMA_VERSION` is the newest this version understands: 2 since tables, #92; 3 since Proposals, #93). A Mind with none reads as version 1. The core records its version on every write and never lowers one. A Mind written by a newer version opens read-only with a notice to update: the editor's Yjs binding deletes node types and marks, and strips attributes, that its schema doesn't know, which would silently destroy another computer's content once sync exists. For such a Mind the core neither compacts nor accepts any update, so its Yjs state stays byte for byte as it was. Raise the version whenever a change would make an older version lose or misread content (a new node type, mark or attribute). A raise that only adds node types needs no rewrite of older Minds: their content is valid as it is, and they record the new version on their next write. The same holds for a raise that only adds marks or attributes: version 3 adds the `proposalInsert` and `proposalDelete` marks and a `proposal` attribute on Blocks and inline nodes, which older content simply doesn't have.
- **Proposals are content, kept with the Mind.** A change the AI made that waits for the User (a Proposal) is stored in the Mind's Yjs document itself, so it survives sync and reload and any device can resolve it: words it removes stay, under `proposalDelete`, and words it adds are there, under `proposalInsert`; a whole Block or an inline node it adds or removes carries the `proposal` attribute (the editor's Yjs binding keeps no marks on nodes). Each names its Proposal and the request that made it; a `proposalRequests` map beside the Blocks says which model made each request, and is kept once written. Accepting or rejecting is one Yjs change, defined once (`src/shared/proposals.ts`) for the core and the editor. What reads a Mind's text (Question context, exports) reads it as accepted: without what Proposals add, with what they would remove.
- **One reference shape.** A reference from one artifact to another is `{ artifactId, anchor }`. The anchor is a Block's stable ID for now (`{ kind: "block", blockId }`), never a position, so the reference survives edits and sync. It is used for a Block's origin and for the link back from a Note to the chat it came from.
- **Citations stay editor-independent.** A Citation's target is always a Document location (a Document and where in it), independent of the editor and of the artifact that holds the Citation. Citations are checked against Documents, never against Blocks or any artifact's structure, so a Citation means the same in a Mind, a chat or a later kind of artifact.

## Amendment: conversations are not Minds (2026-10-11, ADR-0017)

The assistant's conversations are kept by the core in SQLite, not as Minds (ADR-0017), so two things above change. The `chat` kind is retired: `mind` is the only kind this version opens, and `kind` stays for sheets, decks and boards. And a reference names its artifact's kind, `{ artifactKind, artifactId, anchor }`: a Mind's anchor is a Block, a conversation's is a message (`{ kind: "message", messageId }`). A reference stored without `artifactKind` is a Mind's. Citations stay as above: a Citation in a conversation is checked against its Document like any other.
