# Each Mind is stored as one CRDT document (Yjs)

Each Mind's content is a single Yjs document that the editor edits directly. It is not stored as one database row per Block with a numeric order. We chose this because the hosted version must sync Minds across devices (ADR-0002). Yjs merges edits made offline on different devices without conflicts, so adding sync later means relaying updates, not migrating data. It also removes the job of translating editor changes into insert/reorder/delete calls, which the old frontend never got working. The cost is that finding Blocks in a Mind, such as the ones above a Question, means reading the document instead of running a query.

## Consequences

Version 1 has no sync, but it follows these rules so sync can be added without migrating data:

- Documents are identified by a hash of their content, so the same file on two devices is recognised as one Document.
- Every ID is a random UUID generated on the device, never an auto-increment number, so two devices can't produce the same ID.
- Deleting something marks it deleted rather than removing the row, so a later sync can tell other devices about the deletion.
- Secrets (API keys, OAuth tokens) are kept only on the device and never go in data that will be synced.
- Settings are split into per-device (window size, file paths) and per-User (default model, interface language). Only per-User settings will sync.
