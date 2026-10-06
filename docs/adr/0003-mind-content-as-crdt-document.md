# Each Mind is stored as one CRDT document (Yjs)

Each Mind's content is a single Yjs document that the editor edits directly. It is not stored as one database row per Block with a numeric order. We chose this because the hosted version must sync Minds across devices (ADR-0002). Yjs merges edits made offline on different devices without conflicts, so adding sync later means relaying updates, not migrating data. It also removes the job of translating editor changes into insert/reorder/delete calls, which the old frontend never got working. The cost is that finding Blocks in a Mind, such as the ones above a Question, means reading the document instead of running a query.

## Consequences

Documents are identified by a hash of their content, so the same file on two devices is recognised as one Document.
