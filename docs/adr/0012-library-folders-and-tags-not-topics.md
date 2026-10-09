# The Library organises Documents into Folders and Tags, not generated Topics

The Library puts each Document into one in-app Folder (or Unsorted) and gives it Tags, in a single Organize action. A classifier makes the assignment: the User's connected model, Auto local routing between local decision models, or Clef-Flash run locally for image-heavy PDFs. Folders are starter ones the User picks or their own, each with a description of what belongs there. The User's manual choices always win and are kept. Original files never move; Folders only change IncarnaMind's index, and source locations (Linked folders as they are on disk) are shown separately.

We replaced the approved design that grouped Documents into a two-level tree of generated Topics by clustering their vectors (`docs/designs/library-structure-view.md`, engineering-reviewed 2026-10-07). There were two reasons. The User judged it too complicated: they wanted Documents in the right folders with the right tags, which is simple to understand and correct. And the check before building showed that clustering with the built-in embedding model splits Topics by language: in held-out English–Chinese same-subject pairs, 0 of 5 landed together (#51). The classifier reads the Documents' content, so a Chinese and an English Document on one subject can share a Folder.

## Considered options

- **Clustering into generated Topics, with model-written names:** rejected, as above. The grouping functions and `npm run eval:grouping` remain for measurement.
- **Moving or renaming the User's files on disk:** rejected, because ADR-0010 indexes Documents in place and IncarnaMind never changes a Linked folder.

## Consequences

- Organising needs a model: a cloud model under the existing consent rules, or a local one. Without one, Documents stay Unsorted until the User files them by hand.
- "Folder" now means an in-app Folder. A folder on disk is part of a source location (CONTEXT.md).
- Bridges from a Folder into writing, and year, format and status filters, are not built yet (#59).
