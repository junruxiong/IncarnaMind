# Interaction rules

Read this before building or changing any UI, whether you are a person or a coding agent. It sits beside the visual rules in `DESIGN.md` and the words in [`CONTEXT.md`](../../CONTEXT.md).

**The bar:** IncarnaMind should be easy to use and easy to pick up. "Simple" means a first-time User gets what they came for without learning the app or reading help. It doesn't mean fewer features. If a screen needs explaining, it isn't done.

## The rules

Each rule has a check a reviewer can apply. A UI change that breaks a rule says why in its pull request.

0. **Borrow what people already know.** Where Claude, ChatGPT, Finder, Notion or Granola has a settled way to do something, IncarnaMind does it that way. Chinese Users' apps (Feishu, Yuque, WeChat) count as familiar too. A new gesture needs a reason the familiar one can't serve.
   - Check: the issue or pull request names the app whose convention each gesture follows, or says why none fits. The table below lists the ones already chosen.
1. **Edit in place.** Any name the User sees (a Mind, a Chat, a Document, a Folder, a Tag, a tab) is renamed where it is: double-click it, or select it and press Enter or F2. Enter saves, Esc cancels, a blank name changes nothing, and Enter while a Chinese input method is composing never cuts the name off.
   - Check: an end-to-end test renames each object by double-click and by keyboard, and no dialog opens.
2. **Drag where it makes sense, always with another way.** Documents and Minds drag onto Folders; Documents drag onto Tags and into the composer; files from Finder or Explorer drop onto a Folder, the sidebar or the composer. A target highlights only while it can accept the drop, and the drop says what happened, with Undo. Every drag has a menu item and a key: Move to… (⇧⌘M), Tags….
   - Check: every drag in the change names its menu and keyboard twin.
3. **Keep the User's place.** Nothing replaces the Mind the User is in. The Library, a Folder or a Tag opens as a tab; a Document opens beside the Mind; Settings opens over it and closes with Esc.
   - Check: no label starts with "Back to". One that does means this rule is broken.
4. **Undo instead of asking.** Anything the User can take back happens at once and offers Undo: ⌘Z / Ctrl+Z, and one short line such as "Moved 3 Documents to Finance · Undo". Deleting a Mind, Chat, Document, Folder or Tag moves it to Recently deleted, where it waits 30 days, with no question asked. Ask first only before something that can't be taken back, and say what will be lost.
   - Check: each action in the change is marked reversible (with its undo) or irreversible (with its confirmation text).
5. **Every object has one right-click menu, in one order.** Open (and Open in new tab), the object's own actions, Move to…, Tags…, Rename, the file's actions (Show in Finder, Copy path), then Delete last, after a separator. The ⋯ button opens the same menu, and each item shows its shortcut.
   - Check: right-click every kind of object the change shows, and compare with this order.
6. **One way to select.** In every list: a click opens; ⌘-click / Ctrl-click adds or removes; Shift-click extends; ⌘A selects everything in view; Esc clears. An action applies to everything selected and says how many ("Delete 3 Documents"). Dragging one of several selected items drags them all.
   - Check: the same selection test passes in the sidebar and the Library.
7. **The keyboard reaches everything, and ⌘K finds it.** Every action works without a mouse and is in ⌘K / Ctrl+K, which also finds any Mind, Chat, Document, Folder or Tag by name (Minds by their contents too), matching pinyin for Chinese names. Arrow keys move through the sidebar. Focus is always visible. Esc closes the innermost thing that's open.
   - Check: every new action has a ⌘K entry and a keyboard path.
8. **Answer at once.** Every action shows its result within 100 ms of the input (p95, on a Library of 2,000 Documents), updating before the core confirms where that's safe and rolling back with a problem line if the core refuses. Anything that takes over a second shows progress where it happens. Nothing freezes the window.
   - Check: the speed test; by hand, nothing waits on a spinner for an action that should be instant.
9. **One way to do each thing, with more behind it.** One primary action per screen and one visible place to add things. Advanced choices live behind ⋯ or in Settings. A new setting needs a reason in the pull request; prefer a good default.
   - Check: the change lists each visible way to reach its action; a second visible way needs a reason.
10. **Say it plainly.** Use the words in `CONTEXT.md`. Say what happened and what the User can do next. A problem stays where it happened, as one line with one action, until dismissed (see #157); no toast for an error.
    - Check: the copy reads well in English and Chinese, in `CONTEXT.md`'s words.
11. **Hold the layout.** Every screen works at every window size the app supports, with and without the viewer open, in English and in Chinese. Text wraps by word, never by letter, and long names truncate with the full name in a tooltip.
    - Check: screenshots at a narrow and a wide window.
12. **Match the design.** Every UI issue names its design boards (the design canvas's main page, "IncarnaMind: all screens"). Build to them: layout, labels, chips, states. Where the built screen differs, say why in the pull request, or fix it.
    - Check: the pull request shows the built screen beside each board, with every difference listed.
13. **Use the shared building blocks.** Selection, drag and drop, undo, the right-click menus, rename in place, the command list behind ⌘K and the menus, and optimistic updates each have one shared implementation, so the same gesture does the same thing everywhere. Where a block doesn't exist yet, don't build a one-off: say so in the pull request and keep the current behaviour, or build the block in its own issue.
    - Check: code review; no `onContextMenu`, `draggable` or undo handling outside the shared blocks.

## Conventions already chosen

| Task | How IncarnaMind does it | Who does it that way |
|---|---|---|
| Start | A new Mind or Chat opens on the composer; its title comes from the first Question and can be renamed in place | Claude, ChatGPT |
| Ask in a Mind | The composer at the bottom, ⌘J to focus it, scoped by where it was opened | Granola |
| Find anything | ⌘K searches Minds, Chats, Documents, Folders and Tags, and runs any action | ChatGPT, Claude, Notion, Linear |
| Rename | Double-click, or select and press Enter or F2 | Finder, Notion, Obsidian |
| File into a Folder | Drag onto the Folder, or Move to… (⇧⌘M) | ChatGPT projects, Notion, Granola, Finder |
| One Folder per Document | A drop on a Folder moves the Document; Tags group Documents across Folders | Finder |
| Tags | Created, renamed, recoloured, merged and deleted in the sidebar's Tags view; a rename applies everywhere | Finder, Bear, Apple Notes |
| Delete | Moves to Recently deleted for 30 days, no question asked | Notion, Granola, Apple Notes, Finder, Feishu |
| Undo | ⌘Z after any change, not only typing | Linear, Things |
| Look inside a file | Space shows a quick preview beside the list; Enter opens it in the viewer | Finder's Quick Look |
| Files dropped on the composer | They become Documents and that Question's scope | Claude, ChatGPT |
| Insert | "/" in the note, matching pinyin for Chinese | Notion, Feishu, Yuque |
| Changes the AI proposes | Shown as changes in the note; one key accepts all, one rejects all | Cursor, Notion AI |

## In an issue that changes UI

Add the interaction checklist from the pull request template to the acceptance criteria, name the design boards, and name the convention each gesture follows (rule 0).

## In a pull request that changes UI

- Tick the interaction checklist in the pull request template.
- Add before and after screenshots, at a narrow and a wide window, in English and Chinese where text changed.
- Put a screenshot of the built screen beside the design board it implements, and list every difference with its reason.
- Test it like a person: with the mouse and the keyboard, in the built app, not only in unit tests.
