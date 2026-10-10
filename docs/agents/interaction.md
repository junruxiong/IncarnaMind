# Interaction rules

Read this before building or changing any UI, whether you are a person or a coding agent. It sits beside the visual rules in `DESIGN.md` and the words in [`CONTEXT.md`](../../CONTEXT.md).

**The bar:** IncarnaMind should be easy to use and easy to pick up. "Simple" means a first-time User gets what they came for without learning the app or reading help. It doesn't mean fewer features. If a screen needs explaining, it isn't done.

## The rules

Each rule can be checked. A UI change that breaks one says why in its pull request.

1. **Edit in place.** Anything with a name the User can see (a Mind's title, a Document's name, a Folder, a Tag) can be renamed where it is: double-click it, or select it and press Enter. Enter saves; Esc cancels. No dialog for a rename.
2. **Drag where it makes sense, with a way that isn't dragging.** Documents and Minds drag onto Folders; Documents drag onto Tags and into the composer; files from Finder or Explorer drop onto a Folder. The drop target highlights while dragging. Every drag has a menu or keyboard way to do the same thing (Move to…, Tags…).
3. **Keep the User's place.** Nothing replaces the Mind the User is in. A Folder, the Library, a Document or Settings opens in a tab, beside the Mind, or over it as something closed with Esc. A "Back to…" button is a sign this rule is broken.
4. **Undo instead of asking.** Reversible actions happen at once and offer Undo: ⌘Z / Ctrl+Z, and a short line such as "Moved 3 Documents to Finance · Undo". Ask first only before something that can't be undone, and say what will be lost.
5. **Every object has a right-click menu,** with the same order everywhere: Open, the object's main actions, Move to…, Tags…, Rename, then Delete last, separated. The ⋯ button shows the same menu.
6. **One way to select.** Click selects; Shift-click extends; ⌘-click / Ctrl-click adds or removes. An action applies to everything selected, and says how many ("Delete 3 Documents").
7. **The keyboard reaches everything.** Every action works without a mouse and can be found with ⌘K / Ctrl+K. Focus is always visible. Esc closes the innermost thing that's open.
8. **Answer at once.** The UI responds to every action within 100 ms, updating before the core confirms where it's safe. Anything slower than a second shows progress. Nothing freezes the window.
9. **One way to do each thing.** One primary action per screen. Advanced choices live behind ⋯ or in Settings. A new setting needs a reason in the pull request; prefer a good default.
10. **Say it plainly.** Use the words in `CONTEXT.md`. Say what happened and what the User can do next. A problem stays on screen until dismissed and offers one action (see #157).
11. **Hold the layout.** Every screen works at every window size the app supports, with and without the viewer open, in English and in Chinese. Text wraps by word, never by letter, and long names truncate with the full name in a tooltip.
12. **Match the design.** Every UI ticket names its design boards (the design canvas's main page, "IncarnaMind: all screens"). Build to them: layout, labels, chips, states. Where the built screen differs, say why in the pull request, or fix it.
13. **Use the shared building blocks.** Where the app has one (selection, drag and drop, undo, the right-click menu registry, in-place rename, the command list behind ⌘K), use it rather than building a one-off, so the same gesture does the same thing everywhere.

## In a pull request that changes UI

- Tick the interaction checklist in the pull request template.
- Add before and after screenshots, at a narrow and a wide window, in English and Chinese where text changed.
- Put a screenshot of the built screen beside the design board it implements, and list every difference with its reason.
- Test it like a person: with the mouse and the keyboard, in the built app, not only in unit tests.
