import { flip, hide, offset, shift, type VirtualElement } from "@floating-ui/dom";
import { isMacOS } from "@tiptap/core";
import { type KeyboardEvent, useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { Document } from "../../../core/api";
import { viewerSelectionHandle } from "../commands/handles";
import { AddToMindPicker } from "../components/assistant/AddToMindPicker";
import { CopyLineIcon, PlusLineIcon } from "../components/lineIcons";
import { Popover } from "../editor/Popover";
import { useT } from "../i18n";
import { pdfSelection, selectionEnd } from "./selectedText";
import {
  addSelectionToNote,
  addsWithoutAsking,
  copySelectionWithCitation,
  type ViewerSelection,
} from "./selectionActions";

interface Held {
  selection: ViewerSelection;
  range: Range;
}

/** A field the User types in elsewhere (⌘K, the composer, the Mind picker): the selection is still the one acted on. */
const typesInField = (element: Element | null) =>
  element instanceof HTMLInputElement || element instanceof HTMLTextAreaElement;

const inPopover = (target: EventTarget | null) =>
  target instanceof Element && target.closest(".editor-popover") !== null;

/**
 * The small menu under text selected in a PDF (#101; DESIGN.md › Viewer ›
 * Selecting to ask): Add to note and Copy with citation. It shows once the
 * mouse comes up on a selection, or Shift extends one, and goes with it. ⌘/
 * moves into it, ↑ ↓ go through it, Esc closes it, as the Mind's selection
 * menu does. ⌘K reaches both while the selection lasts.
 */
export function SelectionMenu({
  document,
  section,
}: {
  document: Document;
  /** The viewer it is the menu of. */
  section: HTMLElement | null;
}) {
  const t = useT();
  const [held, setHeld] = useState<Held | null>(null);
  const [open, setOpen] = useState(false);
  const [picking, setPicking] = useState(false);
  const menu = useRef<HTMLDivElement>(null);
  const first = useRef<HTMLButtonElement>(null);
  const [adding, setAdding] = useState<HTMLButtonElement | null>(null);
  const addButton = useCallback((element: HTMLButtonElement | null) => {
    first.current = element;
    setAdding(element);
  }, []);
  /** Where the focus was before ⌘/ moved it into the menu, to go back to. */
  const cameFrom = useRef<HTMLElement | null>(null);
  const [focusWanted, setFocusWanted] = useState(false);

  useEffect(() => {
    if (!section || document.kind !== "pdf") return;
    const read = () => {
      const selection = window.getSelection();
      const range =
        selection && !selection.isCollapsed && selection.rangeCount > 0
          ? selection.getRangeAt(0)
          : null;
      if (!range || !section.contains(range.commonAncestorContainer)) return;
      const parts = pdfSelection(range, section);
      setHeld(
        parts.length > 0 ? { selection: { document, parts }, range: range.cloneRange() } : null,
      );
      setOpen(parts.length > 0);
      setPicking(false);
    };
    const onDown = (event: PointerEvent) => {
      if (inPopover(event.target)) return;
      setOpen(false);
      setHeld(null);
    };
    // Once the browser has settled the selection the press made.
    const onUp = (event: PointerEvent) => {
      if (event.button === 0 && !inPopover(event.target)) setTimeout(read);
    };
    const onKeyUp = (event: globalThis.KeyboardEvent) => {
      if (event.shiftKey || event.key === "Shift") read();
    };
    const onChange = () => {
      const selection = window.getSelection();
      if (selection && !selection.isCollapsed && selection.rangeCount > 0) {
        if (!section.contains(selection.getRangeAt(0).commonAncestorContainer)) setHeld(null);
        return;
      }
      const focused = window.document.activeElement;
      if (inPopover(focused)) return;
      if (typesInField(focused)) setOpen(false);
      else setHeld(null);
    };
    section.addEventListener("pointerdown", onDown);
    section.addEventListener("pointerup", onUp);
    section.addEventListener("keyup", onKeyUp);
    window.document.addEventListener("selectionchange", onChange);
    return () => {
      section.removeEventListener("pointerdown", onDown);
      section.removeEventListener("pointerup", onUp);
      section.removeEventListener("keyup", onKeyUp);
      window.document.removeEventListener("selectionchange", onChange);
    };
  }, [section, document]);

  // ⌘/ moves into the menu, opening it again over the selection if Esc closed it.
  useEffect(() => {
    if (!held) return;
    const onKey = (event: globalThis.KeyboardEvent) => {
      const command = isMacOS() ? event.metaKey : event.ctrlKey;
      if (event.key !== "/" || !command || event.altKey || event.shiftKey || event.isComposing) {
        return;
      }
      if (typesInField(window.document.activeElement)) return;
      event.preventDefault();
      if (!menu.current?.contains(window.document.activeElement)) {
        cameFrom.current = window.document.activeElement as HTMLElement | null;
      }
      if (open) first.current?.focus();
      else {
        setFocusWanted(true);
        setOpen(true);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [held, open]);

  const anchor = useMemo<VirtualElement | null>(
    () =>
      held && {
        getBoundingClientRect: () => selectionEnd(held.range),
        contextElement:
          held.range.endContainer instanceof Element
            ? held.range.endContainer
            : (held.range.endContainer.parentElement ?? undefined),
      },
    [held],
  );

  const close = useCallback(() => {
    setOpen(false);
    setPicking(false);
    setFocusWanted(false);
    if (menu.current?.contains(window.document.activeElement)) cameFrom.current?.focus();
  }, []);

  const actions = useMemo(
    () =>
      held && {
        addToNote: () => {
          if (addsWithoutAsking()) {
            close();
            void addSelectionToNote(held.selection);
          } else {
            setOpen(true);
            setPicking(true);
          }
        },
        copy: () => {
          close();
          void copySelectionWithCitation(held.selection);
        },
      },
    [held, close],
  );
  // Only while there is a selection: another Document's tab may hold one.
  useEffect(() => {
    if (!actions) return;
    viewerSelectionHandle.current = actions;
    return () => {
      if (viewerSelectionHandle.current === actions) viewerSelectionHandle.current = null;
    };
  }, [actions]);

  const onMenuKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    const items = [...(menu.current?.querySelectorAll<HTMLElement>('[role="menuitem"]') ?? [])];
    const at = items.indexOf(window.document.activeElement as HTMLElement);
    const step = event.key === "ArrowDown" ? 1 : event.key === "ArrowUp" ? -1 : 0;
    const next =
      event.key === "Home"
        ? items[0]
        : event.key === "End"
          ? items.at(-1)
          : step
            ? items[(at + step + items.length) % items.length]
            : undefined;
    if (!next) return;
    event.preventDefault();
    next.focus();
  };

  if (!held || !actions || !anchor || !open) return null;
  return (
    <>
      <Popover
        anchor={anchor}
        placement="bottom-start"
        middleware={() => [offset(6), flip(), shift({ padding: 8 }), hide()]}
        role="menu"
        aria-label={t("viewer.selection.label")}
        data-testid="viewer-selection-menu"
        {...(focusWanted && { initialFocus: first })}
        onDismiss={close}
        onKeyDown={onMenuKeyDown}
        className="viewer-selection-menu"
      >
        <div ref={menu} className="editor-menu">
          <button
            ref={addButton}
            type="button"
            role="menuitem"
            aria-haspopup={addsWithoutAsking() ? undefined : "dialog"}
            data-testid="viewer-selection-add"
            // The selection stays as it is.
            onMouseDown={(event) => event.preventDefault()}
            onClick={actions.addToNote}
            className="editor-menu-item"
          >
            <PlusLineIcon className="size-4 shrink-0 text-ink-secondary" />
            <span className="flex-1 truncate">{t("viewer.selection.addToNote")}</span>
          </button>
          <button
            type="button"
            role="menuitem"
            data-testid="viewer-selection-copy"
            onMouseDown={(event) => event.preventDefault()}
            onClick={actions.copy}
            className="editor-menu-item"
          >
            <CopyLineIcon className="size-4 shrink-0 text-ink-secondary" />
            <span className="flex-1 truncate">{t("viewer.selection.copy")}</span>
          </button>
        </div>
      </Popover>
      {picking && adding && (
        <AddToMindPicker
          anchor={adding}
          focusedMindId={null}
          atCursor={false}
          newFolderId={document.folderId}
          onChoose={(target) => {
            close();
            void addSelectionToNote(held.selection, target);
          }}
          onClose={() => setPicking(false)}
        />
      )}
    </>
  );
}
