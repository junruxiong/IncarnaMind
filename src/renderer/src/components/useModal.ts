import { useEffect, useRef } from "react";

/**
 * Keeps a native <dialog> shown as a modal while `open` is true. With
 * `focusDialog`, opening focuses the dialog itself (give it tabIndex={-1})
 * instead of its first control, e.g. so no choice looks picked before the
 * User picks one.
 */
export function useModal(open: boolean, { focusDialog = false } = {}) {
  const dialog = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    const element = dialog.current;
    if (!element) return;
    if (open && !element.open) {
      element.showModal();
      if (focusDialog) element.focus();
    }
    if (!open && element.open) {
      element.close();
      // Chromium gives the focus back to the text field the dialog took it from, but leaves the
      // caret outside it, so what is typed next goes nowhere (e.g. a follow-up after "Allow").
      const field = document.activeElement;
      if (
        (field instanceof HTMLTextAreaElement || field instanceof HTMLInputElement) &&
        field.selectionStart !== null &&
        field.selectionEnd !== null
      ) {
        field.setSelectionRange(field.selectionStart, field.selectionEnd);
      }
    }
  }, [open, focusDialog]);
  return dialog;
}
