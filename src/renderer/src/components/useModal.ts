import { useEffect, useRef } from "react";

/** Keeps a native <dialog> shown as a modal while `open` is true. */
export function useModal(open: boolean) {
  const dialog = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    const element = dialog.current;
    if (!element) return;
    if (open && !element.open) element.showModal();
    if (!open && element.open) element.close();
  }, [open]);
  return dialog;
}
