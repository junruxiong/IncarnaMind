import { useEffect } from "react";
import { useT } from "../i18n";
import { CloseIcon } from "./icons";

/**
 * The Document viewer: a panel on the right that is closed by default, like an
 * artifact panel. It renders only while open. Documents and Citations open it in
 * later tickets; for now only the test hook and the dev shortcut do.
 */
export function ViewerPanel({ width, onClose }: { width: number; onClose(): void }) {
  const t = useT();

  // Esc closes the panel, unless it is closing a dialog.
  useEffect(() => {
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key !== "Escape" || event.defaultPrevented) return;
      if (event.target instanceof Element && event.target.closest("dialog")) return;
      onClose();
    };
    window.addEventListener("keydown", closeOnEscape);
    return () => window.removeEventListener("keydown", closeOnEscape);
  }, [onClose]);

  return (
    <section
      data-testid="viewer"
      aria-label={t("viewer.label")}
      className="flex min-w-[220px] shrink flex-col"
      style={{ flexBasis: width }}
    >
      <div className="flex h-10 shrink-0 items-center justify-end pr-2">
        <button
          type="button"
          data-testid="viewer-close"
          aria-label={t("viewer.close")}
          title={t("viewer.close")}
          onClick={onClose}
          className="rounded-[9px] p-1 text-gray-500 hover:bg-gray-300 hover:text-gray-700"
        >
          <CloseIcon className="size-4" />
        </button>
      </div>
      <div className="flex-grow rounded-tl-[6px] bg-white" />
    </section>
  );
}
