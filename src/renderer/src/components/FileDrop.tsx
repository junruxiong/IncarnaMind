import { type DragEvent, useState } from "react";
import { useT } from "../i18n";

const carriesFiles = (event: DragEvent) => event.dataTransfer.types.includes("Files");

/**
 * Lets files be dropped anywhere on an element. `active` is true while files
 * are dragged over it. Counting enters and leaves keeps it steady as the
 * pointer crosses child elements.
 */
export function useFileDrop(onDrop: (files: File[]) => void) {
  const [depth, setDepth] = useState(0);
  return {
    active: depth > 0,
    handlers: {
      onDragEnter(event: DragEvent) {
        if (!carriesFiles(event)) return;
        event.preventDefault();
        setDepth((current) => current + 1);
      },
      onDragLeave(event: DragEvent) {
        if (!carriesFiles(event)) return;
        setDepth((current) => Math.max(0, current - 1));
      },
      onDragOver(event: DragEvent) {
        if (!carriesFiles(event)) return;
        // Accepting the drop also stops Electron from opening the file in the window.
        event.preventDefault();
        event.dataTransfer.dropEffect = "copy";
      },
      onDrop(event: DragEvent) {
        if (!carriesFiles(event)) return;
        event.preventDefault();
        setDepth(0);
        const files = Array.from(event.dataTransfer.files);
        if (files.length > 0) onDrop(files);
      },
    },
  };
}

/** Shown over the window while files are dragged over it: blue, as something to act on. */
export function DropOverlay() {
  const t = useT();
  return (
    <div
      data-testid="drop-overlay"
      className="pointer-events-none absolute inset-2 z-30 flex items-center justify-center rounded-xl border-2 border-dashed border-accent bg-accent-wash/85"
    >
      <p className="text-ui font-semibold text-accent">{t("documents.drop")}</p>
    </div>
  );
}
