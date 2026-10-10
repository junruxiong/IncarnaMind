import { type DragEvent, useState } from "react";
import { useT } from "../i18n";
import { linksIn } from "../webPages";

const carriesFiles = (event: DragEvent) => event.dataTransfer.types.includes("Files");
const carriesLinks = (event: DragEvent) => event.dataTransfer.types.includes("text/uri-list");

/**
 * Lets files and folders be dropped anywhere on an element, and with
 * `onLinks`, links from a browser too. `active` is true while they are
 * dragged over it. Counting enters and leaves keeps it steady as the pointer
 * crosses child elements. `onDrop` gets every dropped item, and those of
 * them the drop says are folders; `onLinks` the web addresses dropped.
 */
export function useFileDrop(
  onDrop: (files: File[], folders: File[]) => void,
  onLinks?: (addresses: string[]) => void,
) {
  const [depth, setDepth] = useState(0);
  const accepts = (event: DragEvent) =>
    carriesFiles(event) || (onLinks !== undefined && carriesLinks(event));
  return {
    active: depth > 0,
    handlers: {
      onDragEnter(event: DragEvent) {
        if (!accepts(event)) return;
        event.preventDefault();
        setDepth((current) => current + 1);
      },
      onDragLeave(event: DragEvent) {
        if (!accepts(event)) return;
        setDepth((current) => Math.max(0, current - 1));
      },
      onDragOver(event: DragEvent) {
        if (!accepts(event)) return;
        // Accepting the drop also stops Electron from opening the file in the window.
        event.preventDefault();
        event.dataTransfer.dropEffect = "copy";
      },
      onDrop(event: DragEvent) {
        if (!accepts(event)) return;
        event.preventDefault();
        setDepth(0);
        // Read now: the drop's items are only there while it is handled.
        const items = Array.from(event.dataTransfer.items).filter((item) => item.kind === "file");
        const folders = items.flatMap((item) => {
          const file = item.webkitGetAsEntry()?.isDirectory ? item.getAsFile() : null;
          return file ? [file] : [];
        });
        const files = Array.from(event.dataTransfer.files);
        if (files.length > 0) onDrop(files, folders);
        else if (onLinks) {
          const addresses = linksIn(event.dataTransfer);
          if (addresses.length > 0) onLinks(addresses);
        }
      },
    },
  };
}

/**
 * Shown over the window while files are dragged over it: blue, as something
 * to act on. Opaque, so nothing under it (an empty Mind's words) shows through its own.
 */
export function DropOverlay() {
  const t = useT();
  return (
    <div
      data-testid="drop-overlay"
      className="pointer-events-none absolute inset-2 z-30 flex items-center justify-center rounded-xl border-2 border-dashed border-accent bg-accent-wash"
    >
      <p className="text-ui font-semibold text-accent">{t("documents.drop")}</p>
    </div>
  );
}
