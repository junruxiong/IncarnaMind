import { useEffect, useState } from "react";
import type { Document, LibraryGroup } from "../../core/api";
import { core } from "./core";
import type { UnitCount } from "./fileInfo";
import { useAppStore } from "./store";

/**
 * A deck's slides or a workbook's sheets, counted from the text the core kept:
 * the Document itself holds only a PDF's pages. Null for another kind, while it
 * counts, and with no Document. Counted again when the file's version changes.
 */
export function useUnitCount(document: Document | null): UnitCount | null {
  const [count, setCount] = useState<number | null>(null);
  const id = document?.id;
  const kind = document?.kind;
  const contentHash = document?.contentHash;
  // biome-ignore lint/correctness/useExhaustiveDependencies: counted again when the file's version changes.
  useEffect(() => {
    setCount(null);
    if (!id || (kind !== "pptx" && kind !== "xlsx")) return;
    let current = true;
    core
      .readDocumentText(id)
      .then(({ pages }) => {
        if (!current) return;
        setCount(
          kind === "pptx"
            ? pages.length
            : new Set(pages.map((page) => page.label?.sheet ?? "")).size,
        );
      })
      .catch((error: unknown) => console.error(error));
    return () => {
      current = false;
    };
  }, [id, kind, contentHash]);
  if (count === null || count === 0) return null;
  return { key: kind === "pptx" ? "info.slides" : "info.sheets", count };
}

/** The Folder a Document is filed in, or null for Not in a Folder. */
export function useDocumentFolder(documentId: string): LibraryGroup | null {
  return useAppStore((state) => {
    const groupId = state.library?.assignments.find(
      (each) => each.documentId === documentId,
    )?.groupId;
    return groupId ? (state.library?.groups.find((each) => each.id === groupId) ?? null) : null;
  });
}
