import { useEffect, useState } from "react";
import type { SearchScope, UnreadFile, UnreadReason } from "../../core/api";
import type { MessageKey } from "../../shared/i18n";
import { core } from "./core";
import { useAppStore } from "./store";

/** What each reason says, and what to do about it (#740). */
export const UNREAD_REASON_KEYS: Record<UnreadReason, MessageKey> = {
  format: "unread.reason.format",
  scan: "unread.reason.scan",
  password: "unread.reason.password",
  damaged: "unread.reason.damaged",
  "too-large": "unread.reason.too-large",
  pages: "unread.reason.pages",
};

/**
 * The files in a Search scope that couldn't be read (#740): asked of the core,
 * and again when a Document's state or a Linked folder's list of files in a
 * format not read changes. Empty until it answers, and when everything was read.
 */
export function useUnreadFiles(scope: SearchScope | null): UnreadFile[] {
  const [files, setFiles] = useState<UnreadFile[]>([]);
  // A string that changes when what could be unread does, so a Document that is ready doesn't ask again.
  const state = useAppStore((store) =>
    [
      ...store.documents
        .filter((each) => each.status === "failed" || each.status === "no-text")
        .map(
          (each) => `${each.id}:${each.status}:${each.failure?.reason ?? ""}:${each.fileStatus}`,
        ),
      ...store.linkedFolders.map((each) => `${each.id}:${each.unsupportedFiles}`),
      store.documents.length,
    ].join("|"),
  );
  const wanted = JSON.stringify(scope);
  // biome-ignore lint/correctness/useExhaustiveDependencies: `state` is when to ask again.
  useEffect(() => {
    let current = true;
    core
      .listUnreadFiles(JSON.parse(wanted) as SearchScope | null)
      .then((found) => {
        if (current) setFiles(found);
      })
      .catch(() => {
        // Nothing to say when it can't be asked: no line is better than a wrong one.
        if (current) setFiles([]);
      });
    return () => {
      current = false;
    };
  }, [wanted, state]);
  return files;
}
