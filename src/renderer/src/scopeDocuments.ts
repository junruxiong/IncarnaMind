import { useEffect, useState } from "react";
import type { SearchScope } from "../../core/api";
import { core } from "./core";
import { useAppStore } from "./store";

/**
 * Whether a Search scope holds no Document that can be searched now (#748):
 * false for none (every Document), and until the core has answered. Asked
 * again when Documents come or change.
 */
export function useScopeEmpty(scope: SearchScope | null): boolean {
  const [empty, setEmpty] = useState(false);
  const wanted = scope ? JSON.stringify(scope) : null;
  const state = useAppStore((store) =>
    [
      ...store.documents.map((each) => `${each.id}:${each.status}:${each.fileStatus}`),
      // A Folder or Tag deleted, or a Document moved into one, changes what the scope holds.
      ...store.folders.map((each) => each.id),
      ...(store.library?.groups ?? []).map((each) => each.id),
      ...store.tags.map((each) => each.id),
      ...store.documents.map((each) => `${each.id}:${each.tags.length}:${each.folderId}`),
    ].join("|"),
  );
  // biome-ignore lint/correctness/useExhaustiveDependencies: `state` is when to ask again.
  useEffect(() => {
    if (wanted === null) {
      setEmpty(false);
      return;
    }
    let current = true;
    core
      .countScopeDocuments(JSON.parse(wanted) as SearchScope)
      .then((count) => {
        if (current) setEmpty(count === 0);
      })
      .catch(() => {
        if (current) setEmpty(false);
      });
    return () => {
      current = false;
    };
  }, [wanted, state]);
  return empty;
}
