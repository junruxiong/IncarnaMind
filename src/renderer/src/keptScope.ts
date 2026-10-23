import { useEffect, useState } from "react";
import type * as Y from "yjs";
import type { SearchScope } from "../../core/api";
import { mindScopeOf, observeMindScope } from "../../shared/mindScope";
import { sameScope } from "../../shared/searchScope";
import { useComposer } from "./composer";

/** Keeps the Mind's draft on the scope kept with the Mind, as it is and as it changes, here or elsewhere. */
export function useKeptScope(mindId: string, doc: Y.Doc): void {
  useEffect(() => {
    const mirror = () => {
      const kept = mindScopeOf(doc);
      const drafted = useComposer.getState().draft(mindId).scope;
      if (kept === drafted || (kept && drafted && sameScope(kept, drafted))) return;
      useComposer.getState().update(mindId, { scope: kept });
    };
    mirror();
    return observeMindScope(doc, mirror);
  }, [mindId, doc]);
}

/** The scope kept with the Mind, as it is now. */
export function useMindScope(doc: Y.Doc): SearchScope | null {
  const [scope, setScope] = useState(() => mindScopeOf(doc));
  useEffect(() => {
    setScope(mindScopeOf(doc));
    return observeMindScope(doc, () => setScope(mindScopeOf(doc)));
  }, [doc]);
  return scope;
}
