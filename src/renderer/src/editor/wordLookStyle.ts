import { useEffect, useMemo, useState } from "react";
import type * as Y from "yjs";
import { mindLookOf, observeWordLook, setMindLook, wordLookOf } from "../../../shared/wordLook";
import { useAppStore } from "../store";
import { holdWordLook } from "../wordLooks";
import { useWordFile } from "./WordOrigin";
import { wordLookCss, wordLookScope } from "./wordLookCss";

/** The Mind's Word look as its frame's class and stylesheet; null while it is drawn as a plain Mind. */
export function useWordLookStyle(doc: Y.Doc, mindId: string) {
  const read = () => ({ look: wordLookOf(doc), shown: mindLookOf(doc) });
  const [state, setState] = useState(read);
  // biome-ignore lint/correctness/useExhaustiveDependencies: `read` only reads `doc`.
  useEffect(() => {
    setState(read());
    return observeWordLook(doc, () => setState(read()));
  }, [doc]);
  const { file } = useWordFile(doc);
  const documentId = file?.documentId ?? null;
  const originalId = useAppStore((app) =>
    documentId && app.documents.some((each) => each.id === documentId) ? documentId : null,
  );

  useEffect(() => {
    if (!state.look) return;
    return holdWordLook(mindId, {
      shown: state.shown,
      originalId,
      choose: (look) => setMindLook(doc, look),
    });
  }, [state, mindId, originalId, doc]);

  return useMemo(
    () =>
      state.look && state.shown === "word"
        ? { className: wordLookScope(mindId), css: wordLookCss(mindId, state.look) }
        : null,
    [state, mindId],
  );
}
