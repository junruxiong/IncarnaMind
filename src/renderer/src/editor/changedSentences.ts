import { useEffect, useState } from "react";
import type * as Y from "yjs";
import { core } from "../core";
import type { Translate } from "../objects";

const IDLE_MS = 300;

/**
 * How many sentences "Check this Mind" would check now, the ones changed since
 * the last check, as the Mind is edited; 0 when nothing changed (it then checks
 * every sentence), null until known. Not counted while a check runs.
 */
export function useChangedSentences(doc: Y.Doc, mindId: string, paused: boolean): number | null {
  const [count, setCount] = useState<number | null>(null);
  useEffect(() => {
    if (paused) return;
    let current = true;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const refresh = () => {
      timer = undefined;
      core.planMindCheck(mindId, "changed", false).then(
        (plan) => {
          if (current) setCount(plan.kind === "ready" ? plan.sentences : 0);
        },
        (failure: unknown) => {
          console.warn(`Couldn't count what changed in Mind ${mindId}.`, failure);
        },
      );
    };
    const later = () => {
      clearTimeout(timer);
      timer = setTimeout(refresh, IDLE_MS);
    };
    refresh();
    doc.on("update", later);
    return () => {
      current = false;
      clearTimeout(timer);
      doc.off("update", later);
    };
  }, [doc, mindId, paused]);
  return count;
}

/** What "Check this Mind" would check: the changed sentences, or every one when nothing changed. */
export function changedTitle(t: Translate, changed: number): string {
  if (changed === 0) return t("mindCheck.changed.none");
  return t(changed === 1 ? "mindCheck.changed.one" : "mindCheck.changed", { count: changed });
}
