import { assistantTabOf } from "../../layoutSlice";
import { useAppStore } from "../../store";
import { useGroupsMerged } from "../../useWindowWidth";

/**
 * The Mind the assistant reads: the one in focus, whose composer asks in it,
 * and where Add to note puts an Answer. Null with no Mind in focus.
 */
export function useReadMindId(): string | null {
  const merged = useGroupsMerged();
  return useAppStore((state) => {
    const item = assistantTabOf(state.layout, merged)?.item;
    return item?.kind === "mind" ? item.mindId : null;
  });
}
