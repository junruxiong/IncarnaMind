import { useEffect, useState } from "react";
import { useComposer } from "./composer";
import { useConversations } from "./conversations";
import { useFirstOpen } from "./firstOpen";
import { useAppStore } from "./store";

export function useQuietAssistant(): boolean {
  const firstOpen = useFirstOpen();
  const asked = useConversations((state) => state.list.length > 0);
  const askedOwn = useAppStore((state) => state.settings?.device.gettingStarted.askedOwn === true);
  const focusWanted = useComposer((state) => state.focusWanted);
  const [revealed, setRevealed] = useState(false);
  useEffect(() => {
    if (focusWanted) setRevealed(true);
  }, [focusWanted]);
  return firstOpen && !revealed && !asked && !askedOwn;
}
