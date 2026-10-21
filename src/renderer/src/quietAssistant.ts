import { create } from "zustand";
import { useComposer } from "./composer";
import { useConversations } from "./conversations";
import { useFirstOpen } from "./firstOpen";
import { useAppStore } from "./store";

const useRevealed = create<{ revealed: boolean }>(() => ({ revealed: false }));

// The pill and ⌘J ask for the panel by asking for the composer.
useComposer.subscribe((state) => {
  if (state.focusWanted) useRevealed.setState({ revealed: true });
});

export const revealAssistant = (): void => useRevealed.setState({ revealed: true });

export function useQuietAssistant(): boolean {
  const firstOpen = useFirstOpen();
  const asked = useConversations((state) => state.list.length > 0 || state.openId !== null);
  const askedOwn = useAppStore((state) => state.settings?.device.gettingStarted.askedOwn === true);
  const revealed = useRevealed((state) => state.revealed);
  return firstOpen && !revealed && !asked && !askedOwn;
}
