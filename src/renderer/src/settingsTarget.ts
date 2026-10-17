import { useEffect, useRef } from "react";
import type { SettingsTarget } from "./settingsPages";
import { useAppStore } from "./store";

/**
 * Scrolls to the setting Settings was opened at, once it is drawn, then forgets it;
 * `onReached` follows, e.g. to focus its button.
 */
export function useSettingsTarget<T extends HTMLElement>(
  targets: readonly SettingsTarget[],
  drawn: boolean,
  onReached?: () => void,
) {
  const ref = useRef<T>(null);
  const at = useAppStore((state) => state.settingsAt);
  useEffect(() => {
    if (!drawn || at === null || !targets.includes(at)) return;
    useAppStore.setState({ settingsAt: null });
    ref.current?.scrollIntoView({ block: "start" });
    onReached?.();
  }, [at, drawn, targets, onReached]);
  return ref;
}
