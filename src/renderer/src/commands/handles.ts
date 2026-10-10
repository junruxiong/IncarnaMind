import { useEffect } from "react";

/** Something a command acts on that lives in a component, there while the component is. */
export interface Handle<T> {
  current: T | null;
}

/** The open Mind's composer. */
export const composerHandle: Handle<{ focus(): void }> = { current: null };

/** The sidebar's show and hide (see `useSidebarControl` in App.tsx). */
export const sidebarHandle: Handle<{ hidden: boolean; toggle(): void }> = { current: null };

/** Lends `value` to commands through `handle` while the calling component is mounted. */
export function useHandle<T>(handle: Handle<T>, value: T): void {
  useEffect(() => {
    handle.current = value;
    return () => {
      if (handle.current === value) handle.current = null;
    };
  }, [handle, value]);
}
