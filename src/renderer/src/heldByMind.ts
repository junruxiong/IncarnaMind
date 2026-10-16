import { create } from "zustand";

/**
 * What each Mind open in the window offers commands and menus, by Mind id:
 * `hold` until the function it returns is called. A Mind open in two panes is
 * held by both, the latest shown.
 */
export function heldByMind<T>() {
  const use = create<Record<string, T>>(() => ({}));
  const holders = new Map<string, T[]>();
  const hold = (mindId: string, entry: T): (() => void) => {
    holders.set(mindId, [...(holders.get(mindId) ?? []), entry]);
    use.setState({ [mindId]: entry });
    return () => {
      const left = (holders.get(mindId) ?? []).filter((each) => each !== entry);
      const latest = left.at(-1);
      if (latest !== undefined) {
        holders.set(mindId, left);
        use.setState({ [mindId]: latest });
        return;
      }
      holders.delete(mindId);
      use.setState((state) => {
        const { [mindId]: _gone, ...rest } = state;
        return rest;
      }, true);
    };
  };
  return { use, hold };
}
