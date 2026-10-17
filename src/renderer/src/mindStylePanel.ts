import { create } from "zustand";

/** The Mind whose Style is open, wherever it was asked for. `MindStyleHost` (mounted once) shows it. */
export const useMindStylePanel = create<{
  open: { mindId: string; anchor: HTMLButtonElement | null } | null;
  show(mindId: string, anchor: HTMLButtonElement | null): void;
  close(): void;
}>()((set) => ({
  open: null,
  show: (mindId, anchor) => set({ open: { mindId, anchor } }),
  close: () => set({ open: null }),
}));
