import { create } from "zustand";

interface Asked {
  mindId: string;
  anchor: HTMLButtonElement | null;
}

/**
 * The Mind whose Style is open, and the one whose Other fonts… list is, wherever
 * each was asked for. `MindStyleHost` (mounted once) shows them.
 */
export const useMindStylePanel = create<{
  open: Asked | null;
  fonts: (Asked & { anchor: HTMLButtonElement }) | null;
  show(mindId: string, anchor: HTMLButtonElement | null): void;
  close(): void;
  pickFont(mindId: string, anchor: HTMLButtonElement): void;
  closeFonts(): void;
}>()((set) => ({
  open: null,
  fonts: null,
  show: (mindId, anchor) => set({ open: { mindId, anchor } }),
  close: () => set({ open: null }),
  pickFont: (mindId, anchor) => set({ fonts: { mindId, anchor } }),
  closeFonts: () => set({ fonts: null }),
}));
