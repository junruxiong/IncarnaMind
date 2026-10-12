import { create } from "zustand";

/** The Mind whose export dialog is open, wherever Export was pressed: its tab strip, its References. */
export const useExportDialog = create<{
  mindId: string | null;
  open(mindId: string): void;
  close(): void;
}>()((set) => ({
  mindId: null,
  open: (mindId) => set({ mindId }),
  close: () => set({ mindId: null }),
}));
