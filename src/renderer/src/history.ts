import { create } from "zustand";

/** History (#113): the Mind whose versions fill the window, or none. */
interface HistoryState {
  mindId: string | null;
  open(mindId: string): void;
  close(): void;
}

export const useHistory = create<HistoryState>()((set) => ({
  mindId: null,
  open: (mindId) => set({ mindId }),
  close: () => set({ mindId: null }),
}));
