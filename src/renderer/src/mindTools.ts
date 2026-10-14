import { create } from "zustand";

/**
 * Which Mind shows its outline, or its find bar. Commands set them; the Mind's
 * editor draws them. `findRequests` counts the ⌘F presses, so a second press
 * on an open find bar puts the cursor back in its field.
 */
interface MindTools {
  outlineMindId: string | null;
  findMindId: string | null;
  findRequests: number;
  showOutline(mindId: string): void;
  hideOutline(): void;
  showFind(mindId: string): void;
  hideFind(): void;
}

export const useMindTools = create<MindTools>((set) => ({
  outlineMindId: null,
  findMindId: null,
  findRequests: 0,
  showOutline: (mindId) => set({ outlineMindId: mindId }),
  hideOutline: () => set({ outlineMindId: null }),
  showFind: (mindId) =>
    set((state) => ({ findMindId: mindId, findRequests: state.findRequests + 1 })),
  hideFind: () => set({ findMindId: null }),
}));
