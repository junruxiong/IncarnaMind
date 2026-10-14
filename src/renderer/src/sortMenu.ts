import { create } from "zustand";
import type { SortScope } from "./sortChoices";

export interface SortRequest {
  scope: SortScope;
  /** The Library sorts Documents only: no Folders first, and no choice for every Folder. */
  withFolders: boolean;
  /** What the menu opens beside: the ⋯ or the button that asked for it. */
  anchor: HTMLButtonElement | null;
}

/** The place whose Sort by is open, wherever it was asked for. `SortMenu` (mounted once) shows it. */
export const useSortMenu = create<{
  request: SortRequest | null;
  show(request: SortRequest): void;
  hide(): void;
}>()((set) => ({
  request: null,
  show: (request) => set({ request }),
  hide: () => set({ request: null }),
}));
