import { create } from "zustand";

/**
 * The assistant's height the question waiting in its tray needs to be read whole (#1400), which
 * the bottom dock grows to while it waits; null when no question waits.
 */
export const useQuestionRoom = create<{ fits: number | null }>()(() => ({ fits: null }));
