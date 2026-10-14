import { create } from "zustand";

/** Where feedback goes until there is a mail address: the issue tracker's new-issue page, which is public. */
export const FEEDBACK_URL = "https://github.com/IncarnaMind/IncarnaMind/issues/new/choose";

interface FeedbackState {
  open: boolean;
  show(): void;
  close(): void;
}

/** Help › Send feedback: whether its dialog is open. */
export const useFeedback = create<FeedbackState>((set) => ({
  open: false,
  show: () => set({ open: true }),
  close: () => set({ open: false }),
}));
