import { create } from "zustand";

/** Where feedback goes; the User sends it from their own mail app. */
export const FEEDBACK_ADDRESS = "feedback@incarnamind.app";
export const ISSUES_URL = "https://github.com/IncarnaMind/IncarnaMind/issues/new";

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

/** The draft the User's mail app opens with: their text, as typed, and nothing else. */
export const mailDraft = (subject: string, body: string) =>
  `mailto:${FEEDBACK_ADDRESS}?subject=${encodeURIComponent(subject)}&body=${encodeURIComponent(body)}`;
