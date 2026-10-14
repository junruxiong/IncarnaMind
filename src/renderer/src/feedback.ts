import { create } from "zustand";
import type { Language } from "../../core/language";
import type { TitleBar } from "../../shared/titleBar";

const FORMS: Record<Language, string> = {
  en: "https://tally.so/r/bZVZjo",
  "zh-CN": "https://tally.so/r/M1K1EX",
};

export const ISSUES_URL = "https://github.com/IncarnaMind/IncarnaMind/issues/new/choose";

const OS: Record<TitleBar, string> = { inset: "mac", overlay: "windows", native: "linux" };

/** The feedback form for the interface language, with the app's version and system filled in. */
export const feedbackUrl = (language: Language, version: string, titleBar: TitleBar): string =>
  `${FORMS[language]}?${new URLSearchParams({ version, os: OS[titleBar] })}`;

interface FeedbackState {
  open: boolean;
  show(): void;
  close(): void;
}

export const useFeedback = create<FeedbackState>((set) => ({
  open: false,
  show: () => set({ open: true }),
  close: () => set({ open: false }),
}));
