/**
 * The first own Question with no model set (#103): it waits in the assistant's conversation while
 * the composer becomes the setup card, and is asked as it was once a model is chosen, without
 * being typed again (`askWaitingQuestion`, in ./conversations). "Set up later" gives its words
 * back to the composer.
 */
import { create } from "zustand";
import type { AskInConversationInput } from "../../core/api";
import { NO_MIND, useComposer } from "./composer";

export type WaitingQuestion = Omit<AskInConversationInput, "conversationId"> & {
  mindId: string | null;
  /** The conversation it was typed in (null: a new chat), where it is asked once a model is set. */
  conversationId: string | null;
};

export const useModelSetup = create<{ waiting: WaitingQuestion | null; wanted: boolean }>(() => ({
  waiting: null,
  wanted: false,
}));

/** The setup card with no Question waiting, which the first open's steps card asks for (#1027). */
export const openModelSetup = (): void => useModelSetup.setState({ wanted: true });

export const useSettingUp = (): boolean =>
  useModelSetup((state) => state.waiting !== null || state.wanted);

/** Keeps a Question that couldn't be asked for want of a model, for the setup card to wait on. */
export const waitForModel = (question: WaitingQuestion): void =>
  useModelSetup.setState({ waiting: question });

/** "Set up later": the Question's words go back to the composer they were typed in. */
export function giveUpWaiting(): void {
  const { waiting } = useModelSetup.getState();
  useModelSetup.setState({ waiting: null, wanted: false });
  if (!waiting) return;
  const key = waiting.mindId ?? NO_MIND;
  const draft = useComposer.getState().draft(key);
  if (draft.text === "") useComposer.getState().update(key, { text: waiting.text });
}
