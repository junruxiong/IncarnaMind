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
};

export const useModelSetup = create<{ waiting: WaitingQuestion | null }>(() => ({
  waiting: null,
}));

/** Keeps a Question that couldn't be asked for want of a model, for the setup card to wait on. */
export const waitForModel = (question: WaitingQuestion): void =>
  useModelSetup.setState({ waiting: question });

/** "Set up later": the Question's words go back to the composer they were typed in. */
export function giveUpWaiting(): void {
  const { waiting } = useModelSetup.getState();
  if (!waiting) return;
  useModelSetup.setState({ waiting: null });
  const key = waiting.mindId ?? NO_MIND;
  const draft = useComposer.getState().draft(key);
  if (draft.text === "") useComposer.getState().update(key, { text: waiting.text });
}
