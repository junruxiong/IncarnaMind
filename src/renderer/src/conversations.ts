import { create } from "zustand";
import type { Conversation } from "../../core/api";
import { core } from "./core";
import { useAppStore } from "./store";

/**
 * The assistant's conversations, as History lists them (ADR-0017): every one that
 * isn't deleted, newest first, kept as the core has them by its "conversations.changed".
 */
export const useConversations = create<{ list: readonly Conversation[] }>()(() => ({ list: [] }));

const load = () =>
  core.listConversations().then(
    (list) => useConversations.setState({ list }),
    (error: unknown) => useAppStore.getState().reportError(error),
  );

core.on("conversations.changed", () => void load());
void load();

/** A conversation by id, as last listed; undefined when it isn't (deleted, or not loaded yet). */
export const conversationById = (id: string): Conversation | undefined =>
  useConversations.getState().list.find((each) => each.id === id);
