/**
 * Make this a Skill, the window's side: asking the core to draft a Skill from
 * what the User just did in a Mind, and holding what the Mind's card shows
 * (see `MakeSkillCard`): the AI at work, a problem, the draft to edit, or the
 * Skill saved. Each Mind has its own, so another Mind or a tab change leaves it as it is.
 */
import { create } from "zustand";
import type { ChatReadiness, SkillDraft } from "../../core/api";
import { core } from "./core";
import { errorMessage } from "./errors";
import { useAppStore } from "./store";

export type MakeSkillState =
  | { phase: "drafting"; requestId: string }
  | { phase: "editing"; draft: SkillDraft }
  | { phase: "empty" }
  | { phase: "not-ready"; readiness: Extract<ChatReadiness, { ready: false }> }
  | { phase: "failed"; message: string }
  | { phase: "saved"; name: string };

interface MakeSkillStore {
  /** By Mind. */
  byMind: Readonly<Record<string, MakeSkillState>>;
}

export const useMakeSkill = create<MakeSkillStore>()(() => ({ byMind: {} }));

const set = (mindId: string, state: MakeSkillState | null) =>
  useMakeSkill.setState((store) => {
    const next = { ...store.byMind };
    if (state === null) delete next[mindId];
    else next[mindId] = state;
    return { byMind: next };
  });

/** Opens the Mind and has the AI draft a Skill from it; the card in the Mind shows how it goes. */
export async function startSkillFromMind(mindId: string): Promise<void> {
  const requestId = crypto.randomUUID();
  set(mindId, { phase: "drafting", requestId });
  void useAppStore.getState().openMind(mindId);
  let state: MakeSkillState;
  try {
    const result = await core.draftSkillFromMind({ requestId, mindId });
    switch (result.kind) {
      case "drafted":
        state = { phase: "editing", draft: result.draft };
        break;
      case "empty":
        state = { phase: "empty" };
        break;
      case "not-ready":
        state = { phase: "not-ready", readiness: result.readiness };
        break;
      case "failed":
        state = { phase: "failed", message: result.error.message };
        break;
      case "cancelled":
        state = { phase: "failed", message: "" };
        break;
    }
  } catch (error) {
    state = { phase: "failed", message: errorMessage(error) };
  }
  const current = useMakeSkill.getState().byMind[mindId];
  // Stopped, or started again, meanwhile: this answer is no longer wanted.
  if (current?.phase !== "drafting" || current.requestId !== requestId) return;
  set(mindId, state.phase === "failed" && state.message === "" ? null : state);
}

/** Stops the AI drafting; the card goes away. */
export function stopSkillDraft(mindId: string): void {
  const current = useMakeSkill.getState().byMind[mindId];
  if (current?.phase !== "drafting") return;
  set(mindId, null);
  core.cancelSkillDraft(current.requestId).catch(useAppStore.getState().reportError);
}

export const skillSaved = (mindId: string, name: string) => set(mindId, { phase: "saved", name });

export const closeMakeSkill = (mindId: string) => set(mindId, null);
