import { create } from "zustand";
import type { AskResult, ChatModelGroup, ChatReadiness, SkillAvailability } from "../../core/api";
import { core } from "./core";
import { errorMessage } from "./errors";
import { useAppStore } from "./store";

/** Why asking a Question didn't go ahead, shown with it until the next try. */
export type AskBlock =
  /** Questions can't be asked yet: the Question explains why. */
  | { kind: "not-ready"; readiness: Extract<ChatReadiness, { ready: false }> }
  /** The User edited the Answer: the Answer asks before replacing it. */
  | { kind: "edited"; answerId: string }
  /** The Question forces a Skill that is off or gone: the Question says which. */
  | { kind: "skill-unavailable"; skill: string; state: Exclude<SkillAvailability, "enabled"> }
  | { kind: "error"; message: string };

interface AnswersState {
  /** By Question Block ID. */
  blocked: Readonly<Record<string, AskBlock>>;
  /** Answers being written, by Block ID, from the core's Answer events. */
  writing: ReadonlySet<string>;
  /** The models a Question's picker offers; null until first asked for. */
  models: ChatModelGroup[] | null;

  /** Asks a Question; with `discardEdits`, replaces an Answer the User edited. */
  ask(mindId: string, questionId: string, discardEdits?: boolean): Promise<AskResult | null>;
  /** Writes an Answer again; with `discardEdits`, even if the User edited it. */
  regenerate(
    mindId: string,
    answerId: string,
    questionId: string,
    discardEdits?: boolean,
  ): Promise<void>;
  stop(mindId: string, answerId: string): void;
  dismiss(questionId: string): void;
  /** Loads the picker's models, unless they are loaded already. */
  loadModels(): void;
}

function settle(questionId: string, result: AskResult | null, error?: unknown): void {
  useAnswers.setState((state) => {
    const blocked = { ...state.blocked };
    delete blocked[questionId];
    if (error !== undefined) blocked[questionId] = { kind: "error", message: errorMessage(error) };
    else if (result && !result.asked) {
      blocked[questionId] =
        result.reason === "not-ready"
          ? { kind: "not-ready", readiness: result.readiness }
          : result.reason === "edited"
            ? { kind: "edited", answerId: result.answerId }
            : {
                kind: "skill-unavailable",
                skill: result.skill,
                state: result.state === "disabled" ? "disabled" : "removed",
              };
    }
    return { blocked };
  });
}

/** Questions being asked and Answers being written, in this window. */
export const useAnswers = create<AnswersState>()((set, get) => ({
  blocked: {},
  writing: new Set(),
  models: null,

  async ask(mindId, questionId, discardEdits) {
    try {
      const result = await core.askQuestion({ mindId, questionId, discardEdits });
      settle(questionId, result);
      return result;
    } catch (error) {
      settle(questionId, null, error);
      return null;
    }
  },

  async regenerate(mindId, answerId, questionId, discardEdits) {
    try {
      settle(questionId, await core.regenerateAnswer({ mindId, answerId, discardEdits }));
    } catch (error) {
      // E.g. its Question was deleted: there may be no Question to show this, so the app does.
      settle(questionId, null);
      useAppStore.getState().reportError(error);
    }
  },

  stop(mindId, answerId) {
    core.stopAnswer({ mindId, answerId }).catch((error: unknown) => {
      useAppStore.getState().reportError(error);
    });
  },

  dismiss(questionId) {
    settle(questionId, null);
  },

  loadModels() {
    if (get().models) return;
    set({ models: [] });
    core.listChatModels().then(
      (models) => set({ models }),
      () => set({ models: null }),
    );
  },
}));

const setWriting = (answerId: string, writing: boolean) =>
  useAnswers.setState((state) => {
    const next = new Set(state.writing);
    if (writing) next.add(answerId);
    else next.delete(answerId);
    return { writing: next };
  });

core.on("answer.started", ({ answerId }) => setWriting(answerId, true));
core.on("answer.finished", ({ answerId }) => setWriting(answerId, false));
core.on("answer.failed", ({ answerId }) => setWriting(answerId, false));

core.on("chatReadiness.changed", (readiness) =>
  useAnswers.setState((state) => ({
    // Saving or removing a provider changes what the picker can offer: list them again next time.
    models: null,
    // Once Questions can be asked, "set up a model first" no longer applies.
    blocked: readiness.ready
      ? Object.fromEntries(
          Object.entries(state.blocked).filter(([, block]) => block.kind !== "not-ready"),
        )
      : state.blocked,
  })),
);
