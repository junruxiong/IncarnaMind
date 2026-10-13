import { create } from "zustand";
import type {
  AnswerPhase,
  AskResult,
  ChatModelGroup,
  ChatReadiness,
  ConversationAskResult,
  SkillAvailability,
} from "../../core/api";
import { core } from "./core";
import { errorMessage } from "./errors";
import { translateNow } from "./i18n";
import { useAppStore } from "./store";

/** Why asking a Question didn't go ahead, shown with it until the next try. */
export type AskBlock =
  /** Questions can't be asked yet: the Question explains why. */
  | { kind: "not-ready"; readiness: Extract<ChatReadiness, { ready: false }> }
  /** The Question forces a Skill that is off or gone: the Question says which. */
  | { kind: "skill-unavailable"; skill: string; state: Exclude<SkillAvailability, "enabled"> }
  | { kind: "error"; message: string };

interface AnswersState {
  /**
   * By Question Block ID, by `composerKey` for a Question asked from a Mind's
   * composer, or by the id of an Answer in the assistant that couldn't be written again.
   */
  blocked: Readonly<Record<string, AskBlock>>;
  /** Answers being written, by Block ID, from the core's Answer events. */
  writing: ReadonlySet<string>;
  /** What each Answer being written is doing, by Block ID, from "answer.phase" events. */
  phases: Readonly<Record<string, AnswerPhase>>;
  /** The models a Question's picker offers; null until first asked for. */
  models: ChatModelGroup[] | null;

  /** Asks a Question; an Answer the User edited comes back as changes over their version. */
  ask(mindId: string, questionId: string): Promise<AskResult | null>;
  /** Writes an Answer again; one the User edited comes back as changes over their version. */
  regenerate(mindId: string, answerId: string, questionId: string): Promise<void>;
  stop(mindId: string, answerId: string): void;
  dismiss(questionId: string): void;
  /** Loads the picker's models, unless they are loaded already. */
  loadModels(): void;
}

/**
 * Keeps why asking didn't go ahead, under `key` (a Question's Block ID,
 * `composerKey`, or an Answer's id in the assistant), or clears it.
 */
export function settleAsk(
  key: string,
  result: AskResult | ConversationAskResult | null,
  error?: unknown,
): void {
  useAnswers.setState((state) => {
    const blocked = { ...state.blocked };
    delete blocked[key];
    if (error !== undefined)
      blocked[key] = { kind: "error", message: errorMessage(error, translateNow) };
    else if (result && !result.asked) {
      blocked[key] =
        result.reason === "not-ready"
          ? { kind: "not-ready", readiness: result.readiness }
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
  phases: {},
  models: null,

  async ask(mindId, questionId) {
    try {
      const result = await core.askQuestion({ mindId, questionId });
      settleAsk(questionId, result);
      return result;
    } catch (error) {
      settleAsk(questionId, null, error);
      return null;
    }
  },

  async regenerate(mindId, answerId, questionId) {
    try {
      settleAsk(questionId, await core.regenerateAnswer({ mindId, answerId }));
    } catch (error) {
      // E.g. its Question was deleted: there may be no Question to show this, so the app does.
      settleAsk(questionId, null);
      useAppStore.getState().reportError(error);
    }
  },

  stop(mindId, answerId) {
    core.stopAnswer({ mindId, answerId }).catch((error: unknown) => {
      useAppStore.getState().reportError(error);
    });
  },

  dismiss(questionId) {
    settleAsk(questionId, null);
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
    // A new run starts without a phase; a finished one has none.
    const { [answerId]: _ended, ...phases } = state.phases;
    return { writing: next, phases };
  });

core.on("answer.phase", ({ answerId, phase }) =>
  useAnswers.setState((state) => ({ phases: { ...state.phases, [answerId]: phase } })),
);

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
