import { create } from "zustand";
import { useApprovals } from "./approvals";
import { core } from "./core";

/** The Minds with an Answer being written, by the Answer's ID, from the core's Answer events. */
const useWritingAnswers = create<{ minds: Readonly<Record<string, string>> }>(() => ({
  minds: {},
}));

const setWriting = (answerId: string, mindId: string | null) =>
  useWritingAnswers.setState((state) => {
    const minds = { ...state.minds };
    if (mindId === null) delete minds[answerId];
    else minds[answerId] = mindId;
    return { minds };
  });

core.on("answer.started", ({ answerId, mindId }) => setWriting(answerId, mindId));
core.on("answer.finished", ({ answerId }) => setWriting(answerId, null));
core.on("answer.failed", ({ answerId }) => setWriting(answerId, null));

/**
 * What a Mind is busy with, for its tab and its sidebar row: an Answer
 * waiting for the User's approval (it needs the User, so it comes first), an
 * Answer being written, or nothing.
 */
export type MindStatus = "waiting-for-approval" | "writing" | null;

export function useMindStatus(mindId: string): MindStatus {
  const waiting = useApprovals((state) =>
    Object.values(state.waiting).some((request) => request.mindId === mindId),
  );
  const writing = useWritingAnswers((state) => Object.values(state.minds).includes(mindId));
  return waiting ? "waiting-for-approval" : writing ? "writing" : null;
}
