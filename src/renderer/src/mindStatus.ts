import { create } from "zustand";
import { useApprovals } from "./approvals";
import { core } from "./core";

/**
 * The Minds with an Answer being written, and those with an Answer waiting
 * for the User to allow its Question to go to the model's service, by the
 * Answer's ID, from the core's Answer events.
 */
const useWritingAnswers = create<{
  minds: Readonly<Record<string, string>>;
  consenting: Readonly<Record<string, string>>;
}>(() => ({ minds: {}, consenting: {} }));

const setWriting = (answerId: string, mindId: string | null) =>
  useWritingAnswers.setState((state) => {
    const minds = { ...state.minds };
    const consenting = { ...state.consenting };
    delete consenting[answerId];
    if (mindId === null) delete minds[answerId];
    else minds[answerId] = mindId;
    return { minds, consenting };
  });

core.on("answer.started", ({ answerId, mindId }) => setWriting(answerId, mindId));
core.on("answer.finished", ({ answerId }) => setWriting(answerId, null));
core.on("answer.failed", ({ answerId }) => setWriting(answerId, null));
core.on("answer.phase", ({ answerId, mindId, phase }) =>
  useWritingAnswers.setState((state) => {
    const consenting = { ...state.consenting };
    if (phase === "waiting-for-consent") consenting[answerId] = mindId;
    else delete consenting[answerId];
    return { consenting };
  }),
);

/**
 * What a Mind is busy with, for its tab and its sidebar row: an Answer
 * waiting for the User, to approve a call or to allow its Question to be
 * sent (it needs the User, so it comes first), an Answer being written, or
 * nothing.
 */
export type MindStatus = "waiting-for-approval" | "writing" | null;

export function useMindStatus(mindId: string): MindStatus {
  const approving = useApprovals((state) =>
    Object.values(state.waiting).some((request) => request.mindId === mindId),
  );
  const consenting = useWritingAnswers((state) => Object.values(state.consenting).includes(mindId));
  const writing = useWritingAnswers((state) => Object.values(state.minds).includes(mindId));
  return approving || consenting ? "waiting-for-approval" : writing ? "writing" : null;
}
