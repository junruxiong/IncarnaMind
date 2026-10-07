import { create } from "zustand";
import type { ApprovalDecision, ApprovalRequest } from "../../core/api";
import { core } from "./core";
import { useAppStore } from "./store";

interface ApprovalsState {
  /**
   * Tool calls waiting for the User's approval, by request id, in every
   * window: from the core's events, and those raised before this window
   * listened. A decision in any window takes the request away in all.
   */
  waiting: Readonly<Record<string, ApprovalRequest>>;
  respond(requestId: string, decision: ApprovalDecision): void;
}

export const useApprovals = create<ApprovalsState>()(() => ({
  waiting: {},
  respond(requestId, decision) {
    // The card goes at once; the core's "approval.resolved" confirms it in every window.
    remove(requestId);
    core.respondToApproval(requestId, decision).catch((error: unknown) => {
      useAppStore.getState().reportError(error);
    });
  },
}));

const add = (requests: readonly ApprovalRequest[]) =>
  useApprovals.setState((state) => ({
    waiting: {
      ...state.waiting,
      ...Object.fromEntries(requests.map((request) => [request.requestId, request])),
    },
  }));

function remove(requestId: string) {
  useApprovals.setState((state) => {
    if (!(requestId in state.waiting)) return state;
    const waiting = { ...state.waiting };
    delete waiting[requestId];
    return { waiting };
  });
}

core.on("approval.requested", (request) => add([request]));
core.on("approval.resolved", ({ requestId }) => remove(requestId));
core.listApprovalRequests().then(add, () => undefined);

/** The requests waiting for one Answer, oldest first. */
export const waitingFor = (
  waiting: Readonly<Record<string, ApprovalRequest>>,
  answerId: string | null,
): ApprovalRequest[] =>
  answerId === null ? [] : Object.values(waiting).filter((each) => each.answerId === answerId);
