import { AsyncLocalStorage } from "node:async_hooks";

/**
 * Which AI flow a model request or a local step works for (#1624): the User's action it serves
 * (`flow`: a Question, Edit with AI, Check this Mind…) and, inside it, the step (`step`: the
 * search rewrite, the rewrite an edit makes, the check of what was written…). Set where each
 * flow and step starts, through every await below it, as `currentTask` is, so the timing record
 * (./timings) is tagged with no id passed through the code. A request made outside any flow is
 * recorded under its data flow alone, so a new flow still shows in the table.
 */
export interface FlowTag {
  flow: string;
  step: string | null;
}

const current = new AsyncLocalStorage<FlowTag>();

/** Runs `work` as the flow `flow`, unless it already runs within one: a flow inside a flow is its step. */
export function inFlow<T>(flow: string, work: () => T): T {
  const outer = current.getStore();
  return current.run(outer ? { flow: outer.flow, step: flow } : { flow, step: null }, work);
}

/** Runs `work` as the step `step` of the flow it runs within; outside any flow, as a flow of its own. */
export function inStep<T>(step: string, work: () => T): T {
  const outer = current.getStore();
  return current.run(outer ? { flow: outer.flow, step } : { flow: step, step: null }, work);
}

export const flowNow = (): FlowTag | null => current.getStore() ?? null;
