/**
 * What the commands of one Run changed, kept so one Undo takes all of it back and one Redo
 * does it again (#1110), as the User's own changes are: in this session only.
 */
import { NotFoundError } from "../../errors";

/** One command's change: how to take it back, and how to make it again once taken back. */
export interface CommandStep {
  undo(): Promise<void>;
  redo(): Promise<void>;
}

export type CommandSteps = ReturnType<typeof createCommandSteps>;

export function createCommandSteps() {
  const runs = new Map<string, { steps: CommandStep[]; undone: boolean }>();

  const runOf = (runId: string) => {
    const run = runs.get(runId);
    if (!run) throw new NotFoundError({ key: "error.commands.undoGone" });
    return run;
  };

  return {
    add(runId: string, step: CommandStep): void {
      const run = runs.get(runId) ?? { steps: [], undone: false };
      run.steps.push(step);
      runs.set(runId, run);
    },

    /** Takes back every change the Run's commands made, the last first. */
    async undo(runId: string): Promise<void> {
      const run = runOf(runId);
      if (run.undone) return;
      for (const step of [...run.steps].reverse()) await step.undo();
      run.undone = true;
    },

    /** Makes them again, in the order they were made. */
    async redo(runId: string): Promise<void> {
      const run = runOf(runId);
      if (!run.undone) return;
      for (const step of run.steps) await step.redo();
      run.undone = false;
    },
  };
}
