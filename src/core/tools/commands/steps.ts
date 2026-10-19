/**
 * What the commands of one Run changed, kept so one Undo takes all of it back and one Redo
 * does it again (#1110), as the User's own changes are: in this session only. Each change
 * remembers whether it is taken back, so one that fails is tried again, alone, the next time.
 */
import type { CommandKept } from "../../api";
import { InvalidInputError, NotFoundError } from "../../errors";

/**
 * One command's change: how to take it back, and how to make it again once taken back. Taking
 * it back gives the Folder or Tag it made and left, when the User filed things in it since.
 */
export interface CommandStep {
  undo(): Promise<CommandKept | null>;
  redo(): Promise<void>;
}

export type CommandSteps = ReturnType<typeof createCommandSteps>;

const reasonOf = (error: unknown) => (error instanceof Error ? error.message : String(error));

export function createCommandSteps() {
  const runs = new Map<string, { step: CommandStep; undone: boolean }[]>();

  const runOf = (runId: string) => {
    const run = runs.get(runId);
    if (!run) throw new NotFoundError({ key: "error.commands.undoGone" });
    return run;
  };

  return {
    add(runId: string, step: CommandStep): void {
      runs.set(runId, [...(runs.get(runId) ?? []), { step, undone: false }]);
    },

    /** Takes back every change of the Run not yet taken back, the last first. */
    async undo(runId: string): Promise<CommandKept[]> {
      const kept: CommandKept[] = [];
      const failed: string[] = [];
      for (const each of [...runOf(runId)].reverse()) {
        if (each.undone) continue;
        try {
          const left = await each.step.undo();
          if (left) kept.push(left);
          each.undone = true;
        } catch (error) {
          failed.push(reasonOf(error));
        }
      }
      if (failed.length > 0) {
        throw new InvalidInputError({
          key: "error.commands.undoPartly",
          values: { reasons: failed.join(" ") },
        });
      }
      return kept;
    },

    /** Makes again every change of the Run taken back, in the order they were made. */
    async redo(runId: string): Promise<void> {
      const failed: string[] = [];
      for (const each of runOf(runId)) {
        if (!each.undone) continue;
        try {
          await each.step.redo();
          each.undone = false;
        } catch (error) {
          failed.push(reasonOf(error));
        }
      }
      if (failed.length > 0) {
        throw new InvalidInputError({
          key: "error.commands.redoPartly",
          values: { reasons: failed.join(" ") },
        });
      }
    },
  };
}
