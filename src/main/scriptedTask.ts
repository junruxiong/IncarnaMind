import { join } from "node:path";
import { fileWrite, type TaskKind } from "../core";

/** The kind of the smoke tests' scripted Task. */
export const SCRIPTED_TASK = "scripted";

interface ScriptedInput {
  folder: string;
  files: string[];
}

function inputOf(input: unknown): ScriptedInput | null {
  if (typeof input !== "object" || input === null) return null;
  const { folder, files } = input as Record<string, unknown>;
  if (typeof folder !== "string" || !Array.isArray(files)) return null;
  return { folder, files: files.filter((each): each is string => typeof each === "string") };
}

/**
 * A Task only a test build has, the way it has the scripted chat model: no
 * Task IncarnaMind ships asks for approval yet, and the smoke tests need one
 * that does (#126). Each file is a call that asks first, then a file written
 * into a test folder as an Effect.
 */
export const scriptedTask: TaskKind = {
  kind: SCRIPTED_TASK,
  describeSteps: (_task, keys) =>
    keys.map((key) => {
      const [step, name] = key.split(":");
      return step === "approve" ? `Asked to write ${name}` : `Wrote ${name}`;
    }),
  async run(task) {
    const input = inputOf(task.input);
    if (!input) return { status: "failed", problem: "unknown" };
    for (const [index, name] of input.files.entries()) {
      const allowed = await task.approve(`approve:${name}`, {
        mindId: task.mindId,
        subject: { kind: "tool", connectorId: "scripted", tool: "write_file" },
        connector: { id: "scripted", name: "Scripted Task" },
        tool: "write_file",
        title: "Write a file",
        input: { name },
        readOnly: false,
        effects: [{ action: "write", scope: { kind: "folder", path: input.folder } }],
        tainted: false,
      });
      if (task.signal.aborted) return { status: "stopped" };
      if (allowed) {
        const write = await fileWrite(join(input.folder, name), `Written by the scripted Task.\n`);
        await task.effect(`write:${name}`, write.effect, write.perform);
      }
      task.progress({ done: index + 1, total: input.files.length });
    }
    return { status: "done" };
  },
};
