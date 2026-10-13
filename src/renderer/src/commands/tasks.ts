import { chipOf } from "../taskGroups";
import { useTasks } from "../tasks";
import type { Command } from "./registry";

export const taskCommands: readonly Command[] = [
  {
    // The Tasks chip's popover (#126), offered while the chip shows.
    id: "tasks.show",
    label: "command.showTasks",
    keywords: ["command.showTasks.keywords"],
    group: "open",
    reversible: true,
    when: () => chipOf(Object.values(useTasks.getState().tasks)) !== null,
    run: () => {
      document.querySelector<HTMLElement>('[data-testid="tasks-chip"]')?.click();
    },
  },
];
