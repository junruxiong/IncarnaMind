import { usePalette } from "../palette";
import type { Command } from "./registry";

export const paletteCommands: readonly Command[] = [
  {
    id: "palette.open",
    label: "command.palette.open",
    shortcut: { keys: "CmdOrCtrl+K" },
    palette: false,
    group: "open",
    reversible: true,
    run: () => usePalette.getState().show(),
  },
];
