import { useAppStore } from "../store";
import type { Command } from "./registry";

export const libraryCommands: readonly Command[] = [
  {
    id: "library.open",
    label: "library.open",
    keywords: ["library.open.keywords"],
    group: "open",
    top: 40,
    reversible: true,
    run: () => useAppStore.getState().openLibrary(),
  },
];
