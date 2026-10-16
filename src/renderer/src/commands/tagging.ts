import { useAppStore } from "../store";
import { stoppedPlaces, stopTagging, taggingPlaces, tagWithAi } from "../taggingStop";
import type { Command } from "./registry";

const store = () => useAppStore.getState();

export const taggingCommands: readonly Command[] = [
  {
    id: "tagging.stop",
    label: "tagging.stop.command",
    group: "main",
    when: () => taggingPlaces(store().documents, store().library).length > 0,
    reversible: true,
    run: () => stopTagging(),
  },
  {
    id: "tagging.tagNotInFolder",
    label: "tagging.tagNotInFolder",
    group: "main",
    when: () => stoppedPlaces(store().library).includes(null),
    reversible: true,
    run: () => tagWithAi([null]),
  },
];
