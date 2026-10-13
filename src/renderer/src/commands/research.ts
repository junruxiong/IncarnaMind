import { useComposer } from "../composer";
import { focusComposer } from "./assistant";
import type { Command } from "./registry";

export const researchCommands: readonly Command[] = [
  {
    // Research (#130): the composer takes a topic, for the Mind the assistant reads.
    id: "research.start",
    label: "research.command",
    keywords: ["research.command.keywords"],
    group: "main",
    // It only marks the composer; the Task starts from the plan card.
    reversible: true,
    enabled: () => (useComposer.getState().seat ? true : { reason: "research.needsMind" }),
    run: () => {
      const { seat, update } = useComposer.getState();
      if (!seat) return;
      update(seat.mindId, { research: true, skill: null });
      focusComposer();
    },
  },
];
