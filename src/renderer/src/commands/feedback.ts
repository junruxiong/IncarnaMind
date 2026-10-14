import { useFeedback } from "../feedback";
import { type Command, fromMenu } from "./registry";

export const feedbackCommands: readonly Command[] = [
  {
    ...fromMenu("help.feedback"),
    group: "main",
    reversible: true,
    run: () => useFeedback.getState().show(),
  },
];
