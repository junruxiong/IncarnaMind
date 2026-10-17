import { allExamples, useAppStore } from "../store";
import { type Command, fromMenu } from "./registry";

const examplesMade = () =>
  allExamples(useAppStore.getState()).some(
    (each) => each.mindId !== null || each.linkedFolderId !== null,
  );

const available = () => useAppStore.getState().examples?.available === true;

export const exampleCommands: readonly Command[] = [
  {
    id: "examples.remove",
    label: "examples.banner.remove",
    group: "danger",
    destructive: true,
    reversible: true,
    when: examplesMade,
    run: () => useAppStore.getState().removeExamples(),
  },
  {
    ...fromMenu("examples.show"),
    group: "main",
    reversible: true,
    when: () => available() && !examplesMade(),
    run: () => useAppStore.getState().showExamples(),
  },
];
