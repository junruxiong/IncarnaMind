import { useEffect } from "react";
import { mindName } from "../../shared/naming";
import { files } from "./core";
import { currentMindIdOf } from "./layoutSlice";
import { useAppStore } from "./store";

export function useExportMenu(): void {
  const name = useAppStore((state) => {
    const id = currentMindIdOf(state.layout);
    const mind = state.minds.find((each) => each.id === id);
    return mind ? mindName(mind) : null;
  });
  useEffect(() => files.setExportMenu(name), [name]);
}

export function useExamplesMenu(): void {
  const removed = useAppStore((state) => {
    const made = [state.examples, ...state.otherExamples].some(
      (each) => each !== null && (each.mindId !== null || each.linkedFolderId !== null),
    );
    return state.examples?.available === true && !made;
  });
  useEffect(() => files.setExamplesRemoved(removed), [removed]);
}
