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
