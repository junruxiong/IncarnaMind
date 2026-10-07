import { createContext, useContext } from "react";

/** The Mind an editor shows, for its node views (Questions and Answers ask the core about it). */
export const MindIdContext = createContext<string | null>(null);

export function useMindId(): string {
  const mindId = useContext(MindIdContext);
  if (!mindId) throw new Error("A node view is drawn outside a Mind's editor.");
  return mindId;
}
