/**
 * Where the assistant is writing each Mind's new Blocks right now (#202), as
 * the core says ("mind.writing"), kept for an editor that opens the Mind
 * while it is being written: a Mind the assistant makes opens just after its
 * first Block goes in.
 */
import type { MindWriting } from "../../core/api";
import { core } from "./core";

const places = new Map<string, { after: string | null }>();
const listeners = new Set<(writing: MindWriting) => void>();

core.on("mind.writing", (writing) => {
  if (writing.writing) places.set(writing.mindId, { after: writing.after });
  else places.delete(writing.mindId);
  for (const listener of listeners) listener(writing);
});

/** Where the next Block goes in a Mind, while the assistant writes it; null when it isn't. */
export const writingPlaceOf = (mindId: string): { after: string | null } | null =>
  places.get(mindId) ?? null;

/** Hears each change of where the assistant writes; returns what stops hearing. */
export function onMindWriting(listener: (writing: MindWriting) => void): () => void {
  listeners.add(listener);
  return () => listeners.delete(listener);
}
