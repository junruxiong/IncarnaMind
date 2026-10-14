import type { ReactNode } from "react";
import type { MessageKey } from "../../shared/i18n";
import type { ObjectKind, ObjectRef, Translate } from "./objects";

/** An object as ⌘K finds it by name. */
export interface FoundObject {
  ref: ObjectRef;
  name: string;
  /** What it is, and where: "Mind · Acme acquisition". */
  meta: string;
  icon(className: string): ReactNode;
}

/**
 * A kind of object ⌘K finds by name. Each kind registers itself, so a new one
 * is found without the palette changing. Kinds that share a group are ranked
 * together under its heading, in the order the first of them registered.
 */
export interface FindableKind {
  kind: ObjectKind;
  group: { id: string; title: MessageKey };
  /** Every object of the kind there is now. */
  objects(t: Translate): readonly FoundObject[];
  /** Calls `changed` whenever `objects` would give something else; returns how to stop. */
  subscribe(changed: () => void): () => void;
  open(ref: ObjectRef, how: { inTab: boolean }): void | Promise<unknown>;
}

const kinds: FindableKind[] = [];
let version = 0;
const listeners = new Set<() => void>();
const unsubscribes = new Map<FindableKind, () => void>();

function watch(kind: FindableKind) {
  unsubscribes.set(
    kind,
    kind.subscribe(() => {
      version++;
      for (const listener of listeners) listener();
    }),
  );
}

export const findable = {
  register(kind: FindableKind): () => void {
    if (kinds.some((each) => each.kind === kind.kind)) {
      throw new Error(`Two findable kinds are called ${kind.kind}.`);
    }
    kinds.push(kind);
    version++;
    if (listeners.size > 0) watch(kind);
    return () => {
      kinds.splice(kinds.indexOf(kind), 1);
      unsubscribes.get(kind)?.();
      unsubscribes.delete(kind);
      version++;
    };
  },

  all: (): readonly FindableKind[] => kinds,

  of: (kind: ObjectKind): FindableKind | undefined => kinds.find((each) => each.kind === kind),

  /** For `useSyncExternalStore`: tells `listener` when any kind's objects change. */
  subscribe(listener: () => void): () => void {
    if (listeners.size === 0) for (const kind of kinds) watch(kind);
    listeners.add(listener);
    return () => {
      listeners.delete(listener);
      if (listeners.size > 0) return;
      for (const stop of unsubscribes.values()) stop();
      unsubscribes.clear();
    };
  },

  /** Changes whenever any kind's objects may have. */
  version: () => version,
};
