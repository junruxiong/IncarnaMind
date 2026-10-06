import type { CoreEventListener, CoreEventName, CoreEvents, Unsubscribe } from "./api";

/** Receives every event the core emits; the desktop host forwards them to the renderer. */
export type AnyEventListener = <E extends CoreEventName>(event: E, payload: CoreEvents[E]) => void;

/**
 * A small typed event hub for the core. A listener that throws doesn't stop the
 * others: the error is reported and the next listener still runs.
 */
export function createEventHub(
  reportError: (error: unknown) => void = (error) => console.error(error),
) {
  const listeners = new Map<CoreEventName, Set<(payload: never) => void>>();
  const anyListeners = new Set<AnyEventListener>();

  const call = (fn: () => void) => {
    try {
      fn();
    } catch (error) {
      reportError(error);
    }
  };

  return {
    on<E extends CoreEventName>(event: E, listener: CoreEventListener<E>): Unsubscribe {
      let set = listeners.get(event);
      if (!set) {
        set = new Set();
        listeners.set(event, set);
      }
      const entry = listener as (payload: never) => void;
      set.add(entry);
      return () => {
        set.delete(entry);
      };
    },

    onAny(listener: AnyEventListener): Unsubscribe {
      anyListeners.add(listener);
      return () => {
        anyListeners.delete(listener);
      };
    },

    emit<E extends CoreEventName>(event: E, payload: CoreEvents[E]): void {
      for (const listener of listeners.get(event) ?? []) {
        call(() => (listener as CoreEventListener<E>)(payload));
      }
      for (const listener of anyListeners) call(() => listener(event, payload));
    },

    clear(): void {
      listeners.clear();
      anyListeners.clear();
    },
  };
}
