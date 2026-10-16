import { create } from "zustand";
import type { TaggingPlace } from "./taggingStop";

const KEY = "incarnamind.tagging.explained";
const NOT_IN_A_FOLDER = "not-in-a-folder";

const keyOf = (place: TaggingPlace) => place ?? NOT_IN_A_FOLDER;

function read(): Set<string> {
  try {
    const value = JSON.parse(localStorage.getItem(KEY) ?? "null") as unknown;
    return Array.isArray(value)
      ? new Set(value.filter((key): key is string => typeof key === "string"))
      : new Set();
  } catch {
    return new Set();
  }
}

function save(explained: ReadonlySet<string>): void {
  try {
    localStorage.setItem(KEY, JSON.stringify([...explained]));
  } catch {
    // Not remembered, then: each Folder's next run says again what is sent.
  }
}

interface TaggingExplained {
  /** The places whose tagging status has said what is sent, remembered on this computer. */
  explained: ReadonlySet<string>;
  /** The places it is saying it for now: the line stays long until this run ends. */
  showing: ReadonlySet<string>;
  /** Whether the status should say what is sent, for a run in these places. */
  says(places: readonly TaggingPlace[]): boolean;
  shown(places: readonly TaggingPlace[]): void;
  ended(): void;
}

/** Once per Folder, the tagging status says what goes to the AI model's company. */
export const useTaggingExplained = create<TaggingExplained>()((set, get) => ({
  explained: read(),
  showing: new Set(),
  says(places) {
    const { explained, showing } = get();
    return places.some((place) => !explained.has(keyOf(place)) || showing.has(keyOf(place)));
  },
  shown(places) {
    const { explained, showing } = get();
    const fresh = places.map(keyOf).filter((key) => !explained.has(key));
    if (fresh.length === 0) return;
    const next = new Set([...explained, ...fresh]);
    save(next);
    set({ explained: next, showing: new Set([...showing, ...fresh]) });
  },
  ended() {
    if (get().showing.size > 0) set({ showing: new Set() });
  },
}));
