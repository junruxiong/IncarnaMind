import { create } from "zustand";
import { core } from "./core";

export const LEVEL_BARS = 5;

/**
 * How loud a recording Meeting is now, for its level bars: the louder
 * of its two sources at each reading, about ten a second, the latest last.
 */
export const useMeetingLevels = create<{ levels: Readonly<Record<string, readonly number[]>> }>(
  () => ({ levels: {} }),
);

core.on("meetings.level", ({ meetingId, microphone, system }) => {
  const latest = useMeetingLevels.getState().levels[meetingId] ?? [];
  useMeetingLevels.setState((state) => ({
    levels: {
      ...state.levels,
      [meetingId]: [...latest, Math.max(microphone, system)].slice(-LEVEL_BARS),
    },
  }));
});
