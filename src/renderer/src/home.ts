import { useEffect, useMemo, useState } from "react";
import { create } from "zustand";
import type { Mind } from "../../core/api";
import { core } from "./core";
import { isProcessing } from "./documentStatus";
import { exampleMindIds, useAppStore } from "./store";

const NO_MINDS: readonly Mind[] = [];

// Home's tab unmounts while another is shown: what the User unfolded and where they scrolled
// wait here for when they come back.
export const useHomeView = create<{ all: boolean; scrollTop: number }>()(() => ({
  all: false,
  scrollTop: 0,
}));

export function useOwnMinds(): readonly Mind[] {
  const minds = useAppStore((state) => state.minds);
  const examples = useAppStore((state) => exampleMindIds(state).join(" "));
  return useMemo(() => {
    const left = new Set(examples.split(" "));
    const own = minds.filter((mind) => !left.has(mind.id));
    if (own.length === 0) return NO_MINDS;
    return own.sort((a, b) => b.updatedAt.localeCompare(a.updatedAt));
  }, [minds, examples]);
}

export function useMindSnippets(): ReadonlyMap<string, string> {
  const minds = useAppStore((state) => state.minds);
  const [snippets, setSnippets] = useState<ReadonlyMap<string, string>>(new Map());
  // biome-ignore lint/correctness/useExhaustiveDependencies: a change to any Mind can change its line.
  useEffect(() => {
    let current = true;
    core
      .listMindSnippets()
      .then((list) => {
        if (current) setSnippets(new Map(list.map((each) => [each.mindId, each.text])));
      })
      .catch(useAppStore.getState().reportError);
    return () => {
      current = false;
    };
  }, [minds]);
  return snippets;
}

export function useHomeNeeds(): { reading: number; suggestions: number } {
  const reading = useAppStore(
    (state) => state.documents.filter((each) => isProcessing(each.status)).length,
  );
  const suggestions = useAppStore(
    (state) => state.library?.assignments.filter((each) => each.suggested).length ?? 0,
  );
  return { reading, suggestions };
}

const MINUTE_MS = 60_000;

export function useMinuteClock(): Date {
  const [now, setNow] = useState(() => new Date());
  useEffect(() => {
    const timer = setInterval(() => setNow(new Date()), MINUTE_MS);
    return () => clearInterval(timer);
  }, []);
  return now;
}
