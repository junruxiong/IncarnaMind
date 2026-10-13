import { useCallback, useEffect, useRef, useState } from "react";
import type { DocumentTextMatch } from "../../core/api";
import { core } from "./core";
import { errorMessage } from "./errors";
import { searchesWords } from "./palette";

/** As ⌘K: the core is asked once typing pauses. */
const WORDS_DELAY_MS = 120;

/** As many Documents as a search can return (`searchDocumentText`). */
const MOST_DOCUMENTS = 200;

interface Answer {
  query: string;
  matches: Map<string, DocumentTextMatch>;
  failed: string | null;
}

/**
 * The words inside Documents for the Library's search, by Document id, once
 * the core answers for what is typed now; `waiting` until it does. `failed`
 * says why it couldn't search, and `retry` asks again.
 */
export function useTextSearch(search: string): {
  matches: ReadonlyMap<string, DocumentTextMatch> | null;
  waiting: boolean;
  failed: string | null;
  retry(): void;
} {
  const query = search.trim();
  const searching = searchesWords(query);
  const [answer, setAnswer] = useState<Answer | null>(null);
  const latest = useRef(query);
  useEffect(() => {
    latest.current = query;
  }, [query]);

  const ask = useCallback((asked: string) => {
    core.searchDocumentText(asked, { limit: MOST_DOCUMENTS }).then(
      (items) => {
        if (latest.current !== asked) return;
        const matches = new Map(items.map((item) => [item.documentId, item]));
        setAnswer({ query: asked, matches, failed: null });
      },
      (error: unknown) => {
        if (latest.current === asked) {
          setAnswer({ query: asked, matches: new Map(), failed: errorMessage(error) });
        }
      },
    );
  }, []);

  useEffect(() => {
    if (!searching) {
      setAnswer(null);
      return;
    }
    const timer = setTimeout(() => ask(query), WORDS_DELAY_MS);
    return () => clearTimeout(timer);
  }, [searching, query, ask]);

  // Each answer stays until the next arrives, so rows don't blink as the User types.
  const shown = searching ? answer : null;
  return {
    matches: shown?.matches ?? null,
    waiting: searching && answer?.query !== query,
    failed: shown?.query === query ? shown.failed : null,
    retry: () => ask(query),
  };
}
