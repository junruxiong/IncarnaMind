/**
 * A Mind's Citations and Bibliography in its citation style (#289), as the editor shows them.
 * The core formats them on its style thread (`CoreApi.formatCitations`); the editor asks again
 * only when what is formatted changes (a Citation added, moved into or out of a cluster, a
 * reference record edited, the style switched), never at a keystroke, and shows the last
 * answer meanwhile.
 */
import type { Editor } from "@tiptap/core";
import { useEffect, useRef, useState } from "react";
import type * as Y from "yjs";
import { create } from "zustand";
import {
  type CitationFormatRequest,
  type FormattedCitations,
  MIND_SETTINGS_FIELD,
} from "../../core/api";
import type { CitationStyleId } from "../../shared/citationStyles";
import { citationStyleIn } from "../../shared/mindCitationStyle";
import { core } from "./core";
import { citationClusters, citedWorkOf } from "./editor/citationClusters";
import { errorMessage } from "./errors";
import { useLanguage } from "./i18n";
import { useAppStore } from "./store";

/** A Mind's formatting, as last answered. */
export interface MindFormatting {
  style: CitationStyleId;
  /** The last answer, and the signature of each cluster it answered, in order. */
  formatted: FormattedCitations | null;
  signatures: readonly string[];
  /** A newer request is being formatted. */
  pending: boolean;
  /** Why the last request failed, in English, or null. */
  problem: string | null;
}

interface FormattingState {
  minds: Readonly<Record<string, MindFormatting>>;
  update(mindId: string, change: Partial<MindFormatting> & { style: CitationStyleId }): void;
  forget(mindId: string): void;
}

export const useCitationFormatting = create<FormattingState>()((set) => ({
  minds: {},
  update: (mindId, change) =>
    set((state) => {
      const before = state.minds[mindId];
      const next: MindFormatting = {
        formatted: null,
        signatures: [],
        pending: false,
        problem: null,
        ...before,
        ...change,
      };
      return { minds: { ...state.minds, [mindId]: next } };
    }),
  forget: (mindId) =>
    set((state) => {
      const { [mindId]: _, ...rest } = state.minds;
      return { minds: rest };
    }),
}));

/** The Mind's formatting, or undefined before the editor has asked. */
export const useMindFormatting = (mindId: string) =>
  useCitationFormatting((state) => state.minds[mindId]);

/** The style a Mind's document says it is in, following its changes and the interface language. */
export function useMindCitationStyle(doc: Y.Doc): CitationStyleId {
  const language = useLanguage();
  const settings = doc.getMap<unknown>(MIND_SETTINGS_FIELD);
  const [style, setStyle] = useState(() => citationStyleIn(settings, language));
  useEffect(() => {
    const read = () => setStyle(citationStyleIn(settings, language));
    read();
    settings.observe(read);
    return () => settings.unobserve(read);
  }, [settings, language]);
  return style;
}

/** How long the editor waits after a change before it asks again: a run of changes asks once. */
const SETTLE_MS = 150;

/**
 * Keeps the Mind's formatting (`useCitationFormatting`) in step with its editor: asks the core
 * when the clusters, the cited Documents' records or the style change, and keeps the newest
 * answer. Mounted once per open Mind.
 */
export function useCitationFormattingFor(editor: Editor, doc: Y.Doc, mindId: string): void {
  const style = useMindCitationStyle(doc);
  const documents = useAppStore((state) => state.documents);
  /** What was asked last, kept across the effect's runs (Documents change often). */
  const last = useRef({ asked: "", latest: 0 });

  useEffect(() => {
    const progress = last.current;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const { update } = useCitationFormatting.getState();

    const ask = () => {
      timer = undefined;
      if (editor.isDestroyed) return;
      const clusters = citationClusters(editor.state);
      const request: CitationFormatRequest = {
        style,
        clusters: clusters.map((cluster) => ({
          works: cluster.members.map((member) => citedWorkOf(member.attributes)),
        })),
      };
      // A record edited since is a new request, though the Citations are as they were.
      const records = new Map(
        documents.map((document) => [document.id, document.reference?.updatedAt ?? ""]),
      );
      const cited = request.clusters.flatMap((cluster) =>
        cluster.works.map((work) => (work.documentId ? (records.get(work.documentId) ?? "-") : "")),
      );
      const key = JSON.stringify([request, cited]);
      if (key === progress.asked) return;
      progress.asked = key;
      const signatures = clusters.map((cluster) => cluster.signature);
      const id = ++progress.latest;
      update(mindId, { style, pending: true });
      core.formatCitations(request).then(
        (formatted) => {
          if (id !== progress.latest) return;
          update(mindId, { style, formatted, signatures, pending: false, problem: null });
        },
        (error: unknown) => {
          if (id !== progress.latest) return;
          progress.asked = "";
          update(mindId, { style, pending: false, problem: errorMessage(error) });
        },
      );
    };

    ask();
    const onTransaction = ({ transaction }: { transaction: { docChanged: boolean } }) => {
      if (!transaction.docChanged) return;
      clearTimeout(timer);
      timer = setTimeout(ask, SETTLE_MS);
    };
    editor.on("transaction", onTransaction);
    const again = () => {
      progress.asked = "";
      ask();
    };
    retries.set(mindId, again);
    return () => {
      clearTimeout(timer);
      editor.off("transaction", onTransaction);
      if (retries.get(mindId) === again) retries.delete(mindId);
    };
  }, [editor, mindId, style, documents]);

  // Closed: an answer still on its way is dropped, and the Mind's formatting forgotten.
  useEffect(() => {
    const progress = last.current;
    return () => {
      progress.latest++;
      progress.asked = "";
      useCitationFormatting.getState().forget(mindId);
    };
  }, [mindId]);
}

/** How to ask again for each open Mind, for its problem line's Try again. */
const retries = new Map<string, () => void>();

/** Asks again for a Mind whose last request failed. */
export const retryFormatting = (mindId: string) => retries.get(mindId)?.();
