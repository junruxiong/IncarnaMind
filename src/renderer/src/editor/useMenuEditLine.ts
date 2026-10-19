import type { Editor } from "@tiptap/core";
import { useEffect } from "react";
import { readinessAction } from "../components/providers/ChatReadinessNotice";
import { readinessKey, serviceName } from "../components/providers/shared";
import { dismissEditProblem, type EditProblem, stopEdit, useEdits } from "../editWithAi";
import { providerErrorText } from "../errors";
import { useLanguage, useT } from "../i18n";
import { useAppStore } from "../store";
import { appliedLineIn, setAppliedLine } from "./appliedEditLine";

/** An edit asked in the selection menu shows nothing in the page for its first 300 ms. */
const SHOWN_AFTER_MS = 300;

/**
 * Where an edit asked in the selection menu reports once the menu is closed
 * (DESIGN.md, Selection menu): in the spot its Undo line lands, which replaces
 * the line there. "Making the changes… · Stop" after 300 ms, or why it gave
 * nothing, with the one thing that helps. Esc in the text stops it.
 */
export function useMenuEditLine(editor: Editor, mindId: string, readOnly: boolean): void {
  const t = useT();
  const language = useLanguage();
  const openSettings = useAppStore((state) => state.openSettings);
  const pending = useEdits((state) => state.pending[mindId]);
  const problem = useEdits((state) => state.problems[mindId]);
  const menuShown = useEdits((state) => state.menuShows[mindId] === true);

  useEffect(() => {
    if (readOnly || editor.isDestroyed) return;
    const clear = () => {
      const kind = appliedLineIn(editor.state)?.kind;
      if (kind === "running" || kind === "failed") setAppliedLine(editor, null);
    };
    if (menuShown) {
      clear();
      return;
    }
    if (pending?.menu) {
      if (pending.stopped) {
        clear();
        return;
      }
      const { blockId, at } = pending.menu;
      const timer = setTimeout(
        () => setAppliedLine(editor, { kind: "running", blockId, stop: () => stopEdit(mindId) }),
        Math.max(0, at + SHOWN_AFTER_MS - Date.now()),
      );
      return () => clearTimeout(timer);
    }
    if (!pending && problem?.menu && problem.kind !== "busy") {
      const words = wordsOf(problem, t, language, openSettings, mindId);
      setAppliedLine(editor, {
        kind: "failed",
        blockId: problem.menu.blockId,
        text: words.text,
        action: words.action ?? null,
        close: () => dismissEditProblem(mindId),
      });
      return;
    }
    clear();
  }, [editor, mindId, readOnly, pending, problem, menuShown, t, language, openSettings]);

  useEffect(() => {
    if (readOnly) return;
    const dom = editor.view.dom;
    const onKey = (event: KeyboardEvent) => {
      const running = useEdits.getState().pending[mindId];
      if (event.key !== "Escape" || event.defaultPrevented || event.isComposing) return;
      if (!running?.menu || running.stopped || useEdits.getState().menuShows[mindId]) return;
      event.preventDefault();
      stopEdit(mindId);
    };
    dom.addEventListener("keydown", onKey);
    return () => dom.removeEventListener("keydown", onKey);
  }, [editor, mindId, readOnly]);
}

function wordsOf(
  problem: EditProblem,
  t: ReturnType<typeof useT>,
  language: ReturnType<typeof useLanguage>,
  openSettings: ReturnType<typeof useAppStore.getState>["openSettings"],
  mindId: string,
): { text: string; action?: { label: string; run(): void } } {
  switch (problem.kind) {
    case "not-ready": {
      const { readiness } = problem;
      const service = "provider" in readiness ? serviceName(readiness.provider, t, language) : "";
      const way = readinessAction(readiness, t, openSettings, t("question.setUp"), () =>
        dismissEditProblem(mindId),
      );
      return {
        text: t(readinessKey(readiness), { service }).replace(/[.。]$/, ""),
        action: { label: way.label, run: way.onClick },
      };
    }
    case "local-mode":
      return {
        text: t("edit.problem.local-mode"),
        action: {
          label: t("edit.problem.chooseModel"),
          run: () => {
            dismissEditProblem(mindId);
            openSettings("models");
          },
        },
      };
    case "failed":
      return {
        text: t("edit.problem.failed", { message: problem.message }),
        action: {
          label: t("edit.problem.retry"),
          run: () => {
            dismissEditProblem(mindId);
            problem.retry();
          },
        },
      };
    case "consent-revoked":
      return { text: providerErrorText(problem.error, t) };
    default:
      return { text: t(`edit.problem.${problem.kind}`) };
  }
}
