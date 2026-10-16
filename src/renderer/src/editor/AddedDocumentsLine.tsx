import type { Editor } from "@tiptap/core";
import { useMemo } from "react";
import type * as Y from "yjs";
import {
  closeAdded,
  useAddedDocuments,
  useClosedAdded,
  useWatchAddedDocuments,
} from "../addedDocuments";
import { useChecks } from "../checks";
import { ProblemLine } from "../components/ProblemLine";
import { useLanguage, useT } from "../i18n";
import { startMindCheck, useMindCheck } from "../mindCheck";

/**
 * "2 Documents added since these checks · Check again · Close" (#993), under
 * the Mind's health line, when Documents were added to the scope of checks
 * they could change. Check again checks only those, with the estimate first
 * as any check made again; the line goes once they are checked again.
 */
export function AddedDocumentsLine({
  editor,
  doc,
  mindId,
}: {
  editor: Editor;
  doc: Y.Doc;
  mindId: string;
}) {
  const t = useT();
  const language = useLanguage();
  useWatchAddedDocuments(editor, mindId);
  const added = useAddedDocuments((state) => state.byMind[mindId]);
  const closed = useClosedAdded(doc);
  const running = useChecks((state) => state.running[mindId] !== undefined);
  const asking = useMindCheck((state) => state.asking[mindId] !== undefined);
  const documentIds = useMemo(() => [...new Set(Object.values(added ?? {}).flat())], [added]);
  if (running || asking || documentIds.every((id) => closed.has(id))) return null;
  const count = documentIds.length;
  return (
    <ProblemLine
      role="status"
      testId="checks-added"
      className="mind-health-line"
      action={{
        label: t("claim.checkAgain"),
        testId: "checks-added-check",
        onClick: () => void startMindCheck(mindId, "added"),
      }}
      also={{
        label: t("checks.close"),
        testId: "checks-added-close",
        onClick: () => closeAdded(doc, documentIds),
      }}
    >
      {count === 1
        ? t("checks.added.one")
        : t("checks.added", { count: new Intl.NumberFormat(language).format(count) })}
    </ProblemLine>
  );
}
