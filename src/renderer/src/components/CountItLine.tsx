import { type ReactNode, useState } from "react";
import type { Document } from "../../../core/api";
import { countAsEvidence } from "../evidence";
import { useT } from "../i18n";
import { useAppStore } from "../store";
import { ProblemLine } from "./ProblemLine";

/**
 * Why a Document isn't counted as evidence, with Count it (#80), in the meta
 * tone of a problem line: under a file IncarnaMind wrote, in its viewer; and
 * under a check's result, for each such Document where it matched ("Also
 * found in “Chapter 2.docx”, which IncarnaMind wrote, … · Count it", #96).
 * Counting offers Undo; the Document's menu takes it back later.
 */
export function CountItLine({
  document,
  children,
  testId,
  className,
}: {
  document: Pick<Document, "id" | "name" | "title">;
  children: ReactNode;
  testId?: string;
  className?: string;
}) {
  const t = useT();
  const [counting, setCounting] = useState(false);
  const count = () => {
    setCounting(true);
    countAsEvidence([document], true)
      .catch(useAppStore.getState().reportError)
      .finally(() => setCounting(false));
  };
  return (
    <ProblemLine
      role="status"
      testId={testId}
      className={className}
      action={{
        label: t("evidence.countIt"),
        testId: "count-it",
        disabled: counting,
        onClick: count,
      }}
    >
      {children}
    </ProblemLine>
  );
}
