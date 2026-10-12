import { useRef } from "react";
import { useCopied } from "../copied";
import { useT } from "../i18n";
import { useFade } from "../motion";
import { CheckLineIcon } from "./lineIcons";

/**
 * "Copied", for a few seconds, at the foot of the sidebar beside the undo
 * line: what the User asked to copy (a path) is on the clipboard. Not an
 * error, so it goes by itself. Read out politely.
 */
export function CopiedLine() {
  const shown = useCopied((state) => state.shown);
  return (
    <div role="status" aria-live="polite" data-testid="copied-line" className="flex flex-col">
      {shown && <Line />}
    </div>
  );
}

function Line() {
  const t = useT();
  const line = useRef<HTMLDivElement>(null);
  useFade(line);
  return (
    <div
      ref={line}
      className="mx-2 flex items-center gap-1 pt-0.5 pb-1 pl-2 text-[12px] leading-4 text-ink-secondary"
    >
      <CheckLineIcon className="size-3.5 shrink-0" />
      {t("info.copied")}
    </div>
  );
}
