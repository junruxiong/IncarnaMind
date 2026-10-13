import { useId, useState } from "react";
import type { UnreadFile } from "../../../core/api";
import { useT } from "../i18n";
import { UNREAD_REASON_KEYS } from "../unreadFiles";
import { ProblemLine } from "./ProblemLine";

/**
 * "3 files couldn't be read · Show" (#740; DESIGN.md, Problems): beside an
 * Answer, and in the Library, when a file in what was searched couldn't be
 * read. Show lists each with why and what to do. Nothing when there are none:
 * a line that always showed would be one nobody read.
 */
export function UnreadFilesLine({
  files,
  testId = "unread-files",
  className = "",
}: {
  files: readonly UnreadFile[];
  testId?: string;
  className?: string;
}) {
  const t = useT();
  const [open, setOpen] = useState(false);
  const listId = useId();
  if (files.length === 0) return null;
  return (
    <div data-testid={testId} className={className}>
      <ProblemLine
        role="status"
        action={{
          label: open ? t("unread.hide") : t("unread.show"),
          testId: `${testId}-show`,
          onClick: () => setOpen((now) => !now),
        }}
      >
        {t(files.length === 1 ? "unread.count.one" : "unread.count", { count: files.length })}
      </ProblemLine>
      {open && (
        <ul id={listId} data-testid={`${testId}-list`} className="mt-1 flex flex-col gap-1">
          {files.map((file) => (
            <li
              key={file.id}
              data-testid="unread-file"
              data-reason={file.reason}
              className="font-sans text-[13px] leading-5 break-words text-ink-meta"
            >
              {t("unread.file", {
                name: file.name,
                reason: t(UNREAD_REASON_KEYS[file.reason], { extension: file.extension ?? "" }),
              })}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
