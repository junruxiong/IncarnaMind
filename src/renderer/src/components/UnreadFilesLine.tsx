import { useId, useState } from "react";
import type { UnreadFile } from "../../../core/api";
import { useT } from "../i18n";
import { useAppStore } from "../store";
import { UNREAD_REASON_KEYS } from "../unreadFiles";
import { ProblemActionButton, ProblemLine } from "./ProblemLine";

/**
 * "3 files couldn't be read · Show" (#740; DESIGN.md, Problems): beside an
 * Answer, and in the Library, when a file in what was searched couldn't be
 * read. Show lists each with why and what to do: a locked PDF offers
 * Unlock…, which opens it where its password is asked for (#763). Nothing
 * when there are none: a line that always showed would be one nobody read.
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
  const documents = useAppStore((store) => store.documents);
  const openDocument = useAppStore((store) => store.openDocument);
  const lockedPdf = (file: UnreadFile) =>
    file.reason === "password"
      ? (documents.find((each) => each.id === file.documentId && each.kind === "pdf") ?? null)
      : null;
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
          {files.map((file) => {
            const locked = lockedPdf(file);
            return (
              <li
                key={file.id}
                data-testid="unread-file"
                data-reason={file.reason}
                className="font-sans text-[13px] leading-5 break-words text-ink-meta"
              >
                {t("unread.file", {
                  name: file.name,
                  reason: locked
                    ? t("unread.reason.passwordPdf")
                    : t(UNREAD_REASON_KEYS[file.reason], { extension: file.extension ?? "" }),
                })}
                {locked?.fileStatus === "available" && (
                  <>
                    {" "}
                    <span className="whitespace-nowrap">
                      <span aria-hidden="true">· </span>
                      <ProblemActionButton
                        action={{
                          label: t("documents.unlockAction"),
                          testId: "unread-file-unlock",
                          onClick: () => openDocument({ documentId: locked.id }),
                        }}
                      />
                    </span>
                  </>
                )}
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}
