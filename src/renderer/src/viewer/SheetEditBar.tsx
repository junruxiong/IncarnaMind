import { useEffect, useState } from "react";
import { columnName } from "../../../shared/units";
import { useT } from "../i18n";
import { folderName } from "../linkedFolders";
import { defaultFolderLabel } from "../outputFiles";
import { useAppStore } from "../store";
import type { SelectedCell, SheetEditing } from "./sheetEditing";

/**
 * The editor's bar under the sheet's toolbar (spike: excel-edit), as Excel's
 * formula bar: the selected cell's reference and its formula or value, typed
 * into in place of the cell; then how many cells were changed, the cells that
 * couldn't be recalculated, and where the copy was saved.
 */
export function SheetEditBar({
  editing,
  sheet,
  selected,
  draft,
  setDraft,
  commit,
}: {
  editing: SheetEditing;
  sheet: string;
  selected: SelectedCell | null;
  draft: string | null;
  setDraft(text: string | null): void;
  commit(text: string): void;
}) {
  const t = useT();
  const openDocument = useAppStore((state) => state.openDocument);
  const shown = selected ? (draft ?? editing.inputOf(sheet, selected)) : "";
  const [, redraw] = useState(0);
  // biome-ignore lint/correctness/useExhaustiveDependencies: shows the new value after each edit
  useEffect(() => redraw((n) => n + 1), [editing.revision]);
  const saved = editing.saved;
  return (
    <div data-testid="sheet-edit-bar" className="viewer-sheet-editbar">
      <div className="viewer-sheet-formula">
        <span data-testid="sheet-edit-ref" className="viewer-sheet-ref">
          {selected ? `${columnName(selected.column)}${selected.row}` : ""}
        </span>
        <input
          data-testid="sheet-formula-input"
          aria-label={t("viewer.sheet.formula")}
          value={shown}
          disabled={!selected}
          onChange={(event) => setDraft(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter") {
              event.preventDefault();
              commit(shown);
            } else if (event.key === "Escape") {
              event.preventDefault();
              setDraft(null);
            }
          }}
          className="viewer-sheet-formula-input"
        />
      </div>
      <div className="viewer-sheet-editstatus">
        {editing.problem ? (
          <span data-testid="sheet-edit-problem" className="viewer-sheet-problem">
            {editing.problem}
          </span>
        ) : saved?.status === "saved" ? (
          <span data-testid="sheet-edit-saved">
            {t("viewer.sheet.saved", {
              name: saved.file.name,
              folder: saved.file.inDefaultFolder
                ? defaultFolderLabel(t)
                : folderName(saved.file.folder),
            })}{" "}
            <button
              type="button"
              data-testid="sheet-open-copy"
              className="viewer-sheet-link"
              onClick={() => openDocument({ documentId: saved.document.id })}
            >
              {t("viewer.sheet.openCopy")}
            </button>
          </span>
        ) : (
          <span data-testid="sheet-edit-count">
            {editing.edits.length === 0
              ? t("viewer.sheet.editHint")
              : t(
                  editing.edits.length === 1
                    ? "viewer.sheet.edited.one"
                    : "viewer.sheet.edited.other",
                  {
                    count: editing.edits.length,
                  },
                )}
            {editing.stale > 0 && ` · ${t("viewer.sheet.stale", { count: editing.stale })}`}
          </span>
        )}
      </div>
    </div>
  );
}
