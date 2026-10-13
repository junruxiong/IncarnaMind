import type { Document } from "../../../core/api";
import { useT } from "../i18n";
import { TagLineIcon } from "./lineIcons";
import { pickerLabel, TagPicker, tagPopoverClass, useTagPopover } from "./TagPicker";
import { buttonStyle } from "./ui";

/**
 * "Tags…" for the Documents selected, which the Tag picker adds to or removes
 * from all at once. The selection's menus open it, too (`data-selection-tags`).
 */
export function SelectionTags({
  documents,
  testId,
}: {
  documents: readonly Document[];
  testId: string;
}) {
  const t = useT();
  const popover = useTagPopover();
  const label = pickerLabel(t, documents);
  return (
    <>
      <button
        {...popover.buttonProps}
        type="button"
        data-testid={testId}
        data-selection-tags=""
        className={buttonStyle("secondary", "sm")}
      >
        <TagLineIcon className="size-3.5" />
        {t("library.selected.tags")}
      </button>
      <div
        {...popover.popoverProps}
        role="dialog"
        aria-label={label}
        data-testid="selection-tags-popover"
        className={tagPopoverClass}
      >
        {popover.open && (
          <TagPicker
            documentIds={documents.map((each) => each.id)}
            label={label}
            onReposition={popover.reposition}
          />
        )}
      </div>
    </>
  );
}
