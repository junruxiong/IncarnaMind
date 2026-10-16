import type { Editor } from "@tiptap/react";
import { type RefObject, useEffect, useState } from "react";
import { createPortal } from "react-dom";
import type { NodeJSON } from "../../../../shared/blockJson";
import { TABLE_SLOT_CLASS, TABLE_SLOT_INDEX, tablesOf } from "../../editor/answerTables";
import { useT } from "../../i18n";
import { TableLineIcon } from "../lineIcons";

function useTableSlots(frame: RefObject<HTMLElement | null>): HTMLElement[] {
  const [slots, setSlots] = useState<HTMLElement[]>([]);
  useEffect(() => {
    const root = frame.current;
    if (!root) return;
    const find = () => {
      const found = [...root.querySelectorAll<HTMLElement>(`.${TABLE_SLOT_CLASS}`)];
      setSlots((shown) =>
        shown.length === found.length && shown.every((slot, at) => slot === found[at])
          ? shown
          : found,
      );
    };
    find();
    const changed = new MutationObserver(find);
    changed.observe(root, { childList: true, subtree: true });
    return () => changed.disconnect();
  }, [frame]);
  return slots;
}

export function OpenAsTable({
  frame,
  editor,
  onOpen,
}: {
  frame: RefObject<HTMLElement | null>;
  editor: Editor;
  onOpen(table: NodeJSON): void;
}) {
  const t = useT();
  const slots = useTableSlots(frame);
  return slots.map((slot) => {
    const index = Number(slot.getAttribute(TABLE_SLOT_INDEX));
    return createPortal(
      <button
        type="button"
        data-testid="answer-open-table"
        className="answer-table-open"
        // Keeps a selection in the Answer as it is.
        onMouseDown={(event) => event.preventDefault()}
        onClick={() => {
          const table = tablesOf(editor.state.doc)[index];
          if (table) onOpen(table);
        }}
      >
        <TableLineIcon className="size-3.5" />
        {t("answer.table.open")}
      </button>,
      slot,
      `table-${index}`,
    );
  });
}
