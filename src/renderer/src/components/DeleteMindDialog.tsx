import { useEffect, useRef } from "react";
import type { Mind } from "../../../core/api";
import { useT } from "../i18n";
import { useAppStore } from "../store";

/** Asks before deleting a Mind, like the old app's "Delete Mind" modal. Open while `mind` is set. */
export function DeleteMindDialog({ mind, onClose }: { mind: Mind | null; onClose(): void }) {
  const t = useT();
  const dialog = useRef<HTMLDialogElement>(null);
  const deleteMind = useAppStore((state) => state.deleteMind);

  useEffect(() => {
    const element = dialog.current;
    if (!element) return;
    if (mind && !element.open) element.showModal();
    if (!mind && element.open) element.close();
  }, [mind]);

  return (
    <dialog
      ref={dialog}
      onClose={onClose}
      aria-labelledby="delete-mind-title"
      className="m-auto max-w-md rounded-[9px] bg-white p-4 text-gray-800 shadow-custom-focus backdrop:bg-black/20"
    >
      <h2 id="delete-mind-title" className="text-lg font-semibold">
        {t("mind.delete")}
      </h2>
      <p className="mt-2 text-sm break-words">
        {t("mind.delete.body", { title: mind?.title || t("mind.untitled") })}
      </p>
      <div className="mt-4 flex justify-end gap-2">
        <button
          type="button"
          onClick={onClose}
          className="rounded-[9px] border border-gray-300 px-4 py-2 text-sm hover:bg-gray-100"
        >
          {t("mind.delete.cancel")}
        </button>
        <button
          type="button"
          data-testid="confirm-delete-mind"
          onClick={() => {
            if (mind) void deleteMind(mind.id);
            onClose();
          }}
          className="rounded-[9px] bg-red-700 px-4 py-2 text-sm text-white hover:bg-red-800"
        >
          {t("mind.delete.confirm")}
        </button>
      </div>
    </dialog>
  );
}
