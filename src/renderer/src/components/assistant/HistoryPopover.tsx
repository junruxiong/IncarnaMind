import { useRef } from "react";
import { Popover } from "../../editor/Popover";
import { useT } from "../../i18n";
import { CloseLineIcon, HistoryLineIcon } from "../lineIcons";
import { iconButtonClass } from "../ui";

/**
 * The assistant's History, from its clock button, its title's ▾ or
 * "Conversation history" in ⌘K. The assistant's conversations are kept with
 * #300 and listed here with #258 (this Folder's first, All Folders, search);
 * until then there are none, and it says so.
 */
export function HistoryPopover({ anchor, onClose }: { anchor: Element; onClose(): void }) {
  const t = useT();
  const close = useRef<HTMLButtonElement>(null);
  return (
    <Popover
      anchor={anchor}
      placement="bottom-end"
      onDismiss={onClose}
      initialFocus={close}
      aria-label={t("assistant.history")}
      data-testid="assistant-history-popover"
    >
      <section className="assistant-history">
        <header className="assistant-history-head">
          <HistoryLineIcon className="size-4 shrink-0 text-ink-meta" />
          <h2 className="min-w-0 flex-1 truncate">{t("assistant.history")}</h2>
          <button
            ref={close}
            type="button"
            aria-label={t("assistant.history.close")}
            title={t("assistant.history.close")}
            onClick={onClose}
            className={iconButtonClass}
          >
            <CloseLineIcon className="size-4" />
          </button>
        </header>
        <p data-testid="assistant-history-empty" className="assistant-history-empty">
          {t("assistant.history.empty")}
        </p>
      </section>
    </Popover>
  );
}
