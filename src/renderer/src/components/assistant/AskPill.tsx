import { focusComposer } from "../../commands/assistant";
import { shortcutLabel } from "../../commands/keymap";
import { useT } from "../../i18n";

export function AskPill() {
  const t = useT();
  return (
    <button
      type="button"
      data-testid="ask-pill"
      onClick={focusComposer}
      className="absolute bottom-5 left-1/2 z-10 flex h-10 -translate-x-1/2 items-center gap-3 rounded-full bg-sheet pr-2 pl-5 font-sans text-[14px] text-ink-meta shadow-[inset_0_0_0_1px_var(--color-rule-strong),0_4px_16px_rgb(23_25_28/0.08)] hover:text-ink"
    >
      {t("assistant.pill")}
      <kbd className="rounded-sm bg-chip px-[5px] py-px font-sans text-[12px] text-ink-strong">
        {shortcutLabel("composer.focus")}
      </kbd>
    </button>
  );
}
