import { DOCKS, dockCommandId } from "../../commands/assistant";
import { commands } from "../../commands/registry";
import { useT } from "../../i18n";
import { useAppStore } from "../../store";
import {
  DockBottomLineIcon,
  DockLeftLineIcon,
  DockRightLineIcon,
  FloatLineIcon,
  MoreLineIcon,
} from "../lineIcons";
import { menuClass, menuItemClass, menuItemUnavailableClass, menuRuleClass } from "../ui";
import { usePopoverMenu } from "../usePopoverMenu";

const ICONS = {
  right: DockRightLineIcon,
  left: DockLeftLineIcon,
  bottom: DockBottomLineIcon,
  float: FloatLineIcon,
} as const;

/**
 * The assistant's ⋯: where it sits (the User's choice is checked, even while
 * a narrow window shows a side dock at the bottom), then what acts on its
 * conversation. Each place is a command, as in ⌘K.
 */
export function AssistantMenu() {
  const t = useT();
  const menu = usePopoverMenu();
  const chosen = useAppStore((state) => state.layout?.assistant.dock);
  const run = (id: Parameters<typeof commands.run>[0]) => {
    menu.close();
    commands.run(id).catch(useAppStore.getState().reportError);
  };
  return (
    <>
      <button
        {...menu.buttonProps}
        type="button"
        data-testid="assistant-more"
        data-open={menu.open || undefined}
        aria-label={t("assistant.more")}
        title={t("assistant.more")}
        className="assistant-button"
      >
        <MoreLineIcon className="size-4" />
      </button>
      <div
        {...menu.menuProps}
        role="menu"
        aria-label={t("assistant.more")}
        data-testid="assistant-menu"
        className={menuClass}
      >
        {menu.open && (
          <>
            {DOCKS.map(({ dock, label }) => {
              const Icon = ICONS[dock];
              return (
                <button
                  key={dock}
                  type="button"
                  role="menuitemradio"
                  aria-checked={chosen === dock}
                  data-testid={`assistant-menu-${dock}`}
                  onClick={() => run(dockCommandId(dock))}
                  className={`${menuItemClass} pl-2`}
                >
                  <Icon className="size-4 shrink-0 text-ink-meta" />
                  <span className="flex-1">{t(label)}</span>
                  {chosen === dock && <span aria-hidden="true">✓</span>}
                </button>
              );
            })}
            <div className={menuRuleClass} />
            {/* What acts on the conversation's Answers comes with them (#201): an empty
                conversation has none, so these wait, saying why. */}
            <ConversationItem
              testId="assistant-menu-last-to-new-mind"
              label={t("assistant.lastToNewMind")}
              reason={t("assistant.lastToNewMind.unavailable")}
            />
            <ConversationItem
              testId="assistant-menu-clear"
              label={t("assistant.clear")}
              reason={t("assistant.clear.unavailable")}
            />
          </>
        )}
      </div>
    </>
  );
}

/** An item that can't be used in an empty conversation: in the arrow keys' reach, greyed, saying why. */
function ConversationItem({
  testId,
  label,
  reason,
}: {
  testId: string;
  label: string;
  reason: string;
}) {
  return (
    <button
      type="button"
      role="menuitem"
      aria-disabled="true"
      data-testid={testId}
      title={reason}
      className={`${menuItemClass} ${menuItemUnavailableClass} pl-8`}
    >
      {label}
    </button>
  );
}
