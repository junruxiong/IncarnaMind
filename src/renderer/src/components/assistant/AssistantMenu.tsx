import { DOCKS, dockCommandId } from "../../commands/assistant";
import { commands, labelOf, offered } from "../../commands/registry";
import { useConversations } from "../../conversations";
import { useT } from "../../i18n";
import { useAppStore } from "../../store";
import {
  CheckLineIcon,
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
  // Offered while there is a Mind to make it from: the one the assistant reads.
  const make = commands.get("skill.fromMind");
  const makeSkill =
    menu.open && make && offered(make, { selection: [], focus: null, surface: "mind" });
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
                  {chosen === dock && <CheckLineIcon className="size-4 shrink-0 text-ink-meta" />}
                </button>
              );
            })}
            <div className={menuRuleClass} />
            {makeSkill && (
              <button
                type="button"
                role="menuitem"
                data-testid="assistant-menu-make-skill"
                onClick={() => run("skill.fromMind")}
                className={`${menuItemClass} pl-8`}
              >
                {t("skills.make.command")}
              </button>
            )}
            <ConversationItem
              id="assistant.lastToNewMind"
              testId="assistant-menu-last-to-new-mind"
              onRun={run}
            />
            <ConversationItem id="assistant.clear" testId="assistant-menu-clear" onRun={run} />
          </>
        )}
      </div>
    </>
  );
}

/**
 * An item that acts on the conversation, as its command does; while it can't,
 * it stays in the arrow keys' reach, greyed, saying why.
 */
function ConversationItem({
  id,
  testId,
  onRun,
}: {
  id: "assistant.lastToNewMind" | "assistant.clear";
  testId: string;
  onRun(id: "assistant.lastToNewMind" | "assistant.clear"): void;
}) {
  const t = useT();
  // The conversation's messages decide whether it can be used.
  useConversations((state) => state.messages);
  const command = commands.get(id);
  if (!command) return null;
  const ctx = { selection: [], focus: null, surface: "mind" } as const;
  const enabled = command.enabled?.(ctx) ?? true;
  const label = labelOf(command, ctx, t);
  return (
    <button
      type="button"
      role="menuitem"
      aria-disabled={enabled === true ? undefined : true}
      data-testid={testId}
      title={enabled === true ? undefined : t(enabled.reason)}
      onClick={() => {
        if (enabled === true) onRun(id);
      }}
      className={`${menuItemClass} ${enabled === true ? "" : menuItemUnavailableClass} pl-8`}
    >
      {label}
    </button>
  );
}
