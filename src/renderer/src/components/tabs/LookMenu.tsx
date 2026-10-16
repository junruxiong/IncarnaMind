import type { CommandId } from "../../../../shared/commands";
import type { MindLook } from "../../../../shared/wordLook";
import { type CommandContext, commands } from "../../commands/registry";
import { useT } from "../../i18n";
import { useAppStore } from "../../store";
import { useWordLooks } from "../../wordLooks";
import { CheckLineIcon, DocumentLineIcon } from "../lineIcons";
import { menuItemClass, menuRuleClass, menuTitleClass } from "../ui";

const LOOK_COMMANDS: Record<MindLook, CommandId> = {
  word: "mind.look.word",
  plain: "mind.look.plain",
};

export function LookMenuItems({
  mindId,
  tabId,
  close,
}: {
  mindId: string;
  tabId: string;
  close(): void;
}) {
  const t = useT();
  const shown = useWordLooks((state) => state[mindId]);
  if (!shown) return null;
  const ctx: CommandContext = { selection: [], focus: { kind: "tab", id: tabId }, surface: "mind" };
  const run = (id: CommandId) => {
    close();
    commands.run(id, ctx).catch(useAppStore.getState().reportError);
  };
  const option = (look: MindLook, label: string) => (
    <button
      type="button"
      role="menuitemradio"
      aria-checked={shown.shown === look}
      data-testid={`look-${look}`}
      onClick={() => (shown.shown === look ? close() : run(LOOK_COMMANDS[look]))}
      className={`${menuItemClass} pl-2`}
    >
      <span className="flex size-4 shrink-0 items-center justify-center">
        {shown.shown === look && <CheckLineIcon className="size-4" />}
      </span>
      <span className="flex-1">{label}</span>
    </button>
  );
  return (
    <>
      <div className={menuTitleClass}>{t("look.title")}</div>
      {option("word", t("look.word"))}
      {option("plain", t("look.plain"))}
      {shown.openOriginal && (
        <>
          <div className={menuRuleClass} />
          <button
            type="button"
            role="menuitem"
            data-testid="look-open-original"
            onClick={() => run("mind.openOriginal")}
            className={`${menuItemClass} pl-2`}
          >
            <DocumentLineIcon kind="docx" className="size-4 shrink-0" />
            <span className="flex-1">{t("wordImport.openOriginal")}</span>
          </button>
        </>
      )}
      <div className={menuRuleClass} />
    </>
  );
}
