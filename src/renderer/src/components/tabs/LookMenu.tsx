import type { MindLook } from "../../../../shared/wordLook";
import { useT } from "../../i18n";
import { openDocumentBeside } from "../../verify";
import { useWordLooks } from "../../wordLooks";
import { CheckLineIcon, DocumentLineIcon } from "../lineIcons";
import { menuItemClass, menuRuleClass, menuTitleClass } from "../ui";

export function LookMenuItems({ mindId, close }: { mindId: string; close(): void }) {
  const t = useT();
  const shown = useWordLooks((state) => state[mindId]);
  if (!shown) return null;
  const choose = (look: MindLook) => {
    shown.choose(look);
    close();
  };
  const option = (look: MindLook, label: string) => (
    <button
      type="button"
      role="menuitemradio"
      aria-checked={shown.shown === look}
      data-testid={`look-${look}`}
      onClick={() => choose(look)}
      className={`${menuItemClass} pl-2`}
    >
      <span className="flex size-4 shrink-0 items-center justify-center">
        {shown.shown === look && <CheckLineIcon className="size-4" />}
      </span>
      <span className="flex-1">{label}</span>
    </button>
  );
  const { originalId } = shown;
  return (
    <>
      <div className={menuTitleClass}>{t("look.title")}</div>
      {option("word", t("look.word"))}
      {option("plain", t("look.plain"))}
      {originalId && (
        <>
          <div className={menuRuleClass} />
          <button
            type="button"
            role="menuitem"
            data-testid="look-open-original"
            onClick={() => {
              close();
              openDocumentBeside(originalId);
            }}
            className={`${menuItemClass} pl-2`}
          >
            <DocumentLineIcon kind="docx" className="size-4 shrink-0" />
            <span className="flex-1">{t("look.openOriginal")}</span>
          </button>
        </>
      )}
      <div className={menuRuleClass} />
    </>
  );
}
