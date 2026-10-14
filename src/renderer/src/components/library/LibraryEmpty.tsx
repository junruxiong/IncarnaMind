import { useT } from "../../i18n";
import { DropLineIcon } from "../lineIcons";

/**
 * The empty Library: where files go. The whole window takes a drop; this
 * says so where the Documents will be, and Add Documents above is the way
 * without dragging.
 */
export function LibraryEmpty() {
  const t = useT();
  return (
    <div
      data-testid="library-empty"
      className="mt-6 rounded-[10px] border-[1.5px] border-dashed border-rule-strong px-6 py-12 text-center"
    >
      <DropLineIcon className="mx-auto size-5 text-ink-meta" />
      <p className="mt-3 font-serif text-heading font-semibold text-ink">
        {t("library.drop.title")}
      </p>
      <p className="mt-1.5 text-[13px] text-ink-meta">{t("library.drop.formats")}</p>
      <p className="text-[13px] text-ink-meta">{t("library.drop.stay")}</p>
    </div>
  );
}
