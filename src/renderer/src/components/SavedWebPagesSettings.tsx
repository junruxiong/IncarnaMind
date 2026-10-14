import { useId } from "react";
import { useT } from "../i18n";
import { useAppStore } from "../store";
import {
  rowTextClass,
  rowTitleClass,
  ruledListClass,
  ruledRowClass,
  sectionTitleClass,
} from "./ui";

export function SavedWebPagesSettings() {
  const t = useT();
  const on = useAppStore((state) => state.settings?.user.keepWebPageLook);
  const updateSettings = useAppStore((state) => state.updateSettings);
  const id = useId();
  if (on === undefined) return null;
  return (
    <section data-testid="saved-web-pages-settings">
      <h4 className={`mb-2 ${sectionTitleClass}`}>{t("web.settings.pages.title")}</h4>
      <div className={ruledListClass}>
        <div className={ruledRowClass}>
          <div className="flex min-w-0 flex-col gap-0.5">
            <label htmlFor={id} className={rowTitleClass}>
              {t("web.settings.keepLook")}
            </label>
            <p id={`${id}-description`} className={rowTextClass}>
              {t("webPages.keepLook.body")}
            </p>
          </div>
          <input
            id={id}
            type="checkbox"
            role="switch"
            aria-checked={on}
            aria-describedby={`${id}-description`}
            data-testid="keep-web-page-look-switch"
            checked={on}
            onChange={(event) =>
              void updateSettings({ user: { keepWebPageLook: event.target.checked } })
            }
            className="switch"
          />
        </div>
      </div>
    </section>
  );
}
