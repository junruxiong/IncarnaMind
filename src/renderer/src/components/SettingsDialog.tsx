import { useEffect, useRef } from "react";
import type { LanguagePreference } from "../../../core/language";
import type { MessageKey } from "../../../shared/i18n";
import { useT } from "../i18n";
import { useAppStore } from "../store";

const languageOptions: readonly { value: LanguagePreference; label: MessageKey }[] = [
  { value: "system", label: "settings.language.system" },
  { value: "en", label: "settings.language.en" },
  { value: "zh-CN", label: "settings.language.zh-CN" },
];

/** A native modal <dialog>, replacing the old MUI modal. */
export function SettingsDialog({ open, onClose }: { open: boolean; onClose(): void }) {
  const t = useT();
  const dialog = useRef<HTMLDialogElement>(null);
  const language = useAppStore((state) => state.settings?.user.language);
  const updateSettings = useAppStore((state) => state.updateSettings);

  useEffect(() => {
    const element = dialog.current;
    if (!element) return;
    if (open && !element.open) element.showModal();
    if (!open && element.open) element.close();
  }, [open]);

  return (
    <dialog
      ref={dialog}
      onClose={onClose}
      aria-labelledby="settings-title"
      className="m-auto w-96 rounded-[9px] bg-white p-4 text-gray-800 shadow-custom-focus backdrop:bg-black/20"
    >
      <h2 id="settings-title" className="text-lg font-semibold">
        {t("settings.title")}
      </h2>
      <fieldset className="mt-3">
        <legend className="mb-1 text-sm text-gray-600">{t("settings.language")}</legend>
        {languageOptions.map((option) => (
          <label key={option.value} className="flex items-center gap-2 py-1 text-sm">
            <input
              type="radio"
              name="language"
              value={option.value}
              checked={language === option.value}
              onChange={() => void updateSettings({ user: { language: option.value } })}
            />
            {t(option.label)}
          </label>
        ))}
      </fieldset>
      <div className="mt-4 flex justify-end">
        <button
          type="button"
          onClick={onClose}
          className="rounded-[9px] border border-gray-300 px-4 py-2 text-sm hover:bg-gray-100"
        >
          {t("settings.done")}
        </button>
      </div>
    </dialog>
  );
}
