import type { LanguagePreference } from "../../../core/language";
import type { MessageKey } from "../../../shared/i18n";
import { useT } from "../i18n";
import { useAppStore } from "../store";
import { ChatGptPlanSettings } from "./providers/ChatGptPlanSettings";
import { ChatModelSettings } from "./providers/ChatModelSettings";
import { ConsentSettings } from "./providers/ConsentSettings";
import { JevSettingsSection } from "./providers/JevSettings";
import { useModal } from "./useModal";

const languageOptions: readonly { value: LanguagePreference; label: MessageKey }[] = [
  { value: "system", label: "settings.language.system" },
  { value: "en", label: "settings.language.en" },
  { value: "zh-CN", label: "settings.language.zh-CN" },
];

/** A native modal <dialog>, replacing the old MUI modal. */
export function SettingsDialog() {
  const t = useT();
  const open = useAppStore((state) => state.settingsOpen);
  const close = useAppStore((state) => state.closeSettings);
  const language = useAppStore((state) => state.settings?.user.language);
  const updateSettings = useAppStore((state) => state.updateSettings);
  const dialog = useModal(open);

  return (
    <dialog
      ref={dialog}
      onClose={close}
      data-testid="settings"
      aria-labelledby="settings-title"
      className="m-auto max-h-[90vh] w-[34rem] max-w-[calc(100vw-2rem)] overflow-y-auto rounded-[9px] bg-white p-5 text-gray-800 shadow-custom-focus backdrop:bg-black/20"
    >
      <h2 id="settings-title" className="text-lg font-semibold">
        {t("settings.title")}
      </h2>
      {/* Rendered only while open, so each section loads fresh data. */}
      {open && (
        <div className="mt-4 flex flex-col gap-6">
          <ChatModelSettings />
          <JevSettingsSection />
          <fieldset>
            <legend className="mb-1 text-sm font-medium">{t("settings.language")}</legend>
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
          <ConsentSettings />
          <ChatGptPlanSettings />
        </div>
      )}
      <div className="mt-5 flex justify-end">
        <button
          type="button"
          onClick={close}
          className="rounded-[9px] border border-gray-300 px-4 py-2 text-sm hover:bg-gray-100"
        >
          {t("settings.done")}
        </button>
      </div>
    </dialog>
  );
}
