import type { KeyboardEvent } from "react";
import type { LanguagePreference } from "../../../core/language";
import type { MessageKey } from "../../../shared/i18n";
import { files } from "../core";
import { errorMessage } from "../errors";
import { useT } from "../i18n";
import { type SettingsPage, useAppStore } from "../store";
import { ApprovalsSettings } from "./ApprovalsSettings";
import { ConnectorsSettings } from "./connectors/ConnectorsSettings";
import { PrivacySettings } from "./PrivacySettings";
import { ChatGptPlanSettings } from "./providers/ChatGptPlanSettings";
import { ChatModelSettings } from "./providers/ChatModelSettings";
import { EmbeddingSettingsSection } from "./providers/EmbeddingSettings";
import { JevSettingsSection } from "./providers/JevSettings";
import { RerankSettingsSection } from "./providers/RerankSettings";
import { buttonClass } from "./providers/shared";
import { SkillsSettings } from "./SkillsSettings";
import { useModal } from "./useModal";

const languageOptions: readonly { value: LanguagePreference; label: MessageKey }[] = [
  { value: "system", label: "settings.language.system" },
  { value: "en", label: "settings.language.en" },
  { value: "zh-CN", label: "settings.language.zh-CN" },
];

const pages: readonly { page: SettingsPage; label: MessageKey }[] = [
  { page: "general", label: "privacy.tabs.general" },
  { page: "privacy", label: "privacy.tabs.privacy" },
];

/** A native modal <dialog>, replacing the old MUI modal, with a General and a Privacy page. */
export function SettingsDialog() {
  const t = useT();
  const open = useAppStore((state) => state.settingsOpen);
  const page = useAppStore((state) => state.settingsPage);
  const showPage = useAppStore((state) => state.showSettingsPage);
  const close = useAppStore((state) => state.closeSettings);
  const dialog = useModal(open);

  // Arrow keys move between the tabs, as in any tab list.
  const onTabKey = (event: KeyboardEvent<HTMLDivElement>) => {
    if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
    event.preventDefault();
    const index = pages.findIndex((each) => each.page === page);
    const next =
      pages[(index + (event.key === "ArrowRight" ? 1 : -1) + pages.length) % pages.length];
    if (!next) return;
    showPage(next.page);
    event.currentTarget.querySelector<HTMLElement>(`[data-page="${next.page}"]`)?.focus();
  };

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
      <div
        role="tablist"
        aria-label={t("privacy.tabs.label")}
        onKeyDown={onTabKey}
        className="mt-3 flex gap-1 border-b border-gray-200"
      >
        {pages.map((each) => (
          <button
            key={each.page}
            type="button"
            role="tab"
            id={`settings-tab-${each.page}`}
            data-page={each.page}
            data-testid={`settings-tab-${each.page}`}
            aria-selected={page === each.page}
            aria-controls={`settings-page-${each.page}`}
            tabIndex={page === each.page ? 0 : -1}
            onClick={() => showPage(each.page)}
            className={`-mb-px border-b-2 px-3 py-1.5 text-sm ${
              page === each.page
                ? "border-gray-800 font-medium text-gray-900"
                : "border-transparent text-gray-500 hover:text-gray-800"
            }`}
          >
            {t(each.label)}
          </button>
        ))}
      </div>
      {/* Rendered only while open, so each page loads fresh data. */}
      {open && (
        <div
          role="tabpanel"
          id={`settings-page-${page}`}
          aria-labelledby={`settings-tab-${page}`}
          className="mt-4"
        >
          {page === "general" ? <GeneralSettings /> : <PrivacySettings />}
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

function GeneralSettings() {
  const t = useT();
  const language = useAppStore((state) => state.settings?.user.language);
  const updateSettings = useAppStore((state) => state.updateSettings);
  return (
    <div className="flex flex-col gap-6">
      <ChatModelSettings />
      <EmbeddingSettingsSection />
      <RerankSettingsSection />
      <ConnectorsSettings />
      <SkillsSettings />
      <ApprovalsSettings />
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
      <DataFolderSettings />
      <ChatGptPlanSettings />
    </div>
  );
}

/** Settings → Data folder: where everything is kept, opened in the file manager for a backup. */
function DataFolderSettings() {
  const t = useT();
  const open = () =>
    files.openDataFolder().catch((failure: unknown) => {
      useAppStore.setState({ actionError: errorMessage(failure) });
    });
  return (
    <section>
      <h3 className="mb-1 text-sm font-medium">{t("export.dataFolder.title")}</h3>
      <p className="mb-2 text-sm text-gray-600">{t("export.dataFolder.body")}</p>
      <button
        type="button"
        data-testid="open-data-folder"
        onClick={() => void open()}
        className={buttonClass}
      >
        {t("export.dataFolder.open")}
      </button>
    </section>
  );
}
