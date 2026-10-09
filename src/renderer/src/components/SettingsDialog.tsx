import { type ReactNode, useState } from "react";
import type { LanguagePreference } from "../../../core/language";
import type { MessageKey } from "../../../shared/i18n";
import { files } from "../core";
import { errorMessage } from "../errors";
import { useT } from "../i18n";
import { type SettingsPage, settingsPages, useAppStore } from "../store";
import { ApprovalsSettings } from "./ApprovalsSettings";
import { ConnectorsSettings } from "./connectors/ConnectorsSettings";
import { CloseLineIcon } from "./lineIcons";
import { OrganizationSettings } from "./OrganizationSettings";
import { PrivacySettings } from "./PrivacySettings";
import { ChatGptPlanSettings } from "./providers/ChatGptPlanSettings";
import { ChatModelSettings } from "./providers/ChatModelSettings";
import { EmbeddingSettingsSection } from "./providers/EmbeddingSettings";
import { JevSettingsSection } from "./providers/JevSettings";
import { RerankSettingsSection } from "./providers/RerankSettings";
import { SkillScriptsSettings, SkillsSettings } from "./SkillsSettings";
import {
  buttonClass,
  dialogFrameClass,
  hintClass,
  navRowClass,
  ruledListClass,
  sectionNoteClass,
  sectionTitleClass,
} from "./ui";
import { useModal } from "./useModal";

const languageOptions: readonly { value: LanguagePreference; label: MessageKey }[] = [
  { value: "system", label: "settings.language.system" },
  { value: "en", label: "settings.language.en" },
  { value: "zh-CN", label: "settings.language.zh-CN" },
];

/** Each page's name, in the list and as its title. */
const pageLabels: Record<SettingsPage, MessageKey> = {
  general: "privacy.tabs.general",
  "chat-model": "providers.settings.title",
  search: "embeddingProviders.settings.title",
  organization: "library.settingsTitle",
  connectors: "connectors.settings.title",
  skills: "skills.settings.title",
  approvals: "approvals.settings.title",
  privacy: "privacy.tabs.privacy",
};

const REPOSITORY_URL = "https://github.com/junruxiong/IncarnaMind";

/**
 * Settings, in a native modal <dialog>: a list of pages on the frame on the
 * left (General, Chat model, Document search, Connectors, Skills, Approvals,
 * Privacy), and the chosen page on the right under its serif title.
 */
export function SettingsDialog() {
  const t = useT();
  const open = useAppStore((state) => state.settingsOpen);
  const page = useAppStore((state) => state.settingsPage);
  const showPage = useAppStore((state) => state.showSettingsPage);
  const close = useAppStore((state) => state.closeSettings);
  const dialog = useModal(open);
  /** The pages shown since Settings opened. Closing forgets them. */
  const [visited, setVisited] = useState<readonly SettingsPage[]>([]);
  if (open && !visited.includes(page)) setVisited([...visited, page]);
  if (!open && visited.length > 0) setVisited([]);

  return (
    <dialog
      ref={dialog}
      onClose={close}
      data-testid="settings"
      data-page={page}
      aria-labelledby="settings-title"
      className={`${dialogFrameClass} h-[min(740px,calc(100vh-48px))] max-h-none w-[min(880px,calc(100vw-48px))] max-w-none overflow-hidden`}
    >
      <div className="flex h-full min-h-0">
        <nav
          aria-labelledby="settings-title"
          data-testid="settings-nav"
          className="flex w-[200px] shrink-0 flex-col overflow-y-auto border-r border-rule bg-frame p-2"
        >
          <div className="flex h-11 shrink-0 items-center px-2">
            <h2 id="settings-title" className="text-ui font-semibold text-ink">
              {t("settings.title")}
            </h2>
          </div>
          {settingsPages.map((each) => (
            <button
              key={each}
              type="button"
              data-testid={`settings-nav-${each}`}
              aria-current={page === each ? "page" : undefined}
              onClick={() => showPage(each)}
              className={navRowClass(page === each)}
            >
              {t(pageLabels[each])}
            </button>
          ))}
        </nav>
        <section aria-labelledby="settings-page-title" className="flex min-w-0 flex-1 flex-col">
          <header className="flex h-[60px] shrink-0 items-center justify-between gap-3 pt-2 pr-4 pl-8">
            <h3
              id="settings-page-title"
              data-testid="settings-page-title"
              className="font-serif text-heading font-semibold text-ink [font-variation-settings:'opsz'_32]"
            >
              {t(pageLabels[page])}
            </h3>
            <button
              type="button"
              data-testid="settings-close"
              aria-label={t("settings.close")}
              title={t("settings.close")}
              onClick={close}
              className="inline-flex size-8 shrink-0 items-center justify-center rounded-md text-ink-secondary outline-none hover:bg-chip hover:text-ink focus-visible:outline-2 focus-visible:outline-accent"
            >
              <CloseLineIcon className="size-4" />
            </button>
          </header>
          {/*
           * Rendered only while open, so each opening loads fresh data. A page
           * stays rendered (hidden) once shown, so a form half filled in, or
           * Tools unfolded, are still there after a look at another page.
           */}
          <div className="relative min-h-0 flex-1">
            {visited.map((each) => (
              <div
                key={each}
                data-testid={`settings-page-${each}`}
                hidden={each !== page}
                className="absolute inset-0 overflow-y-auto px-8 pt-1 pb-8"
              >
                <PageContent page={each} />
              </div>
            ))}
          </div>
        </section>
      </div>
    </dialog>
  );
}

function PageContent({ page }: { page: SettingsPage }) {
  switch (page) {
    case "general":
      return (
        <Page>
          <LanguageSettings />
          <DataFolderSettings />
          <AboutSettings />
        </Page>
      );
    case "chat-model":
      return (
        <Page>
          <ChatModelSettings />
          <JevSettingsSection />
          <ChatGptPlanSettings />
        </Page>
      );
    case "search":
      return (
        <Page>
          <EmbeddingSettingsSection />
          <RerankSettingsSection />
        </Page>
      );
    case "organization":
      return <OrganizationSettings />;
    case "connectors":
      return (
        <Page>
          <ConnectorsSettings />
        </Page>
      );
    case "skills":
      return (
        <Page>
          <SkillsSettings />
          <SkillScriptsSettings />
        </Page>
      );
    case "approvals":
      return (
        <Page>
          <ApprovalsSettings />
        </Page>
      );
    case "privacy":
      return (
        <Page>
          <PrivacySettings />
        </Page>
      );
  }
}

/** A page's sections, 28px apart. */
function Page({ children }: { children: ReactNode }) {
  return <div className="flex flex-col gap-7">{children}</div>;
}

/** Settings → General → Interface language: a ruled list of choices. */
function LanguageSettings() {
  const t = useT();
  const language = useAppStore((state) => state.settings?.user.language);
  const updateSettings = useAppStore((state) => state.updateSettings);
  return (
    <section data-testid="language-settings">
      <h4 id="language-title" className={`mb-2 ${sectionTitleClass}`}>
        {t("settings.language")}
      </h4>
      <div role="radiogroup" aria-labelledby="language-title" className={ruledListClass}>
        {languageOptions.map((option) => (
          <label
            key={option.value}
            className="flex h-10 cursor-pointer items-center gap-3 text-ui text-ink"
          >
            <input
              type="radio"
              name="language"
              value={option.value}
              checked={language === option.value}
              onChange={() => void updateSettings({ user: { language: option.value } })}
              className="size-4 shrink-0 accent-ink"
            />
            {t(option.label)}
          </label>
        ))}
      </div>
    </section>
  );
}

/**
 * Settings → General → Data folder: where everything is kept, opened in the
 * file manager for a backup, and its logs folder, e.g. for a bug report.
 */
function DataFolderSettings() {
  const t = useT();
  const open = (folder: Promise<void>) =>
    folder.catch((failure: unknown) => {
      useAppStore.setState({ actionError: errorMessage(failure) });
    });
  return (
    <section data-testid="data-folder-settings">
      <h4 className={sectionTitleClass}>{t("export.dataFolder.title")}</h4>
      <p className={sectionNoteClass}>{t("export.dataFolder.body")}</p>
      <div className="flex flex-wrap gap-2">
        <button
          type="button"
          data-testid="open-data-folder"
          onClick={() => void open(files.openDataFolder())}
          className={buttonClass}
        >
          {t("export.dataFolder.open")}
        </button>
        <button
          type="button"
          data-testid="open-logs-folder"
          onClick={() => void open(files.openLogsFolder())}
          className={buttonClass}
        >
          {t("export.dataFolder.openLogs")}
        </button>
      </div>
      <p className={hintClass}>{t("export.dataFolder.logs")}</p>
    </section>
  );
}

/** Settings → General → About: where IncarnaMind's source lives. */
function AboutSettings() {
  const t = useT();
  return (
    <section data-testid="about-settings">
      <h4 className={sectionTitleClass}>{t("settings.about.title")}</h4>
      <p className={sectionNoteClass}>{t("settings.about.body")}</p>
      <a
        href={REPOSITORY_URL}
        target="_blank"
        rel="noreferrer"
        className="text-ui text-accent underline-offset-2 hover:text-accent-strong hover:underline"
      >
        {t("settings.about.repository")}
      </a>
    </section>
  );
}
