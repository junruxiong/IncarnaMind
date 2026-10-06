import { useState } from "react";
import type { Mind } from "../../../core/api";
import { useT } from "../i18n";
import { useAppStore } from "../store";
import { DeleteMindDialog } from "./DeleteMindDialog";
import { DocumentsSection } from "./DocumentsSection";
import { GitHubIcon, LogoIcon, MindIcon, PlusIcon, SettingsIcon, TrashIcon } from "./icons";

const REPOSITORY_URL = "https://github.com/junruxiong/IncarnaMind";

const utilityButton =
  "flex flex-col items-center rounded-[9px] px-[7px] py-[3px] text-gray-800 hover:bg-gray-100";

export function Sidebar({ width, onOpenSettings }: { width: number; onOpenSettings(): void }) {
  const t = useT();
  const minds = useAppStore((state) => state.minds);
  const openMindId = useAppStore((state) => state.openMindId);
  const createMind = useAppStore((state) => state.createMind);
  const openMind = useAppStore((state) => state.openMind);
  const [confirmingDelete, setConfirmingDelete] = useState<Mind | null>(null);

  return (
    <aside
      aria-label={t("sidebar.label")}
      className="flex min-w-[165px] shrink flex-col bg-gray-50"
      style={{ flexBasis: width }}
    >
      <div className="mx-3 mt-3 mb-2 flex items-center gap-2">
        <LogoIcon className="size-[30px] shrink-0" />
        <span className="truncate font-medium text-gray-700">{t("app.name")}</span>
      </div>

      <button
        type="button"
        data-testid="new-mind"
        onClick={() => void createMind()}
        className="group mx-3 my-1 flex items-center gap-2 rounded-[9px] px-1 py-[5px] text-sm text-gray-600 hover:bg-gray-100"
      >
        <PlusIcon className="size-4 shrink-0 text-gray-500" />
        <span className="group-hover:text-gradient-mind">{t("sidebar.newMind")}</span>
      </button>

      <div className="hide-scrollbar flex-grow overflow-y-auto">
        <h2 className="mx-4 mt-3 mb-1 text-[11px] font-medium tracking-wide text-gray-400 uppercase">
          {t("sidebar.minds")}
        </h2>
        <nav aria-label={t("sidebar.minds")}>
          {minds.length === 0 ? (
            <p className="mx-4 py-[5px] text-sm text-gray-400">{t("sidebar.noMinds")}</p>
          ) : (
            <ul className="mx-3">
              {minds.map((mind) => {
                const isOpen = mind.id === openMindId;
                return (
                  <li key={mind.id} className="group relative">
                    <button
                      type="button"
                      data-testid="mind-list-item"
                      data-mind-id={mind.id}
                      aria-current={isOpen ? "page" : undefined}
                      onClick={() => openMind(mind.id)}
                      className={`my-[1px] flex w-full items-center gap-[6px] rounded-[9px] py-[5px] pr-7 pl-1 text-left text-sm ${
                        isOpen ? "bg-gray-200" : "hover:bg-gray-100"
                      }`}
                    >
                      <MindIcon className="size-4 shrink-0" />
                      <span
                        className={`truncate ${mind.title ? "text-gray-700" : "text-gray-500"}`}
                      >
                        {mind.title || t("mind.untitled")}
                      </span>
                    </button>
                    <button
                      type="button"
                      data-testid="delete-mind"
                      aria-label={t("mind.delete")}
                      title={t("mind.delete")}
                      onClick={() => setConfirmingDelete(mind)}
                      className="absolute top-1/2 right-1 -translate-y-1/2 rounded-[9px] p-[3px] text-gray-500 opacity-0 group-hover:opacity-100 hover:bg-gray-300 hover:text-gray-700 focus-visible:opacity-100"
                    >
                      <TrashIcon className="size-[14px]" />
                    </button>
                  </li>
                );
              })}
            </ul>
          )}
        </nav>
        <DocumentsSection />
      </div>

      <footer className="mx-2 my-4 flex items-center">
        <button type="button" onClick={onOpenSettings} className={utilityButton}>
          <SettingsIcon className="size-[25px]" />
          <span className="text-[11px]">{t("sidebar.settings")}</span>
        </button>
        <a href={REPOSITORY_URL} target="_blank" rel="noreferrer" className={utilityButton}>
          <GitHubIcon className="size-[25px]" />
          <span className="text-[11px]">{t("sidebar.github")}</span>
        </a>
      </footer>
      <DeleteMindDialog mind={confirmingDelete} onClose={() => setConfirmingDelete(null)} />
    </aside>
  );
}
