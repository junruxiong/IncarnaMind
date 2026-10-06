import { useT } from "../i18n";
import { useAppStore } from "../store";
import { MindIcon, PlusIcon } from "./icons";
import { ChatReadinessNotice } from "./providers/ChatReadinessNotice";

/** The centre: the open Mind. Editing its Blocks arrives in a later ticket. */
export function MindPane() {
  const t = useT();
  const mind = useAppStore((state) => state.minds.find((each) => each.id === state.openMindId));
  const createMind = useAppStore((state) => state.createMind);
  const title = mind ? mind.title || t("mind.untitled") : "";

  return (
    <main data-testid="mind-area" className="flex min-w-[300px] flex-1 flex-col overflow-hidden">
      <div className="flex h-10 shrink-0 items-end">
        {mind && (
          <div
            title={title}
            className="relative ml-2 flex h-8 max-w-[220px] items-center rounded-t-[9px] bg-white pr-4 pl-8 text-sm text-gray-700"
          >
            <MindIcon className="absolute left-[10px] size-4" />
            <span className="truncate">{title}</span>
          </div>
        )}
      </div>

      <div className="flex-grow overflow-auto rounded-tl-[6px] bg-white">
        {mind ? (
          <article data-testid="mind-pane" data-mind-id={mind.id} className="mt-2 px-10">
            <h1
              data-testid="mind-title"
              className={`mt-4 p-3 text-3xl font-medium break-words ${mind.title ? "" : "text-gray-400"}`}
            >
              {title}
            </h1>
            <ChatReadinessNotice />
          </article>
        ) : (
          <div className="flex h-full flex-col items-center justify-center gap-2 px-10 text-center">
            <h1 className="text-xl font-medium text-gray-700">{t("mind.noneOpen.title")}</h1>
            <p className="text-sm text-gray-500">{t("mind.noneOpen.body")}</p>
            <button
              type="button"
              onClick={() => void createMind()}
              className="group mt-2 flex items-center gap-2 rounded-[9px] border border-gray-300 px-4 py-2 text-sm text-gray-700 hover:shadow-custom-unfocus"
            >
              <PlusIcon className="size-4 text-gray-500" />
              <span className="group-hover:text-gradient-mind">{t("sidebar.newMind")}</span>
            </button>
          </div>
        )}
      </div>
    </main>
  );
}
