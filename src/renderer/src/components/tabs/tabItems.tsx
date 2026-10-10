import { createContext, useContext } from "react";
import type { TabItem } from "../../../../core/api";
import { displayName } from "../../../../shared/documentNames";
import { useT } from "../../i18n";
import { useMindStatus } from "../../mindStatus";
import { useAppStore } from "../../store";
import { DocumentLineIcon, LooseDocumentsLineIcon, MindLineIcon } from "../lineIcons";

/** The tab a piece of UI is in, and its group: a Citation in a Mind opens beside that group. */
export interface TabPlace {
  tabId: string;
  groupId: string;
}

export const TabPlaceContext = createContext<TabPlace | null>(null);

/** The tab this is drawn in, if it is in one. */
export const useTabPlace = () => useContext(TabPlaceContext);

/** What the Library's tab is called: the Folder it shows, or "All Documents". */
function useLibraryTitle(): string {
  const t = useT();
  return useAppStore((state) => {
    const filter = state.libraryFilter;
    if (filter === "unsorted") return t("library.unsorted");
    const folder = state.library?.groups.find((group) => group.id === filter);
    return folder?.name ?? t("library.all");
  });
}

/** What a tab is called: a Mind's title, the Library's Folder, or a Document's name only. */
export function useTabTitle(item: TabItem): string {
  const t = useT();
  const library = useLibraryTitle();
  return useAppStore((state) => {
    if (item.kind === "library") return library;
    if (item.kind === "mind") {
      return state.minds.find((mind) => mind.id === item.mindId)?.title || t("mind.untitled");
    }
    const document = state.documents.find((each) => each.id === item.documentId);
    return document ? displayName(document) : t("viewer.label");
  });
}

/** A tab's icon: a Mind's (or what it is busy with), the Library's, or the Document's kind. */
export function TabIcon({ item }: { item: TabItem }) {
  const kind = useAppStore((state) =>
    item.kind === "document"
      ? (state.documents.find((each) => each.id === item.documentId)?.kind ?? "pdf")
      : null,
  );
  if (item.kind === "mind") return <MindIcon mindId={item.mindId} />;
  if (item.kind === "library") return <LooseDocumentsLineIcon />;
  return <DocumentLineIcon kind={kind ?? "pdf"} />;
}

function MindIcon({ mindId }: { mindId: string }) {
  const t = useT();
  const status = useMindStatus(mindId);
  if (status === null) return <MindLineIcon />;
  return (
    <span
      role="img"
      data-testid="mind-tab-status"
      data-status={status}
      aria-label={status === "writing" ? t("tabs.writing") : t("approvals.answer.waiting")}
      className="tab-dot"
    />
  );
}
