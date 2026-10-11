import { createContext, useContext } from "react";
import type { TabItem } from "../../../../core/api";
import { useT } from "../../i18n";
import { useMindStatus } from "../../mindStatus";
import { useAppStore } from "../../store";
import { tabTitle } from "../../tabs";
import { DocumentLineIcon, LooseDocumentsLineIcon, MindLineIcon } from "../lineIcons";

/** The tab a piece of UI is in, and its group: a Citation in a Mind opens beside that group. */
export interface TabPlace {
  tabId: string;
  groupId: string;
}

export const TabPlaceContext = createContext<TabPlace | null>(null);

/** The tab this is drawn in, if it is in one. */
export const useTabPlace = () => useContext(TabPlaceContext);

export function useTabTitle(item: TabItem): string {
  const t = useT();
  return useAppStore((state) => tabTitle(state, item, t));
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
