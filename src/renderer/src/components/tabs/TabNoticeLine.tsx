import { useEffect } from "react";
import type { TabItem } from "../../../../core/api";
import { useT } from "../../i18n";
import { useAppStore } from "../../store";
import { CloseLineIcon } from "../lineIcons";
import { SPLIT_SHORTCUT } from "./tabActions";
import { useTabTitle } from "./tabItems";

/** How long a line about the tabs stays, unless dismissed or used first. */
const SHOWN_FOR_MS = 8000;

/**
 * One line about the tabs, above the sidebar's foot: a tab closed (with
 * Reopen, as ⇧⌘T does), or why a shortcut did nothing. It goes by itself
 * after a few seconds; nothing here is an error.
 */
export function TabNoticeLine() {
  const t = useT();
  const notice = useAppStore((state) => state.tabNotice);
  const showTabNotice = useAppStore((state) => state.showTabNotice);
  const reopenTab = useAppStore((state) => state.reopenTab);

  useEffect(() => {
    if (!notice) return;
    const timer = setTimeout(() => showTabNotice(null), SHOWN_FOR_MS);
    return () => clearTimeout(timer);
  }, [notice, showTabNotice]);

  if (!notice) return null;
  return (
    <div role="status" data-testid="tab-notice" data-kind={notice.kind} className="tab-notice">
      <span className="min-w-0 flex-1">
        {notice.kind === "closed" ? (
          notice.count > 1 ? (
            t("tabs.notice.closedMany", { count: notice.count })
          ) : (
            <Titled item={notice.item} message="tabs.notice.closed" />
          )
        ) : notice.kind === "pinned" ? (
          <Titled item={notice.item} message="tabs.notice.pinned" />
        ) : notice.kind === "noGroup" ? (
          t("tabs.notice.noGroup", { n: notice.group, shortcut: SPLIT_SHORTCUT })
        ) : notice.kind === "maxGroups" ? (
          t("tabs.notice.maxGroups")
        ) : (
          t("tabs.notice.merged")
        )}
      </span>
      {notice.kind === "closed" && (
        <button
          type="button"
          data-testid="tab-notice-reopen"
          onClick={reopenTab}
          className="tab-notice-action"
        >
          {t("tabs.notice.reopen")}
        </button>
      )}
      <button
        type="button"
        aria-label={t("tabs.notice.dismiss")}
        title={t("tabs.notice.dismiss")}
        onClick={() => showTabNotice(null)}
        className="tab-notice-close"
      >
        <CloseLineIcon />
      </button>
    </div>
  );
}

function Titled({
  item,
  message,
}: {
  item: TabItem;
  message: "tabs.notice.closed" | "tabs.notice.pinned";
}) {
  const t = useT();
  const title = useTabTitle(item);
  return <>{t(message, { title })}</>;
}
