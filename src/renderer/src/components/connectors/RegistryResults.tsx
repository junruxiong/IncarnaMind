import type { ConnectorRegistrySearch, ConnectorRegistryServer } from "../../../../core/api";
import { useT } from "../../i18n";
import { useAppStore } from "../../store";
import { buttonStyle } from "../ui";
import { ConnectorTile } from "./ConnectorTile";

/**
 * The MCP Registry's community servers for what the User typed, under the
 * reviewed ones: marked "Community · not reviewed", with why it was asked and
 * what that means. While it is asked, a line says so; a failure says why,
 * with what to do.
 */
export function RegistryResults({
  words,
  search,
  onOpen,
  onRetry,
}: {
  words: string;
  /** Null while the Registry is being asked. */
  search: ConnectorRegistrySearch | null;
  onOpen(server: ConnectorRegistryServer): void;
  onRetry(): void;
}) {
  const t = useT();
  return (
    <div data-testid="connectors-registry" className="flex flex-col gap-2">
      <div className="flex flex-wrap items-center gap-1.5">
        <h4 className="text-[13px] leading-5 font-semibold text-ink-secondary">
          {t("connectors.registry.title")}
        </h4>
        <span
          data-testid="connectors-registry-badge"
          className="rounded-sm bg-warning-wash px-1.5 text-[11px] leading-[18px] text-warning"
        >
          {t("connectors.registry.badge")}
        </span>
      </div>
      <p className="max-w-[600px] text-[12px] leading-[18px] text-ink-meta">
        {t("connectors.registry.note")}
      </p>
      {search === null ? (
        <p role="status" className="text-[13px] leading-5 text-ink-meta">
          {t("connectors.registry.searching")}
        </p>
      ) : search.outcome === "failed" ? (
        <div
          role="alert"
          data-testid="connectors-registry-problem"
          data-reason={search.reason}
          className="flex flex-wrap items-center gap-2 text-[13px] leading-5 text-ink"
        >
          <span>{t(`connectors.registry.failed.${search.reason}`)}</span>
          {search.reason === "declined" ? (
            <button
              type="button"
              onClick={() => useAppStore.getState().showSettingsPage("privacy")}
              className={buttonStyle("secondary", "sm")}
            >
              {t("connectors.registry.privacy")}
            </button>
          ) : (
            search.reason !== "local-mode" && (
              <button type="button" onClick={onRetry} className={buttonStyle("secondary", "sm")}>
                {t("connectors.registry.retry")}
              </button>
            )
          )}
        </div>
      ) : search.servers.length === 0 ? (
        <p className="text-[13px] leading-5 text-ink-meta">
          {t("connectors.registry.none", { query: words })}
        </p>
      ) : (
        <div className="grid grid-cols-1 gap-2 @[30rem]:grid-cols-2">
          {search.servers.map((server) => (
            <button
              key={server.id}
              type="button"
              data-testid="registry-server"
              data-registry-id={server.id}
              data-opens={`registry:${server.id}`}
              onClick={() => onOpen(server)}
              className="flex min-w-0 items-start gap-2.5 rounded-lg p-2.5 text-left shadow-[inset_0_0_0_1px_var(--color-rule)] hover:bg-frame hover:shadow-[inset_0_0_0_1px_var(--color-rule-strong)]"
            >
              <ConnectorTile name={server.name} />
              <span className="flex min-w-0 flex-1 flex-col">
                <span className="truncate text-[13px] leading-[18px] font-semibold text-ink">
                  {server.name}
                </span>
                <span className="line-clamp-2 text-[12px] leading-[17px] text-ink-meta">
                  {server.description}
                </span>
              </span>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
