import { useCallback, useEffect, useState } from "react";
import type { DataFlowStatus } from "../../../../core/api";
import { core } from "../../core";
import { errorMessage } from "../../errors";
import { useT } from "../../i18n";
import { useAppStore } from "../../store";

/** Settings → Data sent to other services: every flow, the User's decision, and revoking it. */
export function ConsentSettings() {
  const t = useT();
  const [flows, setFlows] = useState<DataFlowStatus[] | null>(null);

  const refresh = useCallback(() => {
    core.listDataFlows().then(setFlows, () => undefined);
  }, []);

  useEffect(() => {
    refresh();
    const stopResolved = core.on("consent.resolved", refresh);
    const stopReadiness = core.on("chatReadiness.changed", refresh);
    // Setting Jev up or removing it moves the tagging flow to another service.
    const stopJev = core.on("jev.changed", refresh);
    return () => {
      stopResolved();
      stopReadiness();
      stopJev();
    };
  }, [refresh]);

  const revoke = async ({ flow }: DataFlowStatus) => {
    try {
      await core.revokeConsent(flow.id, flow.service.id);
    } catch (failure) {
      useAppStore.setState({ actionError: errorMessage(failure) });
    }
    refresh();
  };

  return (
    <section data-testid="consent-settings">
      <h3 className="mb-1 text-sm font-medium">{t("consent.settings.title")}</h3>
      {flows?.length === 0 && (
        <p className="text-sm text-gray-600">{t("consent.settings.empty")}</p>
      )}
      <ul className="flex flex-col gap-1">
        {flows?.map((status) => (
          <li
            key={`${status.flow.id}:${status.flow.service.id}`}
            className="flex items-center gap-3 text-sm"
          >
            <span className="flex-1">
              {t(`consent.flow.${status.flow.id}`)} → {status.flow.service.name}
            </span>
            <span className="text-gray-500">{t(`consent.settings.status.${status.consent}`)}</span>
            {status.consent !== "not-asked" && (
              <button
                type="button"
                onClick={() => void revoke(status)}
                className="rounded-[9px] px-2 py-1 text-gray-700 hover:bg-gray-100"
              >
                {status.consent === "accepted"
                  ? t("consent.settings.revoke")
                  : t("consent.settings.askAgain")}
              </button>
            )}
          </li>
        ))}
      </ul>
    </section>
  );
}
