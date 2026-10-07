import { useEffect, useState } from "react";
import type { ConsentRequest } from "../../../core/api";
import { core } from "../core";
import { errorMessage } from "../errors";
import { useT } from "../i18n";
import { useAppStore } from "../store";
import { buttonClass, primaryButtonClass } from "./providers/shared";
import { useModal } from "./useModal";

/**
 * Asks before data leaves the machine: the core raises a consent request
 * before the first request on a flow, and waits. It lists what the flow sends
 * and to whom. The User must choose; Esc doesn't dismiss it.
 */
export function ConsentDialog() {
  const t = useT();
  const [queue, setQueue] = useState<ConsentRequest[]>([]);

  useEffect(() => {
    const add = (requests: readonly ConsentRequest[]) =>
      setQueue((current) => [
        ...current,
        ...requests.filter((request) =>
          current.every((queued) => queued.requestId !== request.requestId),
        ),
      ]);
    const stopRequested = core.on("consent.requested", (request) => add([request]));
    const stopResolved = core.on("consent.resolved", ({ requestId }) =>
      setQueue((current) => current.filter((request) => request.requestId !== requestId)),
    );
    // Requests raised before this window was listening.
    core.listConsentRequests().then(add, () => undefined);
    return () => {
      stopRequested();
      stopResolved();
    };
  }, []);

  const request = queue[0];
  const dialog = useModal(request !== undefined);

  const respond = async (accept: boolean) => {
    if (!request) return;
    setQueue((current) => current.filter((queued) => queued.requestId !== request.requestId));
    try {
      await core.respondToConsent(request.requestId, accept);
    } catch (failure) {
      useAppStore.setState({ actionError: errorMessage(failure) });
    }
  };

  const service = request?.flow.service.name ?? "";
  const purpose = request ? t(`consent.flow.${request.flow.id}.purpose`) : "";
  const isFirstAsk = request !== undefined && request.newKinds.length === request.flow.sends.length;

  return (
    <dialog
      ref={dialog}
      data-testid="consent-dialog"
      aria-labelledby="consent-title"
      onCancel={(event) => event.preventDefault()}
      className="m-auto w-[30rem] max-w-[calc(100vw-2rem)] rounded-[9px] bg-white p-5 text-gray-800 shadow-custom-focus backdrop:bg-black/30"
    >
      {request && (
        <>
          <h2 id="consent-title" className="text-lg font-semibold">
            {t("consent.dialog.title", { service })}
          </h2>
          <p className="mt-2 text-sm text-gray-700">
            {t(isFirstAsk ? "consent.dialog.body" : "consent.dialog.bodyMore", {
              purpose,
              service,
            })}
          </p>
          <ul className="mt-2 list-disc pl-5 text-sm text-gray-700">
            {request.newKinds.map((kind) => (
              <li key={kind}>{t(`consent.data.${kind}`)}</li>
            ))}
          </ul>
          {request.flow.id === "connectors" && (
            <p className="mt-3 text-sm text-gray-700">
              {/* A local Connector's service is "connector:<id>"; a remote one's is its server's origin. */}
              {request.flow.service.id.startsWith("connector:")
                ? t("connectors.consent.note")
                : t("remoteConnectors.consent.note")}
            </p>
          )}
          <p className="mt-3 text-xs text-gray-500">{t("consent.dialog.note")}</p>
          <div className="mt-4 flex justify-end gap-2">
            <button
              type="button"
              data-testid="consent-decline"
              onClick={() => void respond(false)}
              className={buttonClass}
            >
              {t("consent.dialog.decline")}
            </button>
            <button
              type="button"
              data-testid="consent-allow"
              onClick={() => void respond(true)}
              className={primaryButtonClass}
            >
              {t("consent.dialog.allow")}
            </button>
          </div>
        </>
      )}
    </dialog>
  );
}
