import { useState } from "react";
import type { DataFlowId, RegisteredDataFlow } from "../../../core/api";
import { formatDate } from "../../../shared/dates";
import { core } from "../core";
import { useLanguage, useT } from "../i18n";
import { decisionOf, groupByService, type ServiceGroup, type ServiceUse } from "../privacyServices";
import { useAppStore } from "../store";
import { useUndo } from "../undo";
import { DetailsLine } from "./DetailsLine";
import { UndoLineAt } from "./UndoLine";
import { buttonStyle, rowButtonsClass, rowTextClass, ruledListClass, ruledRowClass } from "./ui";

const SERVICES_UNDO_PLACE = "settings:privacy:services";

const reportError = (failure: unknown) => useAppStore.getState().reportError(failure);

interface Revoked {
  flow: DataFlowId;
  at: string;
}

async function revokeAll(service: string, uses: readonly ServiceUse[]): Promise<Revoked[]> {
  const revoked: Revoked[] = [];
  for (const use of uses) {
    const at = await core.revokeConsent(use.flow, service);
    if (at) revoked.push({ flow: use.flow, at });
  }
  return revoked;
}

async function restoreAll(service: string, revoked: readonly Revoked[]): Promise<void> {
  for (const { flow, at } of revoked) await core.restoreConsent(flow, service, at);
}

async function allowAll(service: string, uses: readonly ServiceUse[]): Promise<void> {
  for (const use of uses) await core.allowDataFlow(use.flow, service);
}

export function PrivacyServices({ flows }: { flows: readonly RegisteredDataFlow[] }) {
  return (
    <div className="flex flex-col">
      <ul className={ruledListClass}>
        {groupByService(flows).map((group) => (
          <ServiceItem key={group.service.id} group={group} />
        ))}
      </ul>
      <UndoLineAt place={SERVICES_UNDO_PLACE} />
    </div>
  );
}

/** A button keeps its slot as the decision changes, so the focus stays on it. */
interface Action {
  slot: "first" | "second";
  label: string;
  testId: string;
  primary: boolean;
  run: () => Promise<void>;
}

function ServiceItem({ group }: { group: ServiceGroup }) {
  const t = useT();
  const language = useLanguage();
  const [busy, setBusy] = useState(false);
  const { service, uses } = group;
  const decision = decisionOf(uses);

  const run = async (action: () => Promise<unknown>) => {
    if (busy) return;
    setBusy(true);
    try {
      await action();
    } catch (failure) {
      reportError(failure);
    } finally {
      setBusy(false);
    }
  };
  const withConsent = (consent: ServiceUse["status"]["consent"]) =>
    uses.filter((use) => use.status.consent === consent);
  const stop = () =>
    run(async () => {
      const allowed = withConsent("accepted");
      let revoked = await revokeAll(service.id, allowed);
      useUndo.getState().push({
        said: (words) => words("privacy.flows.stopped", { service: service.name }),
        undo: () => restoreAll(service.id, revoked),
        redo: async () => {
          revoked = await revokeAll(service.id, allowed);
        },
        shownAt: SERVICES_UNDO_PLACE,
      });
    });
  const allow = () =>
    run(() =>
      allowAll(
        service.id,
        uses.filter((use) => use.status.consent !== "accepted"),
      ),
    );
  const askAgain = () => run(() => revokeAll(service.id, withConsent("declined")));

  const when = (at: string | null) => (at ? formatDate(at, language) : null);
  const line = (() => {
    switch (decision.kind) {
      case "accepted": {
        const date = when(decision.at);
        if (!date) return t("consent.settings.status.accepted");
        return decision.chosen
          ? t("privacy.flows.chosen", { service: service.name, date })
          : t("privacy.flows.decided", { status: t("consent.settings.status.accepted"), date });
      }
      case "declined": {
        const date = when(decision.at);
        const status = t("consent.settings.status.declined");
        return date ? t("privacy.flows.decided", { status, date }) : status;
      }
      case "partly":
        return t("privacy.flows.partly");
      case "sendsMore":
        return t("consent.settings.status.sendsMore");
      case "not-asked":
        return t("consent.settings.status.not-asked");
    }
  })();

  const allowUse = (use: ServiceUse) => void run(() => allowAll(service.id, [use]));

  const actions: Action[] = [];
  if (decision.kind === "accepted") {
    actions.push({
      slot: "first",
      label: t("consent.settings.revoke"),
      testId: "data-service-revoke",
      primary: false,
      run: stop,
    });
  } else {
    actions.push({
      slot: "first",
      label: t("privacy.flows.allow"),
      testId: "data-service-allow",
      primary: true,
      run: allow,
    });
    if (decision.kind === "partly") {
      actions.push({
        slot: "second",
        label: t("consent.settings.revoke"),
        testId: "data-service-revoke",
        primary: false,
        run: stop,
      });
    } else if (decision.kind === "declined") {
      actions.push({
        slot: "second",
        label: t("consent.settings.askAgain"),
        testId: "data-service-ask-again",
        primary: false,
        run: askAgain,
      });
    }
  }

  return (
    <li
      data-testid="data-service"
      data-service-id={service.id}
      data-consent={decision.kind}
      className={ruledRowClass}
    >
      <div className="flex min-w-0 flex-col gap-0.5">
        <p className="text-ui break-words font-semibold text-ink">{service.name}</p>
        <DetailsLine
          testId="data-service-decision"
          className={rowTextClass}
          details={<Uses uses={uses} partly={decision.kind === "partly"} onAllow={allowUse} />}
        >
          {line}
        </DetailsLine>
      </div>
      <div className={rowButtonsClass}>
        {actions.map((action) => (
          <button
            key={action.slot}
            type="button"
            data-testid={action.testId}
            aria-busy={busy}
            onClick={() => void action.run()}
            className={buttonStyle(action.primary ? "primary" : "secondary", "sm")}
          >
            {action.label}
          </button>
        ))}
      </div>
    </li>
  );
}

const waitingStatus = (use: ServiceUse) =>
  use.status.consent === "declined"
    ? "consent.settings.status.declined"
    : use.status.decidedAt !== null
      ? "consent.settings.status.sendsMore"
      : "consent.settings.status.not-asked";

function Uses({
  uses,
  partly,
  onAllow,
}: {
  uses: readonly ServiceUse[];
  partly: boolean;
  onAllow(use: ServiceUse): void;
}) {
  const t = useT();
  return (
    <ul className="flex flex-col gap-2 pt-1.5">
      {uses.map((use) => {
        const waiting = partly && use.status.consent !== "accepted";
        return (
          <li
            key={use.flow}
            data-testid="data-flow"
            data-flow-id={use.flow}
            data-consent={use.status.consent}
          >
            <p className="font-semibold text-ink">{t(`consent.flow.${use.flow}`)}</p>
            <ul aria-label={t("privacy.flows.sends")} className="list-disc pl-4">
              {use.sends.map((kind) => (
                <li key={kind}>{t(`consent.data.${kind}`)}</li>
              ))}
            </ul>
            {waiting && (
              <p className="mt-1 flex flex-wrap items-center gap-x-3 gap-y-1">
                <span className="text-ink-meta">{t(waitingStatus(use))}</span>
                <button
                  type="button"
                  data-testid="data-flow-allow"
                  onClick={() => onAllow(use)}
                  className={buttonStyle("secondary", "sm")}
                >
                  {t("privacy.flows.allow")}
                </button>
              </p>
            )}
          </li>
        );
      })}
    </ul>
  );
}
