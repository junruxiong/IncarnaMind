import { useCallback, useEffect, useId, useState } from "react";
import type {
  DataFlowStatus,
  NetworkTraffic,
  PrivacySettings as PrivacyChoices,
  PrivacySettingsPatch,
  RegisteredDataFlow,
} from "../../../core/api";
import { core } from "../core";
import { errorMessage } from "../errors";
import { useLanguage, useT } from "../i18n";
import { useAppStore } from "../store";

const reportError = (failure: unknown) =>
  useAppStore.setState({ actionError: errorMessage(failure) });

const smallButtonClass =
  "shrink-0 rounded-[9px] px-2 py-1 text-gray-700 hover:bg-gray-100 disabled:opacity-50";

/**
 * Settings → Privacy: everything IncarnaMind sends from this computer, in one
 * place. Every registered data flow with its consent (revoke, allow), network
 * traffic that carries nothing of the User's (with the update-check switch),
 * what Skill scripts can do, and crash reports when this copy can send them.
 */
export function PrivacySettings() {
  const t = useT();
  const [flows, setFlows] = useState<RegisteredDataFlow[] | null>(null);
  const [traffic, setTraffic] = useState<NetworkTraffic[] | null>(null);
  const [choices, setChoices] = useState<PrivacyChoices | null>(null);

  const refreshFlows = useCallback(() => {
    core.listRegisteredDataFlows().then(setFlows, reportError);
  }, []);
  const refreshTraffic = useCallback(() => {
    core.listNetworkTraffic().then(setTraffic, reportError);
  }, []);

  useEffect(() => {
    refreshFlows();
    refreshTraffic();
    core.getPrivacySettings().then(setChoices, reportError);
    const stops = [
      core.on("dataFlows.changed", setFlows),
      core.on("consent.resolved", refreshFlows),
      // A new provider or tagger moves a flow to another service.
      core.on("chatReadiness.changed", refreshFlows),
      core.on("jev.changed", refreshFlows),
      // So does switching the embedding model, or setting rerank up.
      core.on("embedding.changed", refreshFlows),
      core.on("rerank.changed", refreshFlows),
      // Each Connector that is on is a service of the "connectors" flow.
      core.on("connectors.changed", refreshFlows),
      core.on("privacy.changed", (changed) => {
        setChoices(changed);
        refreshTraffic();
      }),
      core.on("chatGptPlan.changed", refreshTraffic),
    ];
    return () => {
      for (const stop of stops) stop();
    };
  }, [refreshFlows, refreshTraffic]);

  const update = async (patch: PrivacySettingsPatch) => {
    // The switch moves at once; the core's answer then confirms it, or puts it back.
    setChoices(
      (current) =>
        current && {
          crashReports: {
            ...current.crashReports,
            enabled: patch.crashReports ?? current.crashReports.enabled,
          },
          automaticUpdateChecks: patch.automaticUpdateChecks ?? current.automaticUpdateChecks,
        },
    );
    try {
      setChoices(await core.updatePrivacySettings(patch));
    } catch (failure) {
      reportError(failure);
      core.getPrivacySettings().then(setChoices, () => undefined);
    }
    refreshTraffic();
  };

  return (
    <div data-testid="privacy-settings" className="flex flex-col gap-6">
      <p className="text-sm text-gray-600">{t("privacy.intro")}</p>

      <section data-testid="consent-settings">
        <h3 className="mb-1 text-sm font-medium">{t("consent.settings.title")}</h3>
        <p className="mb-2 text-xs text-gray-500">{t("privacy.flows.intro")}</p>
        <ul className="flex flex-col gap-3">
          {flows?.map((flow) => (
            <DataFlowItem key={flow.id} flow={flow} />
          ))}
        </ul>
      </section>

      <section data-testid="network-traffic">
        <h3 className="mb-1 text-sm font-medium">{t("privacy.traffic.title")}</h3>
        <p className="mb-2 text-xs text-gray-500">{t("privacy.traffic.intro")}</p>
        <ul className="flex flex-col gap-3">
          {traffic?.map((item) => (
            <TrafficItem
              key={`${item.id} ${item.service.id}`}
              traffic={item}
              choices={choices}
              onChange={(patch) => void update(patch)}
            />
          ))}
        </ul>
      </section>

      <section data-testid="skill-scripts-note" role="note">
        <h3 className="mb-1 text-sm font-medium">{t("privacy.skills.title")}</h3>
        <p className="text-sm text-gray-600">{t("privacy.skills.body")}</p>
      </section>

      {/* Only a copy built with a crash-report address can send reports, so only it offers them. */}
      {choices?.crashReports.available && (
        <CrashReports
          enabled={choices.crashReports.enabled}
          onChange={(enabled) => void update({ crashReports: enabled })}
        />
      )}
    </div>
  );
}

/** A data flow, what it sends, and the User's decision for each service it goes to. */
function DataFlowItem({ flow }: { flow: RegisteredDataFlow }) {
  const t = useT();
  return (
    <li data-testid="data-flow" data-flow-id={flow.id} className="text-sm">
      <p className="font-medium text-gray-800">{t(`consent.flow.${flow.id}`)}</p>
      <p className="text-xs text-gray-500">{t("privacy.flows.sends")}</p>
      <ul className="list-disc pl-5 text-xs text-gray-600">
        {flow.sends.map((kind) => (
          <li key={kind}>{t(`consent.data.${kind}`)}</li>
        ))}
      </ul>
      {flow.services.length === 0 ? (
        <p data-testid="data-flow-not-in-use" className="mt-1 text-xs text-gray-500">
          {t("privacy.flows.notInUse")}
        </p>
      ) : (
        <ul className="mt-1 flex flex-col gap-1">
          {flow.services.map((status) => (
            <ServiceDecision key={status.flow.service.id} status={status} />
          ))}
        </ul>
      )}
    </li>
  );
}

/** One service of a flow: the decision and when it was made, with revoke and allow. */
function ServiceDecision({ status }: { status: DataFlowStatus }) {
  const t = useT();
  const language = useLanguage();
  const [busy, setBusy] = useState(false);
  const { flow, consent, decidedAt } = status;

  const run = async (action: () => Promise<void>) => {
    setBusy(true);
    try {
      await action();
    } catch (failure) {
      reportError(failure);
    } finally {
      setBusy(false);
    }
  };
  const revoke = () => run(() => core.revokeConsent(flow.id, flow.service.id));
  const allow = () => run(() => core.allowDataFlow(flow.id, flow.service.id));

  const label = t(`consent.settings.status.${consent}`);
  const decision = decidedAt
    ? t("privacy.flows.decided", {
        status: label,
        date: new Intl.DateTimeFormat(language, { dateStyle: "medium" }).format(
          new Date(decidedAt),
        ),
      })
    : label;

  return (
    <li
      data-testid="data-flow-service"
      data-service-id={flow.service.id}
      data-consent={consent}
      className="flex flex-wrap items-center gap-x-3 gap-y-1 rounded-[9px] bg-gray-50 px-2 py-1"
    >
      <span className="min-w-0 flex-1 break-words">
        {t("privacy.flows.service", { service: flow.service.name })}
      </span>
      <span data-testid="data-flow-decision" className="text-xs text-gray-500">
        {decision}
      </span>
      {consent === "accepted" ? (
        <button
          type="button"
          data-testid="data-flow-revoke"
          disabled={busy}
          onClick={() => void revoke()}
          className={smallButtonClass}
        >
          {t("consent.settings.revoke")}
        </button>
      ) : (
        <button
          type="button"
          data-testid="data-flow-allow"
          disabled={busy}
          onClick={() => void allow()}
          className={smallButtonClass}
        >
          {t("privacy.flows.allow")}
        </button>
      )}
      {consent === "declined" && (
        <button
          type="button"
          data-testid="data-flow-ask-again"
          disabled={busy}
          onClick={() => void revoke()}
          className={smallButtonClass}
        >
          {t("consent.settings.askAgain")}
        </button>
      )}
    </li>
  );
}

/** Traffic without the User's content: where it goes, and the switch for update checks. */
function TrafficItem({
  traffic,
  choices,
  onChange,
}: {
  traffic: NetworkTraffic;
  choices: PrivacyChoices | null;
  onChange(patch: PrivacySettingsPatch): void;
}) {
  const t = useT();
  const id = useId();
  return (
    <li
      data-testid="network-traffic-item"
      data-traffic-id={traffic.id}
      data-enabled={traffic.enabled}
      className="text-sm"
    >
      <p className="flex items-center gap-2">
        <span className="flex-1 font-medium text-gray-800">
          {t(`privacy.traffic.${traffic.id}`)}
          <span className="font-normal text-gray-500"> · {traffic.service.name}</span>
        </span>
        <span className="text-xs text-gray-500">
          {traffic.enabled ? t("privacy.traffic.on") : t("privacy.traffic.off")}
        </span>
      </p>
      <p id={`${id}-description`} className="text-xs text-gray-600">
        {t(`privacy.traffic.${traffic.id}.description`)}
      </p>
      {traffic.id === "update-check" && choices && (
        <label className="mt-1 flex items-center gap-2 text-sm">
          <input
            type="checkbox"
            role="switch"
            data-testid="automatic-update-checks"
            aria-describedby={`${id}-description`}
            checked={choices.automaticUpdateChecks}
            aria-checked={choices.automaticUpdateChecks}
            onChange={(event) => onChange({ automaticUpdateChecks: event.target.checked })}
          />
          {t("privacy.traffic.update-check.toggle")}
        </label>
      )}
    </li>
  );
}

function CrashReports({
  enabled,
  onChange,
}: {
  enabled: boolean;
  onChange(enabled: boolean): void;
}) {
  const t = useT();
  const id = useId();
  return (
    <section data-testid="crash-reports">
      <h3 className="mb-1 text-sm font-medium">{t("privacy.crashReports.title")}</h3>
      <label className="flex items-start gap-2 text-sm">
        <input
          type="checkbox"
          role="switch"
          data-testid="crash-reports-switch"
          className="mt-1"
          checked={enabled}
          aria-checked={enabled}
          aria-describedby={`${id}-description`}
          onChange={(event) => onChange(event.target.checked)}
        />
        <span>
          {t("privacy.crashReports.toggle")}
          <span id={`${id}-description`} className="block text-xs text-gray-500">
            {t("privacy.crashReports.body")}
          </span>
        </span>
      </label>
    </section>
  );
}
