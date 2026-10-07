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
import {
  buttonStyle,
  pageIntroClass,
  rowStatusClass,
  rowTextClass,
  rowTitleClass,
  ruledListClass,
  ruledRow,
  ruledRowClass,
  sectionNoteClass,
  sectionTitleClass,
} from "./ui";

const reportError = (failure: unknown) =>
  useAppStore.setState({ actionError: errorMessage(failure) });

/**
 * Settings → Privacy: everything IncarnaMind sends from this computer, in one
 * place, as rows split by rules with each status on the right. Every
 * registered data flow with its consent (revoke, allow), network traffic that
 * carries nothing of the User's (with the update-check switch), what Skill
 * scripts can do, and crash reports when this copy can send them.
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
    <div data-testid="privacy-settings" className="flex flex-col gap-7">
      <p className={pageIntroClass}>{t("privacy.intro")}</p>

      <section data-testid="consent-settings" className="flex flex-col">
        <h4 className={sectionTitleClass}>{t("consent.settings.title")}</h4>
        <p className={sectionNoteClass}>{t("privacy.flows.intro")}</p>
        <ul className={ruledListClass}>
          {flows?.map((flow) => (
            <DataFlowItem key={flow.id} flow={flow} />
          ))}
        </ul>
      </section>

      <section data-testid="network-traffic" className="flex flex-col">
        <h4 className={sectionTitleClass}>{t("privacy.traffic.title")}</h4>
        <p className={sectionNoteClass}>{t("privacy.traffic.intro")}</p>
        <ul className={ruledListClass}>
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

      <section data-testid="skill-scripts-note" role="note" className="flex flex-col">
        <h4 className={sectionTitleClass}>{t("privacy.skills.title")}</h4>
        <p className={`mt-0.5 ${rowTextClass}`}>{t("privacy.skills.body")}</p>
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

/**
 * A data flow: a row for each service it goes to (what it sends, and the
 * User's decision on the right), or one row saying it stays on this computer.
 */
function DataFlowItem({ flow }: { flow: RegisteredDataFlow }) {
  const t = useT();
  const name = t(`consent.flow.${flow.id}`);
  return (
    <li
      data-testid="data-flow"
      data-flow-id={flow.id}
      className="flex flex-col [&>*+*]:border-t [&>*+*]:border-rule"
    >
      {flow.services.length === 0 ? (
        <div className={ruledRow("center", true)}>
          <span className="text-ui text-ink">{name}</span>
          <span
            data-testid="data-flow-not-in-use"
            title={t("privacy.flows.notInUse")}
            className={rowStatusClass}
          >
            {t("privacy.flows.local")}
          </span>
        </div>
      ) : (
        flow.services.map((status) => (
          <ServiceDecision
            key={status.flow.service.id}
            name={name}
            sends={flow.sends}
            status={status}
          />
        ))
      )}
    </li>
  );
}

/** One service of a flow: what it sends, the decision and when it was made, with revoke and allow. */
function ServiceDecision({
  name,
  sends,
  status,
}: {
  name: string;
  sends: RegisteredDataFlow["sends"];
  status: DataFlowStatus;
}) {
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
    <div
      data-testid="data-flow-service"
      data-service-id={flow.service.id}
      data-consent={consent}
      className={ruledRowClass}
    >
      <div className="flex min-w-0 flex-col gap-1">
        <p className="text-ui break-words text-ink">
          <span className="font-semibold">{name}</span>
          <span className="text-ink-meta">
            {" · "}
            {t("privacy.flows.service", { service: flow.service.name })}
          </span>
        </p>
        <ul
          aria-label={t("privacy.flows.sends")}
          className="list-disc pl-4 text-[13px] leading-5 text-ink-secondary"
        >
          {sends.map((kind) => (
            <li key={kind}>{t(`consent.data.${kind}`)}</li>
          ))}
        </ul>
      </div>
      <div className="flex flex-col items-end gap-2">
        <span
          data-testid="data-flow-decision"
          className={`text-[13px] leading-5 ${
            consent === "accepted" ? "font-semibold text-success" : "text-ink-meta"
          }`}
        >
          {decision}
        </span>
        <div className="flex flex-wrap justify-end gap-2">
          {consent === "accepted" ? (
            <button
              type="button"
              data-testid="data-flow-revoke"
              disabled={busy}
              onClick={() => void revoke()}
              className={buttonStyle("secondary", "sm")}
            >
              {t("consent.settings.revoke")}
            </button>
          ) : (
            <button
              type="button"
              data-testid="data-flow-allow"
              disabled={busy}
              onClick={() => void allow()}
              className={buttonStyle("primary", "sm")}
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
              className={buttonStyle("secondary", "sm")}
            >
              {t("consent.settings.askAgain")}
            </button>
          )}
        </div>
      </div>
    </div>
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
  const switchable = traffic.id === "update-check" && choices !== null;
  return (
    <li
      data-testid="network-traffic-item"
      data-traffic-id={traffic.id}
      data-enabled={traffic.enabled}
      className={ruledRowClass}
    >
      <div className="flex min-w-0 flex-col gap-0.5">
        <p className="text-ui break-words text-ink">
          <span id={`${id}-title`} className="font-semibold">
            {t(`privacy.traffic.${traffic.id}`)}
          </span>
          <span className="text-ink-meta"> · {traffic.service.name}</span>
        </p>
        <p id={`${id}-description`} className={rowTextClass}>
          {t(`privacy.traffic.${traffic.id}.description`)}
        </p>
      </div>
      {switchable ? (
        <input
          type="checkbox"
          role="switch"
          data-testid="automatic-update-checks"
          aria-label={t("privacy.traffic.update-check.toggle")}
          aria-describedby={`${id}-description`}
          checked={choices.automaticUpdateChecks}
          aria-checked={choices.automaticUpdateChecks}
          onChange={(event) => onChange({ automaticUpdateChecks: event.target.checked })}
          className="switch"
        />
      ) : (
        <span className={rowStatusClass}>
          {traffic.enabled ? t("privacy.traffic.on") : t("privacy.traffic.off")}
        </span>
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
    <section data-testid="crash-reports" className="flex flex-col">
      <h4 className={`mb-2 ${sectionTitleClass}`}>{t("privacy.crashReports.title")}</h4>
      <div className={ruledListClass}>
        <div className={ruledRowClass}>
          <div className="flex min-w-0 flex-col gap-0.5">
            <label htmlFor={id} className={rowTitleClass}>
              {t("privacy.crashReports.toggle")}
            </label>
            <p id={`${id}-description`} className={rowTextClass}>
              {t("privacy.crashReports.body")}
            </p>
          </div>
          <input
            id={id}
            type="checkbox"
            role="switch"
            data-testid="crash-reports-switch"
            className="switch"
            checked={enabled}
            aria-checked={enabled}
            aria-describedby={`${id}-description`}
            onChange={(event) => onChange(event.target.checked)}
          />
        </div>
      </div>
    </section>
  );
}
