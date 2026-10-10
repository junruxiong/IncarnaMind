import { type FormEvent, useEffect, useState } from "react";
import {
  type ConnectionTestResult,
  JEV_DEFAULT_MODEL,
  type JevSettings,
  type SaveJevSettingsInput,
  type SecretStorageStatus,
} from "../../../../core/api";
import { core } from "../../core";
import { errorMessage } from "../../errors";
import { useT } from "../../i18n";
import { useAppStore } from "../../store";
import {
  buttonClass,
  errorTextClass,
  fieldLabelClass,
  ghostButtonClass,
  hintClass,
  inputClass,
  noticeClass,
  primaryButtonClass,
  rowButtonsClass,
  rowTextClass,
  ruledListClass,
  ruledRowClass,
  sectionNoteClass,
  sectionTitleClass,
} from "../ui";
import { SecretStorageNotice, TestResult } from "./ProviderForm";

/** TypeSafe's hosted API, shown as the server's placeholder. */
const HOSTED_URL = "https://api.typesafe.ai";

const toPercent = (probability: number) => Math.round(probability * 100);

/**
 * Settings → Models → Automatic tagging: the chat model by default, or TypeSafe Jev
 * with a key (and, for Jev-compatible models, another server), plus the band
 * of probabilities whose Tags are marked for review.
 */
export function JevSettingsSection() {
  const t = useT();
  const [jev, setJev] = useState<JevSettings | null>(null);
  const [editing, setEditing] = useState(false);

  useEffect(() => {
    core.getJevSettings().then(setJev, () => undefined);
    return core.on("jev.changed", setJev);
  }, []);

  const remove = async () => {
    try {
      setJev(await core.removeJevSettings());
    } catch (failure) {
      useAppStore.setState({ actionError: errorMessage(failure) });
    }
  };

  if (!jev) return null;
  return (
    <section data-testid="jev-settings" className="flex flex-col">
      <h4 className={sectionTitleClass}>{t("jev.settings.title")}</h4>
      <p className={sectionNoteClass}>{t("jev.settings.body")}</p>

      {jev.enabled && !editing && (
        <div className={ruledListClass}>
          <div className={ruledRowClass}>
            <div className="flex min-w-0 flex-col gap-0.5">
              <p className="text-ui text-ink">{t("jev.settings.inUse")}</p>
              <p className={`${rowTextClass} break-words`}>
                {t("jev.settings.server", { server: jev.endpoint ?? t("jev.settings.hosted") })}
              </p>
              {!jev.hasApiKey && (
                <p className={`mt-1.5 ${noticeClass}`}>{t("jev.settings.keyMissing")}</p>
              )}
            </div>
            <div className={rowButtonsClass}>
              <button type="button" onClick={() => setEditing(true)} className={buttonClass}>
                {t("jev.settings.change")}
              </button>
              <button
                type="button"
                data-testid="jev-remove"
                onClick={() => void remove()}
                className={buttonClass}
              >
                {t("jev.settings.remove")}
              </button>
            </div>
          </div>
        </div>
      )}

      {!jev.enabled && !editing && (
        <div>
          <button
            type="button"
            data-testid="jev-set-up"
            onClick={() => setEditing(true)}
            className={buttonClass}
          >
            {t("jev.settings.setUp")}
          </button>
        </div>
      )}

      {editing && (
        <JevForm
          current={jev}
          onSaved={(saved) => {
            setJev(saved);
            setEditing(false);
          }}
          onCancel={() => setEditing(false)}
        />
      )}
    </section>
  );
}

/** The Jev key, an optional server and model, and the review band; test, then save. */
function JevForm({
  current,
  onSaved,
  onCancel,
}: {
  current: JevSettings;
  onSaved(saved: JevSettings): void;
  onCancel(): void;
}) {
  const t = useT();
  const [apiKey, setApiKey] = useState("");
  const [endpoint, setEndpoint] = useState(current.endpoint ?? "");
  const [model, setModel] = useState(current.model === JEV_DEFAULT_MODEL ? "" : current.model);
  const [low, setLow] = useState(toPercent(current.reviewBand.low));
  const [high, setHigh] = useState(toPercent(current.reviewBand.high));
  const [busy, setBusy] = useState<"testing" | "saving" | null>(null);
  const [test, setTest] = useState<ConnectionTestResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [secretStorage, setSecretStorage] = useState<SecretStorageStatus | null>(null);

  useEffect(() => {
    core.getSecretStorage().then(setSecretStorage, () => undefined);
  }, []);

  const hasKey = apiKey.trim() !== "" || current.hasApiKey;
  const keyBlocked = apiKey.trim() !== "" && secretStorage !== null && !secretStorage.canSave;
  const bandValid = low > 0 && low <= high && high <= 100;

  const input = (): SaveJevSettingsInput => ({
    ...(apiKey.trim() ? { apiKey: apiKey.trim() } : {}),
    endpoint: endpoint.trim() || null,
    model: model.trim() || null,
  });

  const run = async (action: "testing" | "saving") => {
    setBusy(action);
    setError(null);
    if (action === "testing") setTest(null);
    try {
      if (action === "testing") {
        setTest(await core.testJevConnection(input()));
      } else {
        const saved = await core.saveJevSettings({
          ...input(),
          reviewBand: { low: low / 100, high: high / 100 },
        });
        setApiKey("");
        onSaved(saved);
      }
    } catch (failure) {
      setError(errorMessage(failure));
    } finally {
      setBusy(null);
    }
  };

  const submit = (event: FormEvent) => {
    event.preventDefault();
    void run("saving");
  };

  const acceptPlainText = async () => {
    try {
      setSecretStorage(await core.acceptPlainTextSecretStorage());
    } catch (failure) {
      setError(errorMessage(failure));
    }
  };

  const percentInput = (value: number, onChange: (next: number) => void, label: string) => (
    <label className={`flex-1 ${fieldLabelClass}`}>
      {label}
      <input
        type="number"
        min={1}
        max={100}
        step={1}
        required
        value={Number.isNaN(value) ? "" : value}
        onChange={(event) => onChange(event.target.valueAsNumber)}
        className={inputClass}
      />
    </label>
  );

  return (
    <form data-testid="jev-form" onSubmit={submit} className="flex flex-col gap-3">
      <label className={fieldLabelClass}>
        {t("jev.form.apiKey")}
        <input
          type="password"
          value={apiKey}
          onChange={(event) => setApiKey(event.target.value)}
          autoComplete="off"
          spellCheck={false}
          className={inputClass}
        />
        {current.hasApiKey && <span className={hintClass}>{t("providers.form.apiKeySaved")}</span>}
      </label>

      {secretStorage && !secretStorage.canSave && (
        <SecretStorageNotice status={secretStorage} onAccept={() => void acceptPlainText()} />
      )}

      <details className="text-[13px] leading-5 text-ink-secondary">
        <summary className="cursor-pointer select-none hover:text-ink">
          {t("jev.form.advanced")}
        </summary>
        <div className="mt-2 flex flex-col gap-3">
          <label className={fieldLabelClass}>
            {t("jev.form.endpoint")}
            <input
              type="url"
              value={endpoint}
              onChange={(event) => setEndpoint(event.target.value)}
              placeholder={HOSTED_URL}
              spellCheck={false}
              className={inputClass}
            />
            <span className={hintClass}>{t("jev.form.endpointHint")}</span>
          </label>
          <label className={fieldLabelClass}>
            {t("jev.form.model")}
            <input
              value={model}
              onChange={(event) => setModel(event.target.value)}
              placeholder={JEV_DEFAULT_MODEL}
              spellCheck={false}
              className={inputClass}
            />
          </label>
          <div>
            <div className="flex gap-2">
              {percentInput(low, setLow, t("jev.form.reviewFrom"))}
              {percentInput(high, setHigh, t("jev.form.reviewTo"))}
            </div>
            <span className={hintClass}>{t("jev.form.reviewHint")}</span>
          </div>
        </div>
      </details>

      <div className="flex flex-wrap items-center gap-2">
        <button
          type="submit"
          disabled={!hasKey || !bandValid || keyBlocked || busy !== null}
          className={primaryButtonClass}
        >
          {busy === "saving" ? t("providers.form.saving") : t("jev.form.save")}
        </button>
        <button
          type="button"
          disabled={!hasKey || busy !== null}
          onClick={() => void run("testing")}
          className={buttonClass}
        >
          {busy === "testing" ? t("providers.form.testing") : t("providers.form.test")}
        </button>
        <button type="button" onClick={onCancel} className={ghostButtonClass}>
          {t("providers.settings.cancel")}
        </button>
      </div>

      {test && <TestResult result={test} onRetry={() => void run("testing")} />}
      {error && (
        <p role="alert" className={errorTextClass}>
          {error}
        </p>
      )}
    </form>
  );
}
