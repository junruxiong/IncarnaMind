import { type FormEvent, useEffect, useId, useState } from "react";
import {
  type ConnectionTestResult,
  DEFAULT_RERANK_MODELS,
  type RerankProviderKind,
  type RerankSettings,
  rerankProviderKinds,
  type SaveRerankSettingsInput,
  type SecretStorageStatus,
} from "../../../../core/api";
import { core } from "../../core";
import { errorMessage } from "../../errors";
import { useT } from "../../i18n";
import { useAppStore } from "../../store";
import {
  buttonClass,
  choiceListClass,
  compactChoiceRadioClass,
  compactChoiceRowClass,
  errorTextClass,
  fieldLabelClass,
  ghostButtonClass,
  hintClass,
  inputClass,
  noticeClass,
  primaryButtonClass,
  rowButtonsClass,
  rowStatusClass,
  rowTextClass,
  ruledListClass,
  ruledRow,
  ruledRowClass,
  sectionNoteClass,
  sectionTitleClass,
} from "../ui";
import { SecretStorageNotice, TestResult } from "./ProviderForm";
import { useProviderKey } from "./shared";

/**
 * Settings → Reranking: off without a key; with a Cohere or Voyage AI key,
 * document search reorders its best matches with the provider's model.
 */
export function RerankSettingsSection() {
  const t = useT();
  const [rerank, setRerank] = useState<RerankSettings | null>(null);
  const [editing, setEditing] = useState(false);

  useEffect(() => {
    core.getRerankSettings().then(setRerank, () => undefined);
    return core.on("rerank.changed", setRerank);
  }, []);

  const remove = async () => {
    try {
      setRerank(await core.removeRerankSettings());
    } catch (failure) {
      useAppStore.setState({ actionError: errorMessage(failure) });
    }
  };

  if (!rerank) return null;
  const service = rerank.service?.name ?? "";
  return (
    <section data-testid="rerank-settings" className="flex flex-col">
      <h4 className={sectionTitleClass}>{t("rerank.settings.title")}</h4>
      <p className={sectionNoteClass}>{t("rerank.settings.body")}</p>

      {!editing && (
        <div className={ruledListClass}>
          <div className={rerank.enabled ? ruledRowClass : ruledRow("center", true)}>
            <div className="flex min-w-0 flex-col gap-0.5">
              {rerank.enabled ? (
                <>
                  <p className="text-ui text-ink">
                    {t("rerank.settings.inUse", { service, model: rerank.modelId ?? "" })}
                  </p>
                  <p className={rowTextClass}>{t("rerank.settings.sends", { service })}</p>
                  {rerank.paused && (
                    <p className={`mt-1.5 ${noticeClass}`}>{t("rerank.settings.paused")}</p>
                  )}
                  {!rerank.hasApiKey && (
                    <p className={`mt-1.5 ${noticeClass}`}>
                      {t("rerank.settings.keyMissing", { service })}
                    </p>
                  )}
                </>
              ) : (
                <p className={rowStatusClass}>{t("privacy.traffic.off")}</p>
              )}
            </div>
            <div className={rowButtonsClass}>
              {rerank.enabled ? (
                <>
                  <button type="button" onClick={() => setEditing(true)} className={buttonClass}>
                    {t("rerank.settings.change")}
                  </button>
                  <button
                    type="button"
                    data-testid="rerank-remove"
                    onClick={() => void remove()}
                    className={buttonClass}
                  >
                    {t("rerank.settings.remove")}
                  </button>
                </>
              ) : (
                <button
                  type="button"
                  data-testid="rerank-set-up"
                  onClick={() => setEditing(true)}
                  className={buttonClass}
                >
                  {t("rerank.settings.setUp")}
                </button>
              )}
            </div>
          </div>
        </div>
      )}

      {editing && (
        <RerankForm
          current={rerank}
          onSaved={(saved) => {
            setRerank(saved);
            setEditing(false);
          }}
          onCancel={() => setEditing(false)}
        />
      )}
    </section>
  );
}

/** The provider, its key and an optional model; test, then save. */
function RerankForm({
  current,
  onSaved,
  onCancel,
}: {
  current: RerankSettings;
  onSaved(saved: RerankSettings): void;
  onCancel(): void;
}) {
  const t = useT();
  const id = useId();
  const [kind, setKind] = useState<RerankProviderKind>(current.kind ?? "cohere");
  const key = useProviderKey();
  const { apiKey } = key;
  const [model, setModel] = useState(
    current.kind && current.modelId !== DEFAULT_RERANK_MODELS[current.kind]
      ? (current.modelId ?? "")
      : "",
  );
  const [busy, setBusy] = useState<"testing" | "saving" | null>(null);
  const [test, setTest] = useState<ConnectionTestResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [secretStorage, setSecretStorage] = useState<SecretStorageStatus | null>(null);

  useEffect(() => {
    core.getSecretStorage().then(setSecretStorage, () => undefined);
  }, []);

  const keySaved = current.kind === kind && current.hasApiKey;
  const hasKey = apiKey.trim() !== "" || keySaved;
  const keyBlocked = apiKey.trim() !== "" && secretStorage !== null && !secretStorage.canSave;

  const input = (): SaveRerankSettingsInput => ({
    kind,
    ...(apiKey.trim() ? { apiKey: apiKey.trim() } : {}),
    modelId: model.trim() || null,
  });

  const run = async (action: "testing" | "saving") => {
    setBusy(action);
    setError(null);
    if (action === "testing") setTest(null);
    try {
      if (action === "testing") {
        setTest(await core.testRerankConnection(input()));
      } else {
        const saved = await core.saveRerankSettings(input());
        key.clear();
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

  return (
    <form data-testid="rerank-form" onSubmit={submit} className="flex flex-col gap-3">
      <fieldset>
        <legend className={`mb-1.5 ${fieldLabelClass}`}>{t("rerank.form.label")}</legend>
        <div className={choiceListClass}>
          {rerankProviderKinds.map((option) => (
            <label key={option} className={compactChoiceRowClass}>
              <input
                type="radio"
                name={`${id}-kind`}
                value={option}
                checked={kind === option}
                onChange={() => {
                  // A key typed for one provider is never sent to another.
                  if (option !== kind) key.clear();
                  setKind(option);
                  setTest(null);
                }}
                className={compactChoiceRadioClass}
              />
              {t(`rerank.kind.${option}`)}
            </label>
          ))}
        </div>
      </fieldset>

      <label className={fieldLabelClass}>
        {t("providers.form.apiKey")}
        <input
          type="password"
          value={apiKey}
          onChange={(event) => key.type(event.target.value)}
          autoComplete="off"
          spellCheck={false}
          className={inputClass}
        />
        {keySaved && <span className={hintClass}>{t("providers.form.apiKeySaved")}</span>}
      </label>

      {secretStorage && !secretStorage.canSave && (
        <SecretStorageNotice status={secretStorage} onAccept={() => void acceptPlainText()} />
      )}

      <label className={fieldLabelClass}>
        {t("rerank.form.model")}
        <input
          value={model}
          onChange={(event) => setModel(event.target.value)}
          placeholder={DEFAULT_RERANK_MODELS[kind]}
          spellCheck={false}
          className={inputClass}
        />
      </label>

      <div className="flex flex-wrap items-center gap-2">
        <button
          type="submit"
          disabled={!hasKey || keyBlocked || busy !== null}
          className={primaryButtonClass}
        >
          {busy === "saving" ? t("providers.form.saving") : t("rerank.form.save")}
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

      {test && <TestResult result={test} />}
      {error && (
        <p role="alert" className={errorTextClass}>
          {error}
        </p>
      )}
    </form>
  );
}
