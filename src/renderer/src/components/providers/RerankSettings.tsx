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
import { SecretStorageNotice, TestResult } from "./ProviderForm";
import { buttonClass, inputClass, primaryButtonClass } from "./shared";

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
    <section data-testid="rerank-settings">
      <h3 className="mb-1 text-sm font-medium">{t("rerank.settings.title")}</h3>
      <p className="text-sm text-gray-600">{t("rerank.settings.body")}</p>

      {rerank.enabled && !editing && (
        <div className="mt-2 flex flex-col gap-2">
          {rerank.paused && (
            <p className="rounded-[9px] bg-gray-100 p-2 text-sm text-gray-700">
              {t("rerank.settings.paused")}
            </p>
          )}
          {!rerank.hasApiKey && (
            <p className="rounded-[9px] bg-amber-50 p-2 text-sm text-amber-900">
              {t("rerank.settings.keyMissing", { service })}
            </p>
          )}
          <p className="text-sm">
            {t("rerank.settings.inUse", { service, model: rerank.modelId ?? "" })}{" "}
            <span className="text-gray-600">{t("rerank.settings.sends", { service })}</span>
          </p>
          <div className="flex gap-2">
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
          </div>
        </div>
      )}

      {!rerank.enabled && !editing && (
        <button
          type="button"
          data-testid="rerank-set-up"
          onClick={() => setEditing(true)}
          className={`${buttonClass} mt-2`}
        >
          {t("rerank.settings.setUp")}
        </button>
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
  const [apiKey, setApiKey] = useState("");
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

  return (
    <form data-testid="rerank-form" onSubmit={submit} className="mt-3 flex flex-col gap-3">
      <fieldset>
        <legend className="mb-1 text-sm text-gray-600">{t("rerank.form.label")}</legend>
        <div className="grid grid-cols-2 gap-2">
          {rerankProviderKinds.map((option) => (
            <label
              key={option}
              className={`flex cursor-pointer items-center gap-2 rounded-[9px] border px-3 py-2 text-sm ${
                kind === option ? "border-gray-800" : "border-gray-300 hover:bg-gray-50"
              }`}
            >
              <input
                type="radio"
                name={`${id}-kind`}
                value={option}
                checked={kind === option}
                onChange={() => {
                  setKind(option);
                  setTest(null);
                }}
              />
              {t(`rerank.kind.${option}`)}
            </label>
          ))}
        </div>
      </fieldset>

      <label className="text-sm text-gray-600">
        {t("providers.form.apiKey")}
        <input
          type="password"
          value={apiKey}
          onChange={(event) => setApiKey(event.target.value)}
          autoComplete="off"
          spellCheck={false}
          className={inputClass}
        />
        {keySaved && (
          <span className="mt-1 block text-xs text-gray-500">
            {t("providers.form.apiKeySaved")}
          </span>
        )}
      </label>

      {secretStorage && !secretStorage.canSave && (
        <SecretStorageNotice status={secretStorage} onAccept={() => void acceptPlainText()} />
      )}

      <label className="text-sm text-gray-600">
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
          type="button"
          disabled={!hasKey || busy !== null}
          onClick={() => void run("testing")}
          className={buttonClass}
        >
          {busy === "testing" ? t("providers.form.testing") : t("providers.form.test")}
        </button>
        <button
          type="submit"
          disabled={!hasKey || keyBlocked || busy !== null}
          className={primaryButtonClass}
        >
          {busy === "saving" ? t("providers.form.saving") : t("rerank.form.save")}
        </button>
        <button type="button" onClick={onCancel} className={buttonClass}>
          {t("providers.settings.cancel")}
        </button>
      </div>

      {test && <TestResult result={test} />}
      {error && (
        <p role="alert" className="text-sm break-words text-red-700">
          {error}
        </p>
      )}
    </form>
  );
}
