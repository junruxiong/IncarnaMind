import { type FormEvent, useEffect, useId, useState } from "react";
import type {
  ChatProvider,
  ChatProviderKind,
  ConnectionTestResult,
  SecretStorageStatus,
  TestChatConnectionInput,
} from "../../../../core/api";
import { features } from "../../../../shared/features";
import { core } from "../../core";
import { errorMessage } from "../../errors";
import { useT } from "../../i18n";
import { buttonClass, inputClass, primaryButtonClass } from "./shared";

/** Providers set up with a key or a server URL. Ollama has its own one-click card. */
const formKinds = [
  "openai",
  "anthropic",
  "google",
  "openai-compatible",
] as const satisfies readonly ChatProviderKind[];

type FormKind = (typeof formKinds)[number];

/** Prefilled when a provider is picked; the User can type any model their account has. */
const suggestedModels: Record<FormKind, string> = {
  openai: "gpt-5.5",
  anthropic: "claude-opus-5-5",
  google: "gemini-pro-latest",
  "openai-compatible": "",
};

const isSuggestion = (model: string) => Object.values(suggestedModels).includes(model);

const sameServer = (provider: ChatProvider, kind: FormKind, baseUrl: string) =>
  provider.kind === kind &&
  (kind !== "openai-compatible" || provider.baseUrl === baseUrl.trim().replace(/\/+$/, ""));

/**
 * Pick a provider, paste a key (and a URL for an OpenAI-compatible server),
 * choose a model, test the connection, and use it. No provider is preselected.
 */
export function ProviderForm({
  providers,
  onSaved,
}: {
  /** Saved providers, to tell the User when a key is already stored. */
  providers: readonly ChatProvider[];
  onSaved(provider: ChatProvider): void;
}) {
  const t = useT();
  const id = useId();
  const [kind, setKind] = useState<FormKind | null>(null);
  const [apiKey, setApiKey] = useState("");
  const [baseUrl, setBaseUrl] = useState("");
  const [modelId, setModelId] = useState("");
  const [busy, setBusy] = useState<"testing" | "saving" | null>(null);
  const [test, setTest] = useState<ConnectionTestResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [secretStorage, setSecretStorage] = useState<SecretStorageStatus | null>(null);

  useEffect(() => {
    core.getSecretStorage().then(setSecretStorage, () => undefined);
  }, []);

  const saved = kind ? providers.find((each) => sameServer(each, kind, baseUrl)) : undefined;
  const keyRequired = kind !== null && kind !== "openai-compatible";
  const hasKey = apiKey.trim() !== "" || Boolean(saved?.hasApiKey);
  const keyBlocked = apiKey.trim() !== "" && secretStorage !== null && !secretStorage.canSave;
  const complete =
    kind !== null &&
    modelId.trim() !== "" &&
    (kind !== "openai-compatible" || baseUrl.trim() !== "") &&
    (!keyRequired || hasKey);

  const choose = (next: FormKind) => {
    setKind(next);
    setModelId((current) =>
      current === "" || isSuggestion(current) ? suggestedModels[next] : current,
    );
    setTest(null);
    setError(null);
  };

  const input = (): TestChatConnectionInput | null => {
    if (!kind) return null;
    return {
      kind,
      modelId: modelId.trim(),
      ...(kind === "openai-compatible" ? { baseUrl: baseUrl.trim() } : {}),
      ...(apiKey.trim() ? { apiKey: apiKey.trim() } : {}),
    };
  };

  const run = async (action: "testing" | "saving") => {
    const request = input();
    if (!request) return;
    setBusy(action);
    setError(null);
    if (action === "testing") setTest(null);
    try {
      if (action === "testing") {
        setTest(await core.testChatConnection(request));
      } else {
        const provider = await core.saveChatProvider(request);
        setApiKey("");
        onSaved(provider);
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
    <form data-testid="provider-form" onSubmit={submit} className="flex flex-col gap-3">
      <fieldset>
        <legend className="mb-1 text-sm text-gray-600">{t("providers.form.label")}</legend>
        <div className="grid grid-cols-2 gap-2">
          {formKinds.map((option) => (
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
                onChange={() => choose(option)}
              />
              {t(`providers.kind.${option}`)}
            </label>
          ))}
          {features.signInWithChatGpt && (
            <label className="flex items-center gap-2 rounded-[9px] border border-gray-200 px-3 py-2 text-sm text-gray-400">
              <input type="radio" name={`${id}-kind`} disabled />
              <span>
                {t("providers.kind.chatgpt")}
                <span className="block text-xs">{t("providers.chatgpt.unavailable")}</span>
              </span>
            </label>
          )}
        </div>
      </fieldset>

      {kind && (
        <>
          {kind === "openai-compatible" && (
            <label className="text-sm text-gray-600">
              {t("providers.form.baseUrl")}
              <input
                type="url"
                required
                value={baseUrl}
                onChange={(event) => setBaseUrl(event.target.value)}
                placeholder="https://"
                spellCheck={false}
                className={inputClass}
              />
              <span className="mt-1 block text-xs text-gray-500">
                {t("providers.form.baseUrlHint")}
              </span>
            </label>
          )}

          <label className="text-sm text-gray-600">
            {keyRequired ? t("providers.form.apiKey") : t("providers.form.apiKeyOptional")}
            <input
              type="password"
              value={apiKey}
              onChange={(event) => setApiKey(event.target.value)}
              autoComplete="off"
              spellCheck={false}
              className={inputClass}
            />
            {saved?.hasApiKey && (
              <span className="mt-1 block text-xs text-gray-500">
                {t("providers.form.apiKeySaved")}
              </span>
            )}
          </label>

          {secretStorage && !secretStorage.canSave && (
            <SecretStorageNotice status={secretStorage} onAccept={() => void acceptPlainText()} />
          )}

          <label className="text-sm text-gray-600">
            {t("providers.form.model")}
            <input
              required
              value={modelId}
              onChange={(event) => setModelId(event.target.value)}
              spellCheck={false}
              className={inputClass}
            />
          </label>

          <div className="flex flex-wrap items-center gap-2">
            <button
              type="button"
              disabled={!complete || busy !== null}
              onClick={() => void run("testing")}
              className={buttonClass}
            >
              {busy === "testing" ? t("providers.form.testing") : t("providers.form.test")}
            </button>
            <button
              type="submit"
              disabled={!complete || keyBlocked || busy !== null}
              className={primaryButtonClass}
            >
              {busy === "saving" ? t("providers.form.saving") : t("providers.form.save")}
            </button>
          </div>

          {test && <TestResult result={test} />}
          {error && (
            <p role="alert" className="text-sm text-red-700">
              {error}
            </p>
          )}
        </>
      )}
    </form>
  );
}

function TestResult({ result }: { result: ConnectionTestResult }) {
  const t = useT();
  if (result.ok) {
    return (
      <p data-testid="connection-test" className="text-sm text-emerald-700">
        {t("providers.test.ok")}
      </p>
    );
  }
  return (
    <div data-testid="connection-test" role="alert" className="text-sm text-red-700">
      <p>{t(`providers.test.${result.error.kind}`)}</p>
      {/* The provider's own words help with its errors; a declined consent needs none. */}
      {result.error.message && result.error.kind !== "consent-declined" && (
        <p className="mt-1 text-xs break-words text-gray-500">{result.error.message}</p>
      )}
    </div>
  );
}

/** Linux without a keyring (safeStorage "basic_text"), or no encryption at all. */
function SecretStorageNotice({
  status,
  onAccept,
}: {
  status: SecretStorageStatus;
  onAccept(): void;
}) {
  const t = useT();
  if (status.protection === "unavailable") {
    return (
      <p role="alert" className="rounded-[9px] bg-amber-50 p-3 text-sm text-amber-900">
        {t("providers.secrets.unavailable")}
      </p>
    );
  }
  return (
    <div
      role="alert"
      data-testid="plain-text-secrets"
      className="rounded-[9px] bg-amber-50 p-3 text-sm text-amber-900"
    >
      <p className="font-medium">{t("providers.secrets.plainText.title")}</p>
      <p className="mt-1">{t("providers.secrets.plainText.body")}</p>
      <label className="mt-2 flex items-start gap-2">
        <input type="checkbox" checked={status.plainTextAccepted} onChange={onAccept} />
        <span>{t("providers.secrets.plainText.accept")}</span>
      </label>
    </div>
  );
}
