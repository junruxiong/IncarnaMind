import { type FormEvent, useEffect, useId, useState } from "react";
import {
  type ChatProvider,
  type ChatProviderKind,
  type ConnectionTestResult,
  chatProviderKinds,
  type SecretStorageStatus,
  type TestChatConnectionInput,
} from "../../../../core/api";
import { catalogProviderOfKind, endpointOf } from "../../../../core/providers/catalog/providers";
import { features } from "../../../../shared/features";
import { core } from "../../core";
import { errorMessage } from "../../errors";
import { useLanguage, useT } from "../../i18n";
import {
  buttonClass,
  choiceListClass,
  compactChoiceRadioClass,
  compactChoiceRowClass,
  errorTextClass,
  fieldLabelClass,
  hintClass,
  inputClass,
  noticeClass,
  primaryButtonClass,
  successTextClass,
} from "../ui";
import { testErrorKey, useProviderKey } from "./shared";

/**
 * Providers set up with a key or a server URL: every hosted provider in the
 * catalog, then a server the User gives. Ollama has its own one-click card.
 */
const formKinds: readonly ChatProviderKind[] = [
  ...chatProviderKinds.filter((kind) => (catalogProviderOfKind(kind)?.endpoints.length ?? 0) > 0),
  "openai-compatible",
];

type FormKind = ChatProviderKind;

/**
 * Prefilled when a provider is picked: its Answers model in the catalog, none
 * for a server. The User can type any model their account has.
 */
const suggestedModel = (kind: FormKind): string => catalogProviderOfKind(kind)?.roles.answers ?? "";

const isSuggestion = (model: string) => formKinds.some((kind) => suggestedModel(kind) === model);

const sameServer = (provider: ChatProvider, kind: FormKind, endpoint: string, baseUrl: string) =>
  provider.kind === kind &&
  (kind === "openai-compatible"
    ? provider.baseUrl === baseUrl.trim().replace(/\/+$/, "")
    : provider.endpoint === endpoint);

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
  const language = useLanguage();
  const [kind, setKind] = useState<FormKind | null>(null);
  /** The region of a provider that has several; its first otherwise. */
  const [endpointId, setEndpointId] = useState("");
  const key = useProviderKey();
  const { apiKey } = key;
  const [baseUrl, setBaseUrl] = useState("");
  const [modelId, setModelId] = useState("");
  const [busy, setBusy] = useState<"testing" | "saving" | null>(null);
  const [test, setTest] = useState<ConnectionTestResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [secretStorage, setSecretStorage] = useState<SecretStorageStatus | null>(null);

  useEffect(() => {
    core.getSecretStorage().then(setSecretStorage, () => undefined);
  }, []);

  const catalog = kind ? catalogProviderOfKind(kind) : undefined;
  const endpoint = catalog ? endpointOf(catalog, endpointId) : undefined;
  const saved = kind
    ? providers.find((each) => sameServer(each, kind, endpoint?.id ?? "", baseUrl))
    : undefined;
  const keyRequired = kind !== null && kind !== "openai-compatible";
  const hasKey = apiKey.trim() !== "" || Boolean(saved?.hasApiKey);
  const keyBlocked = apiKey.trim() !== "" && secretStorage !== null && !secretStorage.canSave;
  const complete =
    kind !== null &&
    modelId.trim() !== "" &&
    (kind !== "openai-compatible" || baseUrl.trim() !== "") &&
    (!keyRequired || hasKey);

  const choose = (next: FormKind) => {
    // A key typed for one provider is never sent to another.
    if (next !== kind) key.clear();
    setKind(next);
    setEndpointId("");
    setModelId((current) =>
      current === "" || isSuggestion(current) ? suggestedModel(next) : current,
    );
    setTest(null);
    setError(null);
  };

  const input = (): TestChatConnectionInput | null => {
    if (!kind) return null;
    return {
      kind,
      modelId: modelId.trim(),
      ...(endpoint ? { endpoint: endpoint.id } : {}),
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
        key.clear();
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
        <legend className={`mb-1.5 ${fieldLabelClass}`}>{t("providers.form.label")}</legend>
        <div className={`${choiceListClass} bg-sheet`}>
          {formKinds.map((option) => (
            <label key={option} className={compactChoiceRowClass}>
              <input
                type="radio"
                name={`${id}-kind`}
                value={option}
                checked={kind === option}
                onChange={() => choose(option)}
                className={compactChoiceRadioClass}
              />
              {t(`providers.kind.${option}`)}
            </label>
          ))}
          {features.signInWithChatGpt && (
            <label className={compactChoiceRowClass}>
              <input
                type="radio"
                name={`${id}-kind`}
                disabled
                className={compactChoiceRadioClass}
              />
              <span className="flex min-w-0 flex-1 items-baseline justify-between gap-3">
                {t("providers.kind.chatgpt")}
                <span className="text-[12px]">{t("providers.chatgpt.unavailable")}</span>
              </span>
            </label>
          )}
        </div>
      </fieldset>

      {kind && (
        <>
          {catalog && (
            <div className="flex flex-col gap-1" data-testid="provider-facts">
              {/* Where the data goes and what the provider does with it, before a key is pasted. */}
              <p className={hintClass}>
                <span className="font-semibold text-ink">{t("providers.form.dataUse")}.</span>{" "}
                {catalog.dataUse.summary[language === "zh-CN" ? "zh" : "en"]}
              </p>
            </div>
          )}

          {catalog && catalog.endpoints.length > 1 && (
            <fieldset>
              <legend className={`mb-1.5 ${fieldLabelClass}`}>{t("providers.form.region")}</legend>
              <div className={`${choiceListClass} bg-sheet`}>
                {catalog.endpoints.map((option) => (
                  <label key={option.id} className={compactChoiceRowClass}>
                    <input
                      type="radio"
                      name={`${id}-region`}
                      value={option.id}
                      checked={endpoint?.id === option.id}
                      onChange={() => {
                        // A key belongs to its region: it is never sent to another.
                        if (endpoint?.id !== option.id) key.clear();
                        setEndpointId(option.id);
                        setTest(null);
                        setError(null);
                      }}
                      className={compactChoiceRadioClass}
                    />
                    {option.label[language === "zh-CN" ? "zh" : "en"]}
                  </label>
                ))}
              </div>
            </fieldset>
          )}

          {kind === "openai-compatible" && (
            <label className={fieldLabelClass}>
              {t("providers.form.baseUrl")}
              <input
                type="url"
                required
                value={baseUrl}
                onChange={(event) => setBaseUrl(event.target.value)}
                onBlur={() => key.leftAddress(baseUrl)}
                placeholder="https://"
                spellCheck={false}
                className={inputClass}
              />
              <span className={hintClass}>{t("providers.form.baseUrlHint")}</span>
            </label>
          )}

          <label className={fieldLabelClass}>
            {keyRequired ? t("providers.form.apiKey") : t("providers.form.apiKeyOptional")}
            <input
              type="password"
              value={apiKey}
              onChange={(event) => key.type(event.target.value, baseUrl)}
              autoComplete="off"
              spellCheck={false}
              className={inputClass}
            />
            {saved?.hasApiKey && (
              <span className={hintClass}>{t("providers.form.apiKeySaved")}</span>
            )}
            {(endpoint?.keyUrl ?? catalog?.keyUrl) && (
              <a
                href={endpoint?.keyUrl ?? catalog?.keyUrl}
                target="_blank"
                rel="noreferrer"
                className={`${hintClass} underline`}
              >
                {t("providers.form.getKey")} ↗
              </a>
            )}
          </label>

          {secretStorage && !secretStorage.canSave && (
            <SecretStorageNotice status={secretStorage} onAccept={() => void acceptPlainText()} />
          )}

          <label className={fieldLabelClass}>
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
              type="submit"
              disabled={!complete || keyBlocked || busy !== null}
              className={primaryButtonClass}
            >
              {busy === "saving" ? t("providers.form.saving") : t("providers.form.save")}
            </button>
            <button
              type="button"
              disabled={!complete || busy !== null}
              onClick={() => void run("testing")}
              className={buttonClass}
            >
              {busy === "testing" ? t("providers.form.testing") : t("providers.form.test")}
            </button>
          </div>

          {test && <TestResult result={test} onRetry={() => void run("testing")} />}
          {error && (
            <p role="alert" className={errorTextClass}>
              {error}
            </p>
          )}
        </>
      )}
    </form>
  );
}

/**
 * A provider's own error message, readable: some send their JSON error body
 * (e.g. `{"error_type":"authentication_error","message":"…"}`), whose
 * message, or else its type, says it in words.
 */
function readableMessage(message: string): string {
  const text = message.trim();
  if (!text.startsWith("{")) return message;
  try {
    const body = JSON.parse(text) as Record<string, unknown>;
    const inner = body.error;
    const nested =
      inner && typeof inner === "object" ? (inner as Record<string, unknown>).message : inner;
    const said = [body.message, nested, body.detail].find(
      (each): each is string => typeof each === "string" && each.trim() !== "",
    );
    if (said) return said;
    // Only a type, e.g. "authentication_error": as words.
    return typeof body.error_type === "string" ? body.error_type.replaceAll("_", " ") : message;
  } catch {
    return message;
  }
}

/** Forgets the User's latest "Don't allow", so the next request asks again. */
async function forgetLatestDecline(): Promise<void> {
  const flows = await core.listDataFlows();
  const latest = flows
    .filter((each) => each.consent === "declined" && each.decidedAt !== null)
    .sort((a, b) => (b.decidedAt ?? "").localeCompare(a.decidedAt ?? ""))[0];
  if (latest) await core.revokeConsent(latest.flow.id, latest.flow.service.id);
}

/**
 * A connection test's outcome. After a "Don't allow", `onRetry` offers to ask
 * again: the decision is forgotten and the test runs again, asking first.
 */
export function TestResult({
  result,
  onRetry,
}: {
  result: ConnectionTestResult;
  onRetry?: () => void;
}) {
  const t = useT();
  if (result.ok) {
    return (
      <p data-testid="connection-test" className={successTextClass}>
        {t("providers.test.ok")}
      </p>
    );
  }
  return (
    <div data-testid="connection-test" role="alert" className={errorTextClass}>
      <p>{t(testErrorKey(result.error.kind))}</p>
      {/* The provider's own words help with its errors; a declined consent needs none. */}
      {result.error.message && result.error.kind !== "consent-declined" && (
        <p className="mt-1 text-[12px] leading-[18px] break-words text-ink-meta">
          {readableMessage(result.error.message)}
        </p>
      )}
      {result.error.kind === "consent-declined" && onRetry && (
        <button
          type="button"
          data-testid="connection-test-ask-again"
          onClick={() => {
            void forgetLatestDecline().then(onRetry, onRetry);
          }}
          className={`mt-2 ${buttonClass}`}
        >
          {t("consent.settings.askAgain")}
        </button>
      )}
    </div>
  );
}

/** Linux without a keyring (safeStorage "basic_text"), or no encryption at all. */
export function SecretStorageNotice({
  status,
  onAccept,
}: {
  status: SecretStorageStatus;
  onAccept(): void;
}) {
  const t = useT();
  if (status.protection === "unavailable") {
    return (
      <p role="alert" className={noticeClass}>
        {t("providers.secrets.unavailable")}
      </p>
    );
  }
  return (
    <div role="alert" data-testid="plain-text-secrets" className={noticeClass}>
      <p className="font-semibold text-ink">{t("providers.secrets.plainText.title")}</p>
      <p className="mt-1">{t("providers.secrets.plainText.body")}</p>
      <label className="mt-2 flex items-start gap-2">
        <input
          type="checkbox"
          checked={status.plainTextAccepted}
          onChange={onAccept}
          className="mt-[3px] size-4 shrink-0 accent-ink"
        />
        <span>{t("providers.secrets.plainText.accept")}</span>
      </label>
    </div>
  );
}
