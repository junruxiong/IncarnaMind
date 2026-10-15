import { type FormEvent, useEffect, useId, useRef, useState } from "react";
import type {
  ChatProvider,
  ChatProviderKind,
  ConnectionTestResult,
  SecretStorageStatus,
  TestChatConnectionInput,
} from "../../../../core/api";
import {
  catalogProviderOfKind,
  endpointOf,
  KEY_PROVIDER_KINDS,
  keyProviderGroups,
} from "../../../../core/providers/catalog/providers";
import type { MessageKey } from "../../../../shared/i18n";
import { modelName } from "../../../../shared/modelName";
import { core } from "../../core";
import { dayOf } from "../../dates";
import { errorMessage } from "../../errors";
import { useLanguage, useT } from "../../i18n";
import type { KeyRefusal } from "../../keyRefusals";
import { ProblemMark } from "../ProblemLine";
import {
  buttonClass,
  errorTextClass,
  fieldLabelClass,
  hintClass,
  inputClass,
  noticeClass,
  primaryButtonClass,
  successTextClass,
} from "../ui";
import { testErrorKey, useProviderKey } from "./shared";

type FormKind = ChatProviderKind;

/**
 * Prefilled when a provider is picked: its Answers model in the catalog, none
 * for a server. The User can type any model their account has.
 */
const suggestedModel = (kind: FormKind): string => catalogProviderOfKind(kind)?.roles.answers ?? "";

const isSuggestion = (model: string) =>
  KEY_PROVIDER_KINDS.some((kind) => suggestedModel(kind) === model);

/** The providers shown as one row of choices; the rest are under "More…". */
const ROW: readonly FormKind[] = ["anthropic", "openai", "google", "deepseek", "qwen"];

/** The "Another model…" choice of the model list. */
const OTHER_MODEL = "\u0000other";

/** A provider's current models from our own list: the recommended one first, then its other roles. */
function modelChoices(kind: FormKind): string[] {
  const roles = catalogProviderOfKind(kind)?.roles;
  return [...new Set([roles?.answers, roles?.strongest, roles?.quickTasks])].filter(
    (id): id is string => id !== undefined,
  );
}

/** An OpenAI-compatible server's address as the User typed it, by its host. */
const hostOf = (url: string): string | null => {
  try {
    return new URL(url.trim()).host || null;
  } catch {
    return null;
  }
};

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
  withMoreProviders = false,
  initialKind,
  refused = null,
}: {
  /** Saved providers, to tell the User when a key is already stored. */
  providers: readonly ChatProvider[];
  onSaved(provider: ChatProvider): void;
  /** Settings lists the best-effort providers too, under "More providers"; first run only the first-class ones. */
  withMoreProviders?: boolean;
  /** The provider chosen when the form opens: the one the User came from. DeepSeek in Chinese, else Anthropic. */
  initialKind?: FormKind;
  /** A key its provider refused: the form opens on that provider, says so by the key, and focuses it. */
  refused?: KeyRefusal | null;
}) {
  const t = useT();
  const id = useId();
  const language = useLanguage();
  const first: FormKind =
    refused?.kind ?? initialKind ?? (language === "zh-CN" ? "deepseek" : "anthropic");
  const [kind, setKind] = useState<FormKind | null>(first);
  const [otherModel, setOtherModel] = useState(
    refused !== null && !modelChoices(refused.kind).includes(refused.modelId),
  );
  const [moreOpen, setMoreOpen] = useState(false);
  /** The region of a provider that has several; its first otherwise. */
  const [endpointId, setEndpointId] = useState(refused?.endpoint ?? "");
  const key = useProviderKey();
  const { apiKey } = key;
  const [baseUrl, setBaseUrl] = useState("");
  const [modelId, setModelId] = useState(refused?.modelId ?? suggestedModel(first));
  const [busy, setBusy] = useState<"saving" | null>(null);
  const [test, setTest] = useState<ConnectionTestResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [secretStorage, setSecretStorage] = useState<SecretStorageStatus | null>(null);

  useEffect(() => {
    core.getSecretStorage().then(setSecretStorage, () => undefined);
  }, []);

  const { main, more } = keyProviderGroups(
    withMoreProviders ? providers.map((each) => each.kind) : [],
  );
  const refusedHere = refused !== null && refused.kind === kind;
  const keyField = useRef<HTMLInputElement>(null);
  // A refused key's field is where "Check the key" leads: in view, focused, and under "More
  // providers" opened when its provider is listed there.
  // biome-ignore lint/correctness/useExhaustiveDependencies: once, as the form opens on it.
  useEffect(() => {
    if (!refused) return;
    if (more.includes(refused.kind)) setMoreOpen(true);
    keyField.current?.scrollIntoView({ block: "center" });
    keyField.current?.focus({ preventScroll: true });
  }, []);
  const catalog = kind ? catalogProviderOfKind(kind) : undefined;
  const words = language === "zh-CN" ? "zh" : "en";
  const endpoint = catalog ? endpointOf(catalog, endpointId) : undefined;
  const saved = kind
    ? providers.find((each) => sameServer(each, kind, endpoint?.id ?? "", baseUrl))
    : undefined;
  const keyRequired = kind !== null && kind !== "openai-compatible";
  const hasKey = apiKey.trim() !== "" || Boolean(saved?.hasApiKey);
  const keyBlocked = apiKey.trim() !== "" && secretStorage !== null && !secretStorage.canSave;
  const missing: MessageKey | null =
    kind === null
      ? null
      : kind === "openai-compatible" && baseUrl.trim() === ""
        ? "providers.form.needsAddress"
        : keyRequired && !hasKey
          ? "providers.form.needsKey"
          : modelId.trim() === ""
            ? "providers.form.needsModel"
            : null;
  const complete = kind !== null && missing === null;

  const chip = (option: FormKind) => (
    <label
      key={option}
      className={`inline-flex h-7 cursor-pointer items-center rounded-md border px-2.5 text-[13px] has-[:checked]:border-ink has-[:checked]:bg-wash has-[:focus-visible]:outline-2 ${
        kind === option ? "border-ink font-semibold text-ink" : "border-rule text-ink-secondary"
      }`}
    >
      <input
        type="radio"
        name={`${id}-kind`}
        value={option}
        checked={kind === option}
        onChange={() => choose(option)}
        className="sr-only"
      />
      {option === "qwen" || option === "deepseek"
        ? t(`providers.kind.short.${option}`)
        : t(`providers.kind.${option}`)}
    </label>
  );

  const choose = (next: FormKind) => {
    // A key typed for one provider is never sent to another.
    if (next !== kind) key.clear();
    setKind(next);
    if (more.includes(next)) setMoreOpen(true);
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

  /** One button: the key is tested as it saves, and a failure stays here, in plain words. */
  const run = async () => {
    const request = input();
    if (!request) return;
    setBusy("saving");
    setError(null);
    setTest(null);
    try {
      const result = await core.testChatConnection(request);
      if (!result.ok) {
        setTest(result);
        return;
      }
      const provider = await core.saveChatProvider(request);
      key.clear();
      onSaved(provider);
    } catch (failure) {
      setError(errorMessage(failure, t));
    } finally {
      setBusy(null);
    }
  };

  const submit = (event: FormEvent) => {
    event.preventDefault();
    void run();
  };

  const acceptPlainText = async () => {
    try {
      setSecretStorage(await core.acceptPlainTextSecretStorage());
    } catch (failure) {
      setError(errorMessage(failure, t));
    }
  };

  const siteOf = (url: string | undefined) => {
    try {
      return url ? new URL(url).host : "";
    } catch {
      return "";
    }
  };
  const site = siteOf(endpoint?.keyUrl ?? catalog?.keyUrl);
  const providerName = kind ? t(`providers.kind.${kind}`) : "";
  const choices = kind && kind !== "openai-compatible" ? modelChoices(kind) : [];
  const recommended = choices[0];
  const shownName = (id: string) =>
    modelName(id, { kind: kind ?? undefined, endpoint: endpoint?.id });
  const typedModel = kind === "openai-compatible" || otherModel || choices.length === 0;
  const rowKinds = ROW.filter((each) => main.includes(each) || more.includes(each));
  const underMore = [...main, ...more].filter((each) => !rowKinds.includes(each));

  return (
    <form data-testid="provider-form" onSubmit={submit} className="flex flex-col gap-3">
      <fieldset>
        <legend className="sr-only">{t("providers.form.label")}</legend>
        <div className="flex flex-wrap gap-1">
          {[...rowKinds, ...(moreOpen ? underMore : [])].map(chip)}
          {underMore.length > 0 && !moreOpen && (
            <button
              type="button"
              data-testid="provider-more-toggle"
              aria-expanded={moreOpen}
              onClick={() => setMoreOpen(true)}
              className={`${buttonClass} h-7 px-2.5`}
            >
              {t("providers.form.moreShort")}
            </button>
          )}
        </div>
      </fieldset>

      {kind && (
        <>
          {catalog && catalog.endpoints.length > 1 && (
            <label className={fieldLabelClass}>
              {t("providers.form.region")}
              <select
                value={endpoint?.id ?? ""}
                onChange={(event) => {
                  // A key belongs to its region: it is never sent to another.
                  key.clear();
                  setEndpointId(event.target.value);
                  setTest(null);
                  setError(null);
                }}
                className={inputClass}
              >
                {catalog.endpoints.map((option) => (
                  <option key={option.id} value={option.id}>
                    {option.label[words]}
                  </option>
                ))}
              </select>
            </label>
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
              ref={keyField}
              aria-describedby={refusedHere ? `${id}-refused` : undefined}
              className={inputClass}
            />
            {refusedHere && <KeyRefusedLine id={`${id}-refused`} refused={refused} />}
            {saved?.hasApiKey && (
              <span className={hintClass}>{t("providers.form.apiKeySaved")}</span>
            )}
            {(endpoint?.keyUrl ?? catalog?.keyUrl) && (
              <a
                href={endpoint?.keyUrl ?? catalog?.keyUrl}
                target="_blank"
                rel="noreferrer"
                data-testid="provider-key-link"
                className={`${hintClass} underline`}
              >
                {t("providers.form.whereKey")} ↗
              </a>
            )}
          </label>

          {secretStorage && !secretStorage.canSave && (
            <SecretStorageNotice status={secretStorage} onAccept={() => void acceptPlainText()} />
          )}

          {typedModel ? (
            <label className={fieldLabelClass}>
              {t("providers.form.model")}
              <input
                required
                value={modelId}
                onChange={(event) => setModelId(event.target.value)}
                spellCheck={false}
                data-testid="provider-model-text"
                className={inputClass}
              />
            </label>
          ) : (
            <label className={fieldLabelClass}>
              {t("providers.form.model")}
              <select
                data-testid="provider-model"
                value={choices.includes(modelId) ? modelId : OTHER_MODEL}
                onChange={(event) => {
                  if (event.target.value === OTHER_MODEL) {
                    setOtherModel(true);
                    setModelId("");
                  } else setModelId(event.target.value);
                  setTest(null);
                  setError(null);
                }}
                className={inputClass}
              >
                {choices.map((id) => (
                  <option key={id} value={id} title={id}>
                    {shownName(id)}
                  </option>
                ))}
                <option value={OTHER_MODEL}>{t("providers.form.anotherModel")}</option>
              </select>
              {recommended && modelId === recommended && (
                <span data-testid="provider-model-reason" className={hintClass}>
                  {t("providers.form.recommended", { model: shownName(recommended) })}
                </span>
              )}
            </label>
          )}

          <ProviderSendsLine
            provider={
              kind === "openai-compatible"
                ? (hostOf(baseUrl) ?? t("providers.kind.openai-compatible"))
                : t(`providers.kind.${kind}`)
            }
          />
          <div className="flex flex-wrap items-center gap-2">
            <button
              type="submit"
              data-testid="provider-use-key"
              disabled={!complete || keyBlocked || busy !== null}
              className={primaryButtonClass}
            >
              {busy ? t("providers.form.checking") : t("providers.form.useKey")}
            </button>
            {missing && (
              <span data-testid="provider-use-key-why" className={hintClass}>
                {t(missing)}
              </span>
            )}
          </div>

          {test && !test.ok && (
            <TestResult
              result={test}
              {...(test.error.kind === "auth" && site
                ? { authLine: t("providers.form.badKey", { provider: providerName, site }) }
                : {})}
            />
          )}
          {error && (
            <p role="alert" className={errorTextClass}>
              <ProblemMark />
              {error}
            </p>
          )}
        </>
      )}
    </form>
  );
}

/** Beside a refused key: who refused it, and when. */
function KeyRefusedLine({ id, refused }: { id: string; refused: KeyRefusal }) {
  const t = useT();
  const language = useLanguage();
  const provider = t(`providers.kind.${refused.kind}`);
  const when = refused.at ? dayOf(new Date(refused.at), new Date(), language, t) : null;
  return (
    <span id={id} data-testid="provider-key-refused" role="status" className={hintClass}>
      {when
        ? t("providers.form.keyRefused", { provider, when })
        : t("providers.form.keyRefusedNow", { provider })}
    </span>
  );
}

/**
 * Above the button that chooses the chat model's provider: what goes to it.
 * Choosing it is the consent for all of that (ADR-0022).
 */
export function ProviderSendsLine({ provider }: { provider: string }) {
  const t = useT();
  return (
    <p data-testid="provider-sends" className={hintClass}>
      {t("providers.sends", { provider })}
    </p>
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

/** A connection test's outcome. A test sends nothing of the User's, so it asks nothing (ADR-0022). */
export function TestResult({
  result,
  authLine,
}: {
  result: ConnectionTestResult;
  /** Says a refused key in the provider's own terms, in place of the general line. */
  authLine?: string;
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
      <p>
        <ProblemMark />
        {authLine ?? t(testErrorKey(result.error.kind))}
      </p>
      {/* The provider's own words help with its errors. */}
      {result.error.message && (
        <p className="mt-1 text-[12px] leading-[18px] break-words text-ink-meta">
          {readableMessage(result.error.message)}
        </p>
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
