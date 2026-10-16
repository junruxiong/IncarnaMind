import { type FormEvent, useEffect, useId, useRef, useState } from "react";
import type {
  ChatProvider,
  ConnectionTestResult,
  SecretStorageStatus,
  TestChatConnectionInput,
} from "../../../../core/api";
import type { DataUse } from "../../../../core/providers/catalog";
import { catalogMenu } from "../../../../core/providers/catalog";
import {
  catalogProvider,
  endpointOf,
  fieldAccepts,
  fieldUrl,
  isPreset,
  KEY_PROVIDERS,
  keyProviderGroups,
  kindOfCatalogId,
  untestedForCitations,
} from "../../../../core/providers/catalog/providers";
import type { Role } from "../../../../core/providers/catalog/types";
import { serviceForUrl } from "../../../../core/providers/service";
import type { MessageKey } from "../../../../shared/i18n";
import { gap } from "../../../../shared/i18n/join";
import { modelName } from "../../../../shared/modelName";
import { core } from "../../core";
import { dayOf } from "../../dates";
import { errorMessage } from "../../errors";
import { useLanguage, useT } from "../../i18n";
import type { KeyRefusal } from "../../keyRefusals";
import { useAppStore } from "../../store";
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
import { KeyHelp } from "./KeyHelp";
import { LocalServerLine, useLocalServer } from "./LocalServerLine";
import { keyProviderName, testErrorKey, useProviderKey } from "./shared";
import { useJevSettings } from "./useJevSettings";

type CatalogId = string;

/**
 * Prefilled when a provider is picked: its Answers model in the catalog, none
 * for a server. The User can type any model their account has.
 */
const suggestedModel = (kind: CatalogId): string => catalogProvider(kind)?.roles.answers ?? "";

const isSuggestion = (model: string) =>
  KEY_PROVIDERS.some((kind) => suggestedModel(kind) === model);

/** The providers shown as one row of choices; the rest are under "More…". */
const ROW: readonly CatalogId[] = ["anthropic", "openai", "google", "deepseek", "qwen"];

/** What a provider's address field holds when it is chosen. */
const fieldDefault = (kind: CatalogId): string => {
  const provider = catalogProvider(kind);
  return (provider && endpointOf(provider, "")?.field?.default) ?? "";
};

/** The "Another model…" choice of the model list. */
const OTHER_MODEL = "\u0000other";

const ROLE_HINTS = {
  answers: "providers.form.hint.answers",
  strongest: "providers.form.hint.strongest",
  quickTasks: "providers.form.hint.quickTasks",
  images: "providers.form.hint.images",
} as const satisfies Record<Exclude<Role, "tagging">, MessageKey>;

function roleHint(kind: CatalogId, modelId: string): MessageKey | undefined {
  const roles = catalogProvider(kind)?.roles;
  const role = (["answers", "strongest", "quickTasks", "images"] as const).find(
    (each) => roles?.[each] === modelId,
  );
  return role && ROLE_HINTS[role];
}

/** An OpenAI-compatible server's address as the User typed it, by its host. */
const hostOf = (url: string): string | null => {
  try {
    return new URL(url.trim()).host || null;
  } catch {
    return null;
  }
};

const sameServer = (
  provider: ChatProvider,
  kind: CatalogId,
  endpoint: string,
  baseUrl: string,
  fieldAddress: string | null,
) =>
  (provider.catalogId ?? provider.kind) === kind &&
  (fieldAddress !== null
    ? provider.baseUrl === fieldAddress
    : kind === "openai-compatible"
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
  only,
  refused = null,
  standalone = false,
}: {
  /** Saved providers, to tell the User when a key is already stored. */
  providers: readonly ChatProvider[];
  onSaved(provider: ChatProvider): void;
  /** Settings lists the best-effort providers too, under "More providers"; first run only the first-class ones. */
  withMoreProviders?: boolean;
  /** The provider chosen when the form opens: the one the User came from. DeepSeek in Chinese, else Anthropic. */
  initialKind?: CatalogId;
  /** The one provider this form sets up, with no chips to change it: a local server's row. */
  only?: CatalogId;
  /** A key its provider refused: the form opens on that provider, says so by the key, and focuses it. */
  refused?: KeyRefusal | null;
  /** The setup card's key step: the form is titled with the company picked, and opens on the key. */
  standalone?: boolean;
}) {
  const t = useT();
  const id = useId();
  const language = useLanguage();
  const first: CatalogId =
    only ?? refused?.catalogId ?? initialKind ?? (language === "zh-CN" ? "deepseek" : "anthropic");
  const [kind, setKind] = useState<CatalogId | null>(first);
  const [otherModel, setOtherModel] = useState(
    refused !== null && !catalogMenu(refused.catalogId, refused.endpoint).includes(refused.modelId),
  );
  const [moreOpen, setMoreOpen] = useState(false);
  /** The region of a provider that has several; its first otherwise. */
  const [endpointId, setEndpointId] = useState(refused?.endpoint ?? "");
  const key = useProviderKey();
  const { apiKey } = key;
  const [fieldValue, setFieldValue] = useState(() => fieldDefault(first));
  const [baseUrl, setBaseUrl] = useState("");
  const [modelId, setModelId] = useState(refused?.modelId ?? suggestedModel(first));
  const [busy, setBusy] = useState<"saving" | null>(null);
  const [test, setTest] = useState<ConnectionTestResult | null>(null);
  const [jev] = useJevSettings();
  // Automatic tags with this key's provider unless a Jev key comes first, or nothing may leave the computer.
  const tagsAutomatically =
    useAppStore(
      (state) =>
        (state.library?.settings.classifier.kind ?? "automatic") === "automatic" &&
        state.embedding?.localOnly !== true,
    ) && !jev?.enabled;
  const [error, setError] = useState<string | null>(null);
  const [secretStorage, setSecretStorage] = useState<SecretStorageStatus | null>(null);

  useEffect(() => {
    core.getSecretStorage().then(setSecretStorage, () => undefined);
  }, []);

  const { main, more } = keyProviderGroups(
    withMoreProviders ? providers.map((each) => each.catalogId ?? each.kind) : [],
  );
  const refusedHere = refused !== null && refused.catalogId === kind;
  const keyField = useRef<HTMLInputElement>(null);
  // A refused key's field is where "Check the key" leads: in view, focused, and under "More
  // providers" opened when its provider is listed there.
  // biome-ignore lint/correctness/useExhaustiveDependencies: once, as the form opens on it.
  useEffect(() => {
    if (!refused) {
      if (standalone) keyField.current?.focus();
      return;
    }
    if (more.includes(refused.catalogId)) setMoreOpen(true);
    keyField.current?.scrollIntoView({ block: "center" });
    keyField.current?.focus({ preventScroll: true });
  }, []);
  const catalog = kind ? catalogProvider(kind) : undefined;
  const local = catalog?.group === "local";
  const server = useLocalServer(local ? kind : null);
  const words = language === "zh-CN" ? "zh" : "en";
  const endpoint = catalog ? endpointOf(catalog, endpointId) : undefined;
  const field = endpoint?.field;
  const fieldUsed = (fieldValue || field?.choices?.[0]?.value || "").trim();
  const fieldAddress =
    endpoint && field && fieldAccepts(field, fieldUsed) ? fieldUrl(endpoint, fieldUsed) : null;
  // An address elsewhere is a service like any other: it says what goes to it, and choosing it is the consent.
  const away = local && fieldAddress !== null && serviceForUrl(fieldAddress) !== null;
  const saved = kind
    ? providers.find((each) => sameServer(each, kind, endpoint?.id ?? "", baseUrl, fieldAddress))
    : undefined;
  const keyRequired = catalog?.apiKey === "required";
  const hasKey = apiKey.trim() !== "" || Boolean(saved?.hasApiKey);
  const keyBlocked = apiKey.trim() !== "" && secretStorage !== null && !secretStorage.canSave;
  const missing: MessageKey | null =
    kind === null
      ? null
      : kind === "openai-compatible" && baseUrl.trim() === ""
        ? "providers.form.needsAddress"
        : field && fieldAddress === null
          ? "providers.form.needsField"
          : local && server.status?.running === false
            ? "providers.form.needsServer"
            : keyRequired && !hasKey
              ? "providers.form.needsKey"
              : modelId.trim() === ""
                ? "providers.form.needsModel"
                : null;
  const complete = kind !== null && missing === null && (!local || server.status?.running === true);

  const shortName = (option: CatalogId) =>
    option === "qwen"
      ? t("providers.kind.short.qwen")
      : option === "deepseek"
        ? t("providers.kind.short.deepseek")
        : keyProviderName(option, t, language);

  const chip = (option: CatalogId) => (
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
      {shortName(option)}
    </label>
  );

  const choose = (next: CatalogId) => {
    // A key typed for one provider is never sent to another.
    if (next !== kind) key.clear();
    setKind(next);
    if (more.includes(next)) setMoreOpen(true);
    setEndpointId("");
    setFieldValue(fieldDefault(next));
    setModelId((current) =>
      current === "" || isSuggestion(current) ? suggestedModel(next) : current,
    );
    setTest(null);
    setError(null);
  };

  useEffect(() => {
    const models = server.status?.running ? server.status.models : [];
    const [first] = models;
    if (first !== undefined) setModelId((current) => (models.includes(current) ? current : first));
  }, [server.status]);

  const changeField = (value: string) => {
    setFieldValue(value);
    setTest(null);
    setError(null);
  };

  const input = (): TestChatConnectionInput | null => {
    if (!kind) return null;
    return {
      kind: kindOfCatalogId(kind),
      ...(isPreset(catalogProvider(kind)) && { catalogId: kind }),
      modelId: modelId.trim(),
      ...(endpoint ? { endpoint: endpoint.id } : {}),
      ...(field ? { endpointValue: fieldUsed } : {}),
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
  const providerName = kind ? keyProviderName(kind, t, language) : "";
  const listed = server.status?.running ? server.status.models : [];
  const choices =
    kind && kind !== "openai-compatible" ? (local ? listed : catalogMenu(kind, endpoint?.id)) : [];
  const recommended = choices[0];
  const shownName = (id: string) =>
    modelName(id, { kind: kind ? kindOfCatalogId(kind) : undefined, endpoint: endpoint?.id });
  const optionName = (id: string) => {
    const hint = kind ? roleHint(kind, id) : undefined;
    return hint
      ? t("providers.form.modelHint", { model: shownName(id), hint: t(hint) })
      : shownName(id);
  };
  const example = catalog?.roles.answers;
  const modelIdField = (
    <input
      id={`${id}-model-id`}
      required
      value={modelId}
      onChange={(event) => setModelId(event.target.value)}
      spellCheck={false}
      placeholder={example && t("providers.form.modelExample", { model: example })}
      data-testid="provider-model-text"
      className={inputClass}
    />
  );
  // First run offers only the first-class providers (ADR-0019); Settings offers them all.
  const offered = withMoreProviders ? [...main, ...more] : main;
  const rowKinds = ROW.filter((each) => offered.includes(each));
  const underMore = offered.filter((each) => !rowKinds.includes(each));

  return (
    <form data-testid="provider-form" onSubmit={submit} className="flex flex-col gap-3">
      {standalone && kind && (
        <h3
          data-testid="provider-form-title"
          className="text-[15px] leading-6 font-semibold text-ink"
        >
          {kind === "openai-compatible"
            ? t("providers.form.title.server")
            : t("providers.form.title", { provider: shortName(kind) })}
        </h3>
      )}
      {!only && (
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
      )}

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
                  if (!otherModel) {
                    setModelId((current) =>
                      catalogMenu(kind, event.target.value).includes(current)
                        ? current
                        : suggestedModel(kind),
                    );
                  }
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

          {field && (
            <label htmlFor={`${id}-field`} className={fieldLabelClass}>
              {field.label[words]}
              {field.choices ? (
                <select
                  id={`${id}-field`}
                  data-testid="provider-field"
                  value={fieldUsed}
                  onChange={(event) => changeField(event.target.value)}
                  className={inputClass}
                >
                  {field.choices.map((option) => (
                    <option key={option.value} value={option.value}>
                      {`${option.label[words]} · ${option.value}`}
                    </option>
                  ))}
                </select>
              ) : (
                <input
                  id={`${id}-field`}
                  data-testid="provider-field"
                  value={fieldValue}
                  onChange={(event) => changeField(event.target.value)}
                  onBlur={() => local && void server.detect(apiKey, fieldUsed)}
                  autoComplete="off"
                  spellCheck={false}
                  className={inputClass}
                />
              )}
              {field.hint && (
                <span data-testid="provider-field-hint" className={hintClass}>
                  {field.hint[words]}
                </span>
              )}
            </label>
          )}

          {local && (
            <LocalServerLine
              provider={keyProviderName(kind, t, language)}
              status={server.status}
              away={away}
              onCheckAgain={() => void server.detect(apiKey, fieldUsed)}
            />
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
              onBlur={() => local && apiKey.trim() !== "" && void server.detect(apiKey, fieldUsed)}
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
          </label>
          {catalog && (
            <KeyHelp
              key={`${kind}:${endpoint?.id}`}
              catalog={catalog}
              keyUrl={endpoint?.keyUrl ?? catalog.keyUrl}
              provider={shortName(kind)}
              firstRun={standalone}
            />
          )}

          {secretStorage && !secretStorage.canSave && (
            <SecretStorageNotice status={secretStorage} onAccept={() => void acceptPlainText()} />
          )}

          {choices.length === 0 ? (
            <div>
              <label htmlFor={`${id}-model-id`} className={fieldLabelClass}>
                {t("providers.form.model")}
              </label>
              {modelIdField}
            </div>
          ) : (
            <div>
              <div className="flex items-baseline justify-between gap-3">
                <label htmlFor={`${id}-model`} className={fieldLabelClass}>
                  {t("providers.form.model")}
                </label>
                {!standalone && catalog?.docsUrl && (
                  <a
                    href={catalog.docsUrl}
                    target="_blank"
                    rel="noreferrer"
                    data-testid="provider-docs-link"
                    className="text-[12px] leading-5 text-ink-meta underline"
                  >
                    {t("providers.form.modelsAndPrices")} ↗
                  </a>
                )}
              </div>
              <select
                id={`${id}-model`}
                data-testid="provider-model"
                value={!otherModel && choices.includes(modelId) ? modelId : OTHER_MODEL}
                onChange={(event) => {
                  const chosen = event.target.value;
                  setOtherModel(chosen === OTHER_MODEL);
                  setModelId(chosen === OTHER_MODEL ? "" : chosen);
                  setTest(null);
                  setError(null);
                }}
                className={inputClass}
              >
                {choices.map((id) => (
                  <option key={id} value={id} title={id}>
                    {optionName(id)}
                  </option>
                ))}
                <option value={OTHER_MODEL}>{t("providers.form.anotherModel")}</option>
              </select>
              {otherModel ? (
                <div className="mt-2">
                  <label htmlFor={`${id}-model-id`} className="sr-only">
                    {t("providers.form.modelId")}
                  </label>
                  {modelIdField}
                  <span className={hintClass}>{t("providers.form.modelIdHint")}</span>
                </div>
              ) : (
                untestedForCitations(catalog) &&
                modelId === recommended && (
                  <span data-testid="provider-model-reason" className={hintClass}>
                    {t("providers.untested")}
                  </span>
                )
              )}
            </div>
          )}

          {(!local || away) && (
            <ProviderSendsLine
              provider={
                kind === "openai-compatible"
                  ? (hostOf(baseUrl) ?? t("providers.kind.openai-compatible"))
                  : providerName
              }
              tagsOnSave={tagsAutomatically}
              dataUse={catalog?.dataUse}
            />
          )}
          <div className="flex flex-wrap items-center gap-2">
            <button
              type="submit"
              data-testid="provider-use-key"
              disabled={!complete || keyBlocked || busy !== null}
              className={primaryButtonClass}
            >
              {busy
                ? t("providers.form.checking")
                : t(local && !keyRequired ? "providers.form.save" : "providers.form.useKey")}
            </button>
            {missing && (
              <span data-testid="provider-use-key-why" className={hintClass}>
                {missing === "providers.form.needsField" && field
                  ? t(missing, { field: field.label[words] })
                  : t(missing, { provider: providerName })}
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
  const provider = keyProviderName(refused.catalogId, t, language);
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
 * Choosing it is the consent for all of that (ADR-0022). `tagsOnSave`: the
 * Tagging setting is Automatic, so saving the key starts tagging with it.
 */
export function ProviderSendsLine({
  provider,
  dataUse,
  tagsOnSave = false,
}: {
  provider: string;
  dataUse?: DataUse;
  tagsOnSave?: boolean;
}) {
  const t = useT();
  const language = useLanguage();
  const words = language === "zh-CN" ? "zh" : "en";
  const sends = t(tagsOnSave ? "providers.sends.tagsOnSave" : "providers.sends", { provider });
  const summary = dataUse && t("providers.form.dataUse", { summary: dataUse.summary[words] });
  const terms = `${t("providers.form.terms")} ↗`;
  return (
    <div className="flex flex-col">
      <p data-testid="provider-sends" className={hintClass}>
        {sends}
      </p>
      {summary && (
        <p data-testid="provider-data-use" className={`${hintClass} mt-0`}>
          {summary}
          {dataUse?.url && (
            <>
              {gap(language, summary, terms)}
              <a href={dataUse.url} target="_blank" rel="noreferrer" className="underline">
                {terms}
              </a>
            </>
          )}
        </p>
      )}
    </div>
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
