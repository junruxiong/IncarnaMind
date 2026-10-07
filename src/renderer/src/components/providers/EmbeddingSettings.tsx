import { type FormEvent, useEffect, useId, useState } from "react";
import {
  type EmbeddingConnectionTestResult,
  type EmbeddingProvider,
  type EmbeddingProviderKind,
  type EmbeddingSettings,
  embeddingProviderKinds,
  type SaveEmbeddingProviderInput,
  type SecretStorageStatus,
  SUGGESTED_EMBEDDING_MODELS,
} from "../../../../core/api";
import type { MessageKey, MessageParams } from "../../../../shared/i18n";
import { core } from "../../core";
import { errorMessage } from "../../errors";
import { useT } from "../../i18n";
import { useAppStore } from "../../store";
import { useModal } from "../useModal";
import { SecretStorageNotice, TestResult } from "./ProviderForm";
import { buttonClass, inputClass, primaryButtonClass, testErrorKey } from "./shared";

type Translate = (key: MessageKey, params?: MessageParams) => string;

/** Ollama's default local address, shown as the URL's placeholder. */
const OLLAMA_URL = "http://127.0.0.1:11434";

/** Providers whose requests always leave this computer. */
const CLOUD_KINDS: ReadonlySet<EmbeddingProviderKind> = new Set(["openai", "google"]);

/** Loopback addresses: a server there runs on this computer. */
function isOnThisComputer(url: string): boolean {
  try {
    const host = new URL(url).hostname.toLowerCase().replace(/^\[|\]$/g, "");
    return (
      host === "localhost" ||
      host.endsWith(".localhost") ||
      host === "::1" ||
      host === "0.0.0.0" ||
      /^127(\.\d{1,3}){3}$/.test(host)
    );
  } catch {
    return false;
  }
}

/** Where a provider the form describes would send Document text, or null when it stays here. */
function cloudServiceOf(kind: EmbeddingProviderKind, baseUrl: string, t: Translate) {
  if (CLOUD_KINDS.has(kind)) return t(`embeddingProviders.kind.${kind}`);
  if (kind === "built-in") return null;
  const url = baseUrl.trim() || (kind === "ollama" ? OLLAMA_URL : "");
  if (!url || isOnThisComputer(url)) return null;
  try {
    return new URL(url).host;
  } catch {
    return url;
  }
}

/** "Built-in model (multilingual-e5-small)", "OpenAI · text-embedding-3-small". */
export function embeddingProviderLabel(provider: EmbeddingProvider, t: Translate): string {
  if (provider.kind === "built-in") {
    return t("embeddingProviders.settings.builtIn", { model: provider.modelId });
  }
  const kind = t(`embeddingProviders.kind.${provider.kind}`);
  const server =
    provider.kind === "openai-compatible" || provider.kind === "ollama"
      ? ` (${provider.service?.name ?? t("providers.settings.local")})`
      : "";
  return `${kind}${server} · ${provider.modelId}`;
}

/**
 * Settings → Document search: the embedding model search uses (the built-in
 * one, or a provider chosen instead), switching it with a warning, its error
 * and rebuild, and local mode.
 */
export function EmbeddingSettingsSection() {
  const t = useT();
  const embedding = useAppStore((state) => state.embedding);
  const retry = useAppStore((state) => state.retryEmbedding);
  const [editing, setEditing] = useState(false);

  if (!embedding) return null;
  const { provider, error, rebuild } = embedding;
  return (
    <section data-testid="embedding-settings">
      <h3 className="mb-1 text-sm font-medium">{t("embeddingProviders.settings.title")}</h3>
      <p className="text-sm text-gray-600">{t("embeddingProviders.settings.body")}</p>

      <div className="mt-2 flex flex-col gap-1">
        <p data-testid="embedding-current" className="text-sm">
          <span className="text-gray-600">{t("embeddingProviders.settings.current")}</span>
          {embeddingProviderLabel(provider, t)}
        </p>
        <p className="text-xs text-gray-500">
          {provider.service
            ? t("embeddingProviders.settings.sendsTo", { service: provider.service.name })
            : t("embeddingProviders.settings.local")}
        </p>
        {error && (
          <div
            role="alert"
            className="flex items-start gap-2 rounded-[9px] bg-amber-50 p-2 text-sm text-amber-900"
          >
            <p className="min-w-0 flex-1 break-words" title={error.message}>
              {t("embeddingProviders.settings.error", {
                reason: t(testErrorKey(error.kind)),
              })}
            </p>
            <button type="button" onClick={() => void retry()} className="shrink-0 font-medium">
              {t("embeddingProviders.settings.retry")}
            </button>
          </div>
        )}
        {rebuild && (
          <p data-testid="embedding-rebuild" className="text-sm text-gray-600">
            {t("embeddingProviders.rebuild.title", { done: rebuild.done, total: rebuild.total })}
          </p>
        )}
      </div>

      {editing ? (
        <EmbeddingForm current={embedding} onDone={() => setEditing(false)} />
      ) : (
        <button
          type="button"
          data-testid="embedding-change"
          onClick={() => setEditing(true)}
          className={`${buttonClass} mt-2`}
        >
          {t("embeddingProviders.settings.change")}
        </button>
      )}

      <LocalOnlySetting localOnly={embedding.localOnly} />
    </section>
  );
}

/** "Keep everything on this computer", and what it doesn't cover. */
function LocalOnlySetting({ localOnly }: { localOnly: boolean }) {
  const t = useT();
  const readiness = useAppStore((state) => state.chatReadiness);
  const [busy, setBusy] = useState(false);
  const chatService = readiness && "provider" in readiness ? readiness.provider.service : null;

  const change = async (enabled: boolean) => {
    setBusy(true);
    try {
      useAppStore.setState({ embedding: await core.setLocalOnly(enabled) });
    } catch (failure) {
      useAppStore.setState({ actionError: errorMessage(failure) });
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="mt-3">
      <label className="flex items-start gap-2 text-sm">
        <input
          type="checkbox"
          data-testid="local-only"
          checked={localOnly}
          disabled={busy}
          onChange={(event) => void change(event.target.checked)}
          className="mt-[3px]"
        />
        <span>
          <span className="font-medium">{t("embeddingProviders.localOnly.label")}</span>
          <span className="block text-xs text-gray-500">
            {t("embeddingProviders.localOnly.body")}
          </span>
        </span>
      </label>
      {localOnly && chatService && (
        <p className="mt-1 rounded-[9px] bg-amber-50 p-2 text-xs text-amber-900">
          {t("embeddingProviders.localOnly.cloudChat", { service: chatService.name })}
        </p>
      )}
    </div>
  );
}

/** Pick a provider, a key, a server and a model; test; then switch, after a warning. */
function EmbeddingForm({ current, onDone }: { current: EmbeddingSettings; onDone(): void }) {
  const t = useT();
  const id = useId();
  const [kind, setKind] = useState<EmbeddingProviderKind>(current.provider.kind);
  const [baseUrl, setBaseUrl] = useState(current.provider.baseUrl ?? "");
  const [apiKey, setApiKey] = useState("");
  const [modelId, setModelId] = useState(
    current.provider.kind === "built-in" ? "" : current.provider.modelId,
  );
  const [busy, setBusy] = useState<"testing" | "saving" | null>(null);
  const [test, setTest] = useState<EmbeddingConnectionTestResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [confirming, setConfirming] = useState(false);
  const [secretStorage, setSecretStorage] = useState<SecretStorageStatus | null>(null);

  useEffect(() => {
    core.getSecretStorage().then(setSecretStorage, () => undefined);
  }, []);

  const saved = current.provider;
  const sameServer =
    saved.kind === kind && (saved.baseUrl ?? "") === baseUrl.trim().replace(/\/+$/, "");
  const takesKey = kind === "openai" || kind === "google" || kind === "openai-compatible";
  const keyRequired = kind === "openai" || kind === "google";
  const hasKey = apiKey.trim() !== "" || (sameServer && saved.hasApiKey);
  const keyBlocked = apiKey.trim() !== "" && secretStorage !== null && !secretStorage.canSave;
  const complete =
    kind === "built-in" ||
    (modelId.trim() !== "" &&
      (kind !== "openai-compatible" || baseUrl.trim() !== "") &&
      (!keyRequired || hasKey));
  const unchanged =
    sameServer && apiKey.trim() === "" && (kind === "built-in" || modelId.trim() === saved.modelId);
  const cloudService = cloudServiceOf(kind, baseUrl, t);

  const choose = (next: EmbeddingProviderKind) => {
    setKind(next);
    setBaseUrl(next === saved.kind ? (saved.baseUrl ?? "") : "");
    setModelId(
      next === "built-in"
        ? ""
        : next === saved.kind
          ? saved.modelId
          : SUGGESTED_EMBEDDING_MODELS[next],
    );
    setTest(null);
    setError(null);
  };

  const input = (): SaveEmbeddingProviderInput =>
    kind === "built-in"
      ? { kind }
      : {
          kind,
          modelId: modelId.trim(),
          ...((kind === "openai-compatible" || kind === "ollama") && baseUrl.trim()
            ? { baseUrl: baseUrl.trim() }
            : {}),
          ...(takesKey && apiKey.trim() ? { apiKey: apiKey.trim() } : {}),
        };

  const runTest = async () => {
    setBusy("testing");
    setError(null);
    setTest(null);
    try {
      setTest(await core.testEmbeddingConnection(input()));
    } catch (failure) {
      setError(errorMessage(failure));
    } finally {
      setBusy(null);
    }
  };

  const save = async () => {
    setConfirming(false);
    setBusy("saving");
    setError(null);
    try {
      useAppStore.setState({ embedding: await core.saveEmbeddingProvider(input()) });
      setApiKey("");
      onDone();
    } catch (failure) {
      setError(errorMessage(failure));
    } finally {
      setBusy(null);
    }
  };

  const submit = (event: FormEvent) => {
    event.preventDefault();
    setConfirming(true);
  };

  const acceptPlainText = async () => {
    try {
      setSecretStorage(await core.acceptPlainTextSecretStorage());
    } catch (failure) {
      setError(errorMessage(failure));
    }
  };

  return (
    <form data-testid="embedding-form" onSubmit={submit} className="mt-3 flex flex-col gap-3">
      <fieldset>
        <legend className="mb-1 text-sm text-gray-600">{t("embeddingProviders.form.label")}</legend>
        <div className="grid grid-cols-2 gap-2">
          {embeddingProviderKinds.map((option) => {
            const blocked = current.localOnly && CLOUD_KINDS.has(option);
            return (
              <label
                key={option}
                className={`flex items-center gap-2 rounded-[9px] border px-3 py-2 text-sm ${
                  blocked
                    ? "cursor-default border-gray-200 text-gray-400"
                    : kind === option
                      ? "cursor-pointer border-gray-800"
                      : "cursor-pointer border-gray-300 hover:bg-gray-50"
                }`}
              >
                <input
                  type="radio"
                  name={`${id}-kind`}
                  value={option}
                  checked={kind === option}
                  disabled={blocked}
                  onChange={() => choose(option)}
                />
                {t(`embeddingProviders.kind.${option}`)}
              </label>
            );
          })}
        </div>
        {current.localOnly && (
          <p className="mt-1 text-xs text-gray-500">{t("embeddingProviders.form.localOnly")}</p>
        )}
      </fieldset>

      {kind === "built-in" && (
        <p className="text-sm text-gray-600">{t("embeddingProviders.form.builtInHint")}</p>
      )}

      {(kind === "openai-compatible" || kind === "ollama") && (
        <label className="text-sm text-gray-600">
          {kind === "ollama" ? t("embeddingProviders.form.ollamaUrl") : t("providers.form.baseUrl")}
          <input
            type="url"
            required={kind === "openai-compatible"}
            value={baseUrl}
            onChange={(event) => setBaseUrl(event.target.value)}
            placeholder={kind === "ollama" ? OLLAMA_URL : "https://"}
            spellCheck={false}
            className={inputClass}
          />
          <span className="mt-1 block text-xs text-gray-500">
            {kind === "ollama"
              ? t("embeddingProviders.form.ollamaHint")
              : t("providers.form.baseUrlHint")}
          </span>
        </label>
      )}

      {takesKey && (
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
          {sameServer && saved.hasApiKey && (
            <span className="mt-1 block text-xs text-gray-500">
              {t("providers.form.apiKeySaved")}
            </span>
          )}
        </label>
      )}

      {takesKey && secretStorage && !secretStorage.canSave && (
        <SecretStorageNotice status={secretStorage} onAccept={() => void acceptPlainText()} />
      )}

      {kind !== "built-in" && (
        <label className="text-sm text-gray-600">
          {t("embeddingProviders.form.model")}
          <input
            required
            value={modelId}
            onChange={(event) => setModelId(event.target.value)}
            spellCheck={false}
            className={inputClass}
          />
        </label>
      )}

      <div className="flex flex-wrap items-center gap-2">
        {kind !== "built-in" && (
          <button
            type="button"
            disabled={!complete || busy !== null}
            onClick={() => void runTest()}
            className={buttonClass}
          >
            {busy === "testing" ? t("providers.form.testing") : t("providers.form.test")}
          </button>
        )}
        <button
          type="submit"
          data-testid="embedding-switch"
          disabled={!complete || unchanged || keyBlocked || busy !== null}
          className={primaryButtonClass}
        >
          {busy === "saving" ? t("providers.form.saving") : t("embeddingProviders.form.switch")}
        </button>
        <button type="button" onClick={onDone} className={buttonClass}>
          {t("providers.settings.cancel")}
        </button>
      </div>

      {test &&
        (test.ok ? (
          <p data-testid="connection-test" className="text-sm text-emerald-700">
            {t("embeddingProviders.test.ok", { dimensions: test.dimensions })}
          </p>
        ) : (
          <TestResult result={test} />
        ))}
      {error && (
        <p role="alert" className="text-sm break-words text-red-700">
          {error}
        </p>
      )}

      <SwitchConfirmation
        open={confirming}
        provider={
          kind === "built-in"
            ? t("embeddingProviders.kind.built-in")
            : `${t(`embeddingProviders.kind.${kind}`)} · ${modelId.trim()}`
        }
        cloudService={cloudService}
        onCancel={() => setConfirming(false)}
        onConfirm={() => void save()}
      />
    </form>
  );
}

/**
 * The warning before a switch: every Document is processed again, and a
 * cloud provider receives all their text and every search. Cancelling changes nothing.
 */
function SwitchConfirmation({
  open,
  provider,
  cloudService,
  onCancel,
  onConfirm,
}: {
  open: boolean;
  provider: string;
  cloudService: string | null;
  onCancel(): void;
  onConfirm(): void;
}) {
  const t = useT();
  const dialog = useModal(open);
  const documentCount = useAppStore((state) => state.documents.length);
  return (
    <dialog
      ref={dialog}
      data-testid="embedding-confirm"
      aria-labelledby="embedding-confirm-title"
      onClose={(event) => {
        // React passes "close" up its tree: it mustn't close the Settings dialog this one sits in.
        event.stopPropagation();
        onCancel();
      }}
      className="m-auto w-[30rem] max-w-[calc(100vw-2rem)] rounded-[9px] bg-white p-5 text-gray-800 shadow-custom-focus backdrop:bg-black/30"
    >
      {open && (
        <>
          <h2 id="embedding-confirm-title" className="text-lg font-semibold">
            {t("embeddingProviders.confirm.title", { provider })}
          </h2>
          <p className="mt-2 text-sm text-gray-700">
            {t("embeddingProviders.confirm.reprocess", { count: documentCount })}
          </p>
          {cloudService ? (
            <p
              data-testid="embedding-confirm-cloud"
              className="mt-2 text-sm font-medium text-amber-900"
            >
              {t("embeddingProviders.confirm.cloud", { service: cloudService })}
            </p>
          ) : (
            <p className="mt-2 text-sm text-gray-700">{t("embeddingProviders.confirm.local")}</p>
          )}
          <div className="mt-4 flex justify-end gap-2">
            <button
              type="button"
              data-testid="embedding-confirm-cancel"
              onClick={onCancel}
              className={buttonClass}
            >
              {t("embeddingProviders.confirm.cancel")}
            </button>
            <button
              type="button"
              data-testid="embedding-confirm-switch"
              onClick={onConfirm}
              className={primaryButtonClass}
            >
              {t("embeddingProviders.confirm.switch")}
            </button>
          </div>
        </>
      )}
    </dialog>
  );
}
