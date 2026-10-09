import { useEffect, useState } from "react";
import type { ChatProvider, LibraryClassifier, LibrarySettings } from "../../../core/api";
import { supportsLibraryImages } from "../../../shared/libraryModels";
import { core } from "../core";
import { errorMessage as messageOf } from "../errors";
import { useT } from "../i18n";
import { useAppStore } from "../store";
import { buttonStyle, errorTextClass, fieldLabelClass, inputClass } from "./ui";

type Action = (work: () => Promise<unknown>) => Promise<boolean>;

export function OrganizationSettings() {
  const t = useT();
  const snapshot = useAppStore((state) => state.library);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [saved, setSaved] = useState(false);
  const act: Action = async (work) => {
    setBusy(true);
    setError(null);
    setSaved(false);
    try {
      await work();
      await useAppStore.getState().refreshLibrary();
      setSaved(true);
      return true;
    } catch (failure) {
      setError(messageOf(failure));
      return false;
    } finally {
      setBusy(false);
    }
  };
  return (
    <section>
      <p className="text-ink-secondary">{t("library.settingsIntro")}</p>
      {error && (
        <p role="alert" className={errorTextClass}>
          {error}
        </p>
      )}
      {snapshot && (
        <ClassifierForm
          key={JSON.stringify(snapshot.settings)}
          settings={snapshot.settings}
          act={act}
          busy={busy}
        />
      )}
      {saved && (
        <p role="status" className="mt-3 text-ink-meta">
          {t("library.settingsSaved")}
        </p>
      )}
    </section>
  );
}

function ClassifierForm({
  settings,
  act,
  busy,
}: {
  settings: LibrarySettings;
  act: Action;
  busy: boolean;
}) {
  const t = useT();
  const [providers, setProviders] = useState<ChatProvider[]>([]);
  const initial = settings.classifier;
  const [selection, setSelection] = useState(
    initial?.kind === "chat" ? initial.choice.providerId : (initial?.kind ?? ""),
  );
  const [modelId, setModelId] = useState(
    initial?.kind === "chat"
      ? initial.choice.modelId
      : initial?.kind === "ollama"
        ? initial.modelId
        : "",
  );
  const [baseUrl, setBaseUrl] = useState(
    initial?.kind === "ollama" || initial?.kind === "auto"
      ? initial.baseUrl
      : "http://localhost:11434",
  );
  const [automatic, setAutomatic] = useState(settings.automatic);
  const [usePageImages, setUsePageImages] = useState(
    initial?.kind === "ollama" && initial.usePageImages === true,
  );
  const [jevEnabled, setJevEnabled] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);
  useEffect(() => {
    let stopped = false;
    const reload = () => {
      void Promise.all([core.listChatProviders(), core.getJevSettings()]).then(
        ([list, jev]) => {
          if (!stopped) {
            setProviders(list);
            setJevEnabled(jev.enabled);
          }
        },
        (failure) => {
          if (!stopped) setLoadError(messageOf(failure));
        },
      );
    };
    reload();
    const stopChat = core.on("chatReadiness.changed", reload);
    const stopJev = core.on("jev.changed", reload);
    return () => {
      stopped = true;
      stopChat();
      stopJev();
    };
  }, []);
  return (
    <form
      className="mt-4 max-w-xl"
      onSubmit={(event) => {
        event.preventDefault();
        let classifier: LibraryClassifier | null = null;
        if (selection === "jev") classifier = { kind: "jev" };
        else if (selection === "auto") classifier = { kind: "auto", baseUrl };
        else if (selection === "ollama")
          classifier = {
            kind: "ollama",
            baseUrl,
            modelId: modelId.trim(),
            usePageImages: supportsLibraryImages(modelId) && usePageImages,
          };
        else if (selection)
          classifier = { kind: "chat", choice: { providerId: selection, modelId: modelId.trim() } };
        void act(() =>
          core.saveLibrarySettings({ classifier, automatic: !!classifier && automatic }),
        );
      }}
    >
      <p className="mb-3 text-[13px] text-ink-meta">{t("library.modelHint")}</p>
      {loadError && (
        <p role="alert" className={errorTextClass}>
          {loadError}
        </p>
      )}
      <label className={fieldLabelClass}>
        {t("library.connection")}
        <select
          aria-label={t("library.connection")}
          className={inputClass}
          value={selection}
          onChange={(event) => {
            const value = event.target.value;
            setSelection(value);
            const current = useAppStore.getState().settings?.user.chatModel;
            setModelId(
              value === "ollama"
                ? "tev1:0.8b"
                : value === current?.providerId
                  ? current.modelId
                  : "",
            );
          }}
        >
          <option value="">{t("library.manualOnly")}</option>
          <option value="auto">{t("library.auto")}</option>
          {providers.map((provider) => (
            <option key={provider.id} value={provider.id}>
              {provider.service?.name ?? (provider.kind === "ollama" ? "Ollama" : provider.kind)}
              {provider.baseUrl ? ` · ${provider.baseUrl}` : ""}
            </option>
          ))}
          <option value="jev" disabled={!jevEnabled}>
            {t("library.jev")}
          </option>
          <option value="ollama">{t("library.ollama")}</option>
        </select>
      </label>
      {selection && selection !== "jev" && selection !== "auto" && (
        <label className={`${fieldLabelClass} mt-3`}>
          {t("library.model")}
          <input
            list="library-model-suggestions"
            className={inputClass}
            required
            maxLength={200}
            value={modelId}
            onChange={(event) => setModelId(event.target.value)}
          />
        </label>
      )}
      <datalist id="library-model-suggestions">
        {(selection === "ollama" ? ["tev1:0.8b", "clef-flash", "tev1:4b"] : []).map((model) => (
          <option key={model} value={model} />
        ))}
      </datalist>
      {(selection === "ollama" || selection === "auto") && (
        <>
          <label className={`${fieldLabelClass} mt-3`}>
            {t("library.server")}
            <input
              required
              type="url"
              className={inputClass}
              value={baseUrl}
              onChange={(event) => setBaseUrl(event.target.value)}
            />
          </label>
          <p className="mt-2 text-[13px] text-ink-meta">
            {t(selection === "auto" ? "library.autoHint" : "library.localHint")}
          </p>
          {selection === "auto" && (
            <p className="mt-2 text-[13px] text-ink-meta">{t("library.autoModels")}</p>
          )}
          {selection === "ollama" && supportsLibraryImages(modelId) && (
            <label className="mt-4 flex items-start gap-2 text-[13px] text-ink-secondary">
              <input
                type="checkbox"
                checked={usePageImages}
                onChange={(event) => setUsePageImages(event.target.checked)}
                className="mt-0.5 size-4 accent-ink"
              />
              {t("library.pageImages")}
            </label>
          )}
        </>
      )}
      {selection && (
        <label className="mt-4 flex items-start gap-2 text-[13px] text-ink-secondary">
          <input
            type="checkbox"
            checked={automatic}
            onChange={(event) => setAutomatic(event.target.checked)}
            className="mt-0.5 size-4 accent-ink"
          />
          {t("library.automatic")}
        </label>
      )}
      <div className="mt-4 flex flex-wrap gap-2">
        <button type="submit" disabled={busy} className={buttonStyle("primary")}>
          {t("library.saveModel")}
        </button>
        <button
          type="button"
          className={buttonStyle("ghost")}
          onClick={() => useAppStore.getState().openSettings("chat-model")}
        >
          {t("library.connectModel")}
        </button>
      </div>
    </form>
  );
}
