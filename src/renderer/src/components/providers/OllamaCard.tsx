import { useCallback, useEffect, useState } from "react";
import type { ChatProvider, OllamaPullProgress, OllamaStatus } from "../../../../core/api";
import { core } from "../../core";
import { errorMessage } from "../../errors";
import { useT } from "../../i18n";
import { buttonClass, primaryButtonClass } from "./shared";

/**
 * Local mode: if Ollama runs on its default port, one click pulls the
 * recommended model (with progress) and makes it the default chat model.
 */
export function OllamaCard({ onSaved }: { onSaved(provider: ChatProvider): void }) {
  const t = useT();
  const [status, setStatus] = useState<OllamaStatus | null>(null);
  const [pulling, setPulling] = useState(false);
  const [progress, setProgress] = useState<OllamaPullProgress | null>(null);
  const [error, setError] = useState<string | null>(null);

  const detect = useCallback(async () => {
    setStatus(null);
    try {
      setStatus(await core.detectOllama());
    } catch {
      setStatus({ running: false, baseUrl: "" });
    }
  }, []);

  useEffect(() => {
    void detect();
  }, [detect]);

  useEffect(() => core.on("ollama.pullProgress", setProgress), []);

  const use = async () => {
    setPulling(true);
    setProgress(null);
    setError(null);
    try {
      onSaved(await core.selectOllama());
    } catch (failure) {
      setError(errorMessage(failure));
    } finally {
      setPulling(false);
    }
  };

  const percent =
    progress?.total && progress.completed !== null
      ? Math.round((progress.completed / progress.total) * 100)
      : null;

  return (
    <section data-testid="ollama" className="rounded-[9px] border border-gray-200 p-3">
      <h3 className="text-sm font-medium">{t("providers.ollama.title")}</h3>
      {status === null ? (
        <p className="mt-1 text-sm text-gray-500">{t("providers.ollama.checking")}</p>
      ) : status.running ? (
        <>
          <p className="mt-1 text-sm text-gray-600">{t("providers.ollama.detected")}</p>
          <button
            type="button"
            data-testid="ollama-use"
            disabled={pulling}
            onClick={() => void use()}
            className={`mt-2 ${primaryButtonClass}`}
          >
            {t("providers.ollama.use", { model: status.recommendedModel })}
          </button>
        </>
      ) : (
        <>
          <p className="mt-1 text-sm text-gray-600">{t("providers.ollama.notDetected")}</p>
          <button type="button" onClick={() => void detect()} className={`mt-2 ${buttonClass}`}>
            {t("providers.ollama.checkAgain")}
          </button>
        </>
      )}
      {pulling && (
        <div className="mt-2 text-sm text-gray-600">
          <p>
            {t("providers.ollama.pulling", {
              model: progress?.model ?? (status?.running ? status.recommendedModel : ""),
            })}
          </p>
          <progress
            className="mt-1 w-full"
            max={100}
            {...(percent === null ? {} : { value: percent })}
          />
          {progress && (
            <p className="text-xs text-gray-500">
              {progress.status}
              {percent === null ? "" : ` · ${percent}%`}
            </p>
          )}
        </div>
      )}
      {error && (
        <p role="alert" className="mt-2 text-sm break-words text-red-700">
          {error}
        </p>
      )}
    </section>
  );
}
