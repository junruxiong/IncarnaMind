import { useCallback, useEffect, useState } from "react";
import type { ChatProvider, OllamaPullProgress, OllamaStatus } from "../../../../core/api";
import { core } from "../../core";
import { errorMessage } from "../../errors";
import { useT } from "../../i18n";
import { errorTextClass } from "../ui";

/**
 * Local mode: whether Ollama runs on its default port, and, if it does, one
 * call that pulls the recommended model (with progress) and makes it the
 * default chat model.
 */
export function useOllama(onSaved: (provider: ChatProvider) => void) {
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

  return { status, pulling, progress, error, detect, use };
}

export type Ollama = ReturnType<typeof useOllama>;

/** What Ollama's choice says about this computer: looking, found, or not found. */
export function ollamaDescription(ollama: Ollama, t: ReturnType<typeof useT>): string {
  if (ollama.status === null) return t("providers.ollama.checking");
  return ollama.status.running ? t("providers.ollama.detected") : t("providers.ollama.notDetected");
}

/** The model being pulled, with its progress, and what went wrong. */
export function OllamaProgress({ ollama }: { ollama: Ollama }) {
  const t = useT();
  const { status, pulling, progress, error } = ollama;
  const percent =
    progress?.total && progress.completed !== null
      ? Math.round((progress.completed / progress.total) * 100)
      : null;
  if (!pulling && !error) return null;
  return (
    <div data-testid="ollama" className="flex flex-col gap-1.5">
      {pulling && (
        <div className="text-[13px] leading-5 text-ink-secondary">
          <p>
            {t("providers.ollama.pulling", {
              model: progress?.model ?? (status?.running ? status.recommendedModel : ""),
            })}
          </p>
          <div className="mt-1.5 h-1 overflow-hidden rounded-full bg-rule">
            <div
              className={`h-full bg-ink ${percent === null ? "w-1/3 motion-safe:animate-pulse" : ""}`}
              style={percent === null ? undefined : { width: `${percent}%` }}
            />
          </div>
          {progress && (
            <p className="mt-1 text-[12px] text-ink-meta">
              {progress.status}
              {percent === null ? "" : ` · ${percent}%`}
            </p>
          )}
        </div>
      )}
      {error && (
        <p role="alert" className={errorTextClass}>
          {error}
        </p>
      )}
    </div>
  );
}
