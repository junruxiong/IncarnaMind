import { useId, useState } from "react";
import type { TranscriptionServiceStatus } from "../../../../core/api";
import { useLanguage, useT } from "../../i18n";
import { useAppStore } from "../../store";
import {
  serviceMeta,
  useTranscriptionServices,
  useTranscriptionServicesList,
} from "../../transcriptionServices";
import { ProblemLine } from "../ProblemLine";
import {
  buttonStyle,
  inputClass,
  rowButtonsClass,
  rowStatusClass,
  rowTextClass,
  rowTitleClass,
  ruledRowClass,
} from "../ui";

/**
 * Under "Transcribe on" in Settings › Meetings (#141): where the chosen route sends meeting
 * audio, in the line that makes choosing it the consent (ADR-0022).
 */
export function TranscriptionRouteLine({ id }: { id?: string }) {
  const t = useT();
  const language = useLanguage();
  const route = useAppStore((state) => state.settings?.device.transcriptionRoute) ?? "local";
  const services = useTranscriptionServicesList();
  const service = services.find((each) => each.route === route);
  if (!service) {
    return (
      <p id={id} data-testid="meetings-route-line" className={rowTextClass}>
        {t("meetings.route.localLine")}
      </p>
    );
  }
  return (
    <div id={id} data-testid="meetings-route-line" className="flex flex-col gap-0.5">
      <p className="text-[13px] leading-5 text-ink-meta tabular-nums">
        {serviceMeta(service, t, language)}
      </p>
      <p className={rowTextClass}>{t("meetings.route.data", { service: service.name })}</p>
    </div>
  );
}

/**
 * Where a job list offers the Meetings choice (first run, Settings › Models), the line saying
 * that meeting audio goes to the chosen cloud service; nothing while it is this computer.
 */
export function TranscriptionRouteConsent() {
  const t = useT();
  const route = useAppStore((state) => state.settings?.device.transcriptionRoute) ?? "local";
  const service = useTranscriptionServicesList().find((each) => each.route === route);
  if (!service) return null;
  return (
    <p data-testid="meetings-route-consent" className="text-[12px] leading-5 text-ink-meta">
      {t("meetings.route.data", { service: service.name })}
    </p>
  );
}

/** A transcription service's own key (Soniox's): pasted, checked with the service, kept in the keychain. */
export function TranscriptionKeyRows() {
  const services = useTranscriptionServicesList();
  return (
    <>
      {services
        .filter((service) => service.key === "own")
        .map((service) => (
          <TranscriptionKeyRow key={service.route} service={service} />
        ))}
    </>
  );
}

function TranscriptionKeyRow({ service }: { service: TranscriptionServiceStatus }) {
  const t = useT();
  const language = useLanguage();
  const inputId = useId();
  const [open, setOpen] = useState(false);
  const [key, setKey] = useState("");
  const [saving, setSaving] = useState(false);
  const [problem, setProblem] = useState<string | null>(null);
  const name = { service: service.name };
  const save = async () => {
    setSaving(true);
    const failed = await useTranscriptionServices.getState().setKey(service.route, key);
    setSaving(false);
    setProblem(failed);
    if (!failed) {
      setKey("");
      setOpen(false);
    }
  };

  return (
    <div data-testid={`transcription-key-${service.route}`} className={ruledRowClass}>
      <div className="flex min-w-0 flex-1 flex-col gap-0.5">
        <label htmlFor={inputId} className={rowTitleClass}>
          {t("meetings.key.title", name)}
        </label>
        <p className={rowTextClass}>
          {serviceMeta(service, t, language)}
          {service.keyUrl && (
            <>
              {" · "}
              <a
                href={service.keyUrl}
                target="_blank"
                rel="noreferrer"
                className="underline decoration-rule-strong underline-offset-2 hover:text-ink"
              >
                {t("meetings.key.get")}
              </a>
            </>
          )}
        </p>
        {open && (
          <form
            className="mt-1.5 flex flex-wrap items-center gap-2"
            onSubmit={(event) => {
              event.preventDefault();
              if (key.trim() !== "" && !saving) void save();
            }}
          >
            <input
              id={inputId}
              type="password"
              autoComplete="off"
              spellCheck={false}
              // biome-ignore lint/a11y/noAutofocus: it opens only when the User asks to add a key.
              autoFocus
              data-testid="transcription-key-input"
              placeholder={t("meetings.key.placeholder")}
              value={key}
              onChange={(event) => setKey(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Escape") {
                  event.stopPropagation();
                  setOpen(false);
                  setKey("");
                }
              }}
              className={`${inputClass} mt-0 min-w-48 flex-1`}
            />
            <button
              type="submit"
              data-testid="transcription-key-save"
              disabled={key.trim() === "" || saving}
              className={buttonStyle("primary")}
            >
              {t(saving ? "meetings.key.checking" : "meetings.key.save")}
            </button>
          </form>
        )}
        {problem && (
          <ProblemLine className="mt-1" testId="transcription-key-problem">
            {problem}
          </ProblemLine>
        )}
      </div>
      {!open && (
        <div className={rowButtonsClass}>
          {service.hasKey && (
            <span data-testid="transcription-key-saved" className={rowStatusClass}>
              {t("meetings.key.saved")}
            </span>
          )}
          <button
            type="button"
            data-testid="transcription-key-add"
            onClick={() => {
              setProblem(null);
              setOpen(true);
            }}
            className={buttonStyle("secondary")}
          >
            {t(service.hasKey ? "meetings.key.replace" : "meetings.key.add")}
          </button>
          {service.hasKey && (
            <button
              type="button"
              data-testid="transcription-key-remove"
              onClick={() =>
                void useTranscriptionServices.getState().removeKey(service.route).then(setProblem)
              }
              className={buttonStyle("ghost")}
            >
              {t("meetings.key.remove")}
            </button>
          )}
        </div>
      )}
    </div>
  );
}
