import { useCallback, useEffect, useId, useState } from "react";
import type { SuggestionStatus } from "../../../core/api";
import { core } from "../core";
import { useT } from "../i18n";
import { useAppStore } from "../store";
import {
  hintClass,
  rowTextClass,
  rowTitleClass,
  ruledListClass,
  ruledRowClass,
  sectionTitleClass,
} from "./ui";

/**
 * Settings → General → Writing: the switch for suggestions as the User
 * writes (#282), off by default, what it does, today's counts, and why
 * suggestions that are on can't be made now. Turning it on asks for consent
 * first when the chat model sends to a service (the core asks, naming it).
 */
export function SuggestionSettings() {
  const t = useT();
  const on = useAppStore((state) => state.settings?.user.suggestWhileWriting);
  const updateSettings = useAppStore((state) => state.updateSettings);
  const reportError = useAppStore((state) => state.reportError);
  const [status, setStatus] = useState<SuggestionStatus | null>(null);
  const id = useId();

  const refresh = useCallback(
    () => core.getSuggestionStatus().then(setStatus, reportError),
    [reportError],
  );
  // Again whenever the switch changes: the reason it is paused may have gone.
  useEffect(() => {
    if (on !== undefined) void refresh();
  }, [on, refresh]);
  if (on === undefined) return null;

  const turn = async (next: boolean) => {
    await updateSettings({ user: { suggestWhileWriting: next } });
    await refresh();
  };
  const paused = on ? status?.paused : null;

  return (
    <section data-testid="suggestion-settings">
      <h4 className={`mb-2 ${sectionTitleClass}`}>{t("suggestions.settings.title")}</h4>
      <div className={ruledListClass}>
        <div className={ruledRowClass}>
          <div className="flex min-w-0 flex-col gap-0.5">
            <label htmlFor={id} className={rowTitleClass}>
              {t("suggestions.settings.switch")}
            </label>
            <p id={`${id}-description`} className={rowTextClass}>
              {t("suggestions.settings.body")}
            </p>
            {status && (
              <p data-testid="suggestion-counts" className={hintClass}>
                {t("suggestions.settings.today", status.today)}
              </p>
            )}
            {paused && (
              <p data-testid="suggestion-paused" role="status" className={hintClass}>
                {paused.reason === "consent"
                  ? t("suggestions.settings.paused.consent", { service: paused.service.name })
                  : t(`suggestions.settings.paused.${paused.reason}`)}
                {paused.reason === "consent" && (
                  <>
                    <span aria-hidden="true"> · </span>
                    <button
                      type="button"
                      data-testid="suggestion-allow"
                      onClick={() => void turn(true)}
                      className="rounded-sm font-semibold text-accent underline-offset-2 hover:underline"
                    >
                      {t("suggestions.settings.allow")}
                    </button>
                  </>
                )}
              </p>
            )}
          </div>
          <input
            id={id}
            type="checkbox"
            role="switch"
            aria-checked={on}
            aria-describedby={`${id}-description`}
            data-testid="suggestion-switch"
            checked={on}
            onChange={(event) => void turn(event.target.checked)}
            className="switch"
          />
        </div>
      </div>
    </section>
  );
}
