import { useEffect, useRef, useState } from "react";
import { TRANSCRIPTION_ROUTES, type TranscriptionRoute } from "../../../../core/api";
import type { MessageKey } from "../../../../shared/i18n";
import { useT } from "../../i18n";
import { useAppStore } from "../../store";

export const ROUTE_LABELS: Record<TranscriptionRoute, MessageKey> = {
  local: "meetings.route.local",
};

/**
 * Where meeting audio is transcribed, one setting wherever it is shown (Settings › Models,
 * Settings › Meetings, the first-run list). Unchosen it reads as this computer, and choosing that
 * stores it: a choice made on purpose is not the same as none (the first meeting asks only for none).
 */
export function TranscriptionRouteSelect({
  id,
  className,
  labelledBy,
  focusOnShow,
  onChosen,
  onBlur,
  onEscape,
}: {
  id: string;
  className: string;
  /** The element that names it, where its label isn't one that points at it. */
  labelledBy?: string;
  /** It takes the focus when it shows (Change was just pressed). */
  focusOnShow?: boolean;
  onChosen?(): void;
  onBlur?(): void;
  onEscape?(): void;
}) {
  const t = useT();
  const saved = useAppStore((state) => state.settings?.device.transcriptionRoute) ?? null;
  const updateSettings = useAppStore((state) => state.updateSettings);
  // Shown at once; the setting follows when the core has it.
  const [value, setValue] = useState<TranscriptionRoute | null>(saved);
  useEffect(() => setValue(saved), [saved]);
  const field = useRef<HTMLSelectElement>(null);
  useEffect(() => {
    if (focusOnShow) field.current?.focus();
  }, [focusOnShow]);
  return (
    <select
      ref={field}
      id={id}
      aria-describedby={`${id}-tip`}
      aria-labelledby={labelledBy}
      data-testid="meetings-route"
      value={value ?? ""}
      onBlur={onBlur}
      onKeyDown={(event) => {
        if (event.key === "Escape" && onEscape) {
          event.preventDefault();
          event.stopPropagation();
          onEscape();
        }
      }}
      onChange={(event) => {
        const chosen = TRANSCRIPTION_ROUTES.find((each) => each === event.target.value);
        if (!chosen) return;
        setValue(chosen);
        void updateSettings({ device: { transcriptionRoute: chosen } });
        onChosen?.();
      }}
      className={className}
    >
      {value === null && (
        <option value="" hidden>
          {t(ROUTE_LABELS.local)}
        </option>
      )}
      {TRANSCRIPTION_ROUTES.map((each) => (
        <option key={each} value={each}>
          {t(ROUTE_LABELS[each])}
        </option>
      ))}
    </select>
  );
}
