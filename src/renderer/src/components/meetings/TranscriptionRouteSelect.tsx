import { TRANSCRIPTION_ROUTES } from "../../../../core/api";
import type { MessageKey } from "../../../../shared/i18n";
import { useT } from "../../i18n";
import { useAppStore } from "../../store";
import { useTranscriptionServicesList } from "../../transcriptionServices";

export const ROUTE_LABELS: Record<(typeof TRANSCRIPTION_ROUTES)[number], MessageKey> = {
  local: "meetings.route.local",
  openai: "meetings.route.openai",
  soniox: "meetings.route.soniox",
};

/**
 * Where meeting audio is transcribed, one setting wherever it is shown (Settings › Models,
 * Settings › Meetings, the first-run list). Unchosen, it is this computer. A cloud service is
 * offered once there is a key for it (#141); choosing it is the consent to send it meeting audio.
 */
export function TranscriptionRouteSelect({
  id,
  className,
  autoFocus,
  onChosen,
  onBlur,
  onEscape,
}: {
  id: string;
  className: string;
  autoFocus?: boolean;
  onChosen?(): void;
  onBlur?(): void;
  onEscape?(): void;
}) {
  const t = useT();
  const route = useAppStore((state) => state.settings?.device.transcriptionRoute) ?? "local";
  const updateSettings = useAppStore((state) => state.updateSettings);
  const services = useTranscriptionServicesList();
  const offered = TRANSCRIPTION_ROUTES.filter(
    (each) =>
      each === "local" ||
      each === route ||
      services.some((service) => service.route === each && service.hasKey),
  );
  return (
    <select
      id={id}
      aria-describedby={`${id}-tip`}
      data-testid="meetings-route"
      value={route}
      // biome-ignore lint/a11y/noAutofocus: opened by Change, which the User just pressed.
      autoFocus={autoFocus}
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
        void updateSettings({ device: { transcriptionRoute: chosen } });
        onChosen?.();
      }}
      className={className}
    >
      {offered.map((each) => (
        <option key={each} value={each}>
          {t(ROUTE_LABELS[each])}
        </option>
      ))}
    </select>
  );
}
