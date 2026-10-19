import type { ReactNode } from "react";
import type { TranscriptionRoute, TranscriptionServiceStatus } from "../../../../core/api";
import { useGap, useLanguage, useT } from "../../i18n";
import { useMeetings } from "../../meetings";
import { serviceMeta, useTranscriptionServices } from "../../transcriptionServices";
import { CloudLineIcon, ComputerLineIcon } from "../lineIcons";
import { buttonStyle, dialogClass, dialogTitleClass } from "../ui";
import { useModal } from "../useModal";

const cardClass = "flex flex-col gap-1 rounded-[10px] px-3.5 py-3 ring-1 ring-rule ring-inset";
const recommendedCardClass =
  "flex flex-col gap-1 rounded-[10px] px-3.5 py-3 ring-[1.5px] ring-rule-strong ring-inset";
const nameClass = "inline-flex items-center gap-2 text-[15px] leading-5 font-semibold text-ink";
const lineClass = "text-[13px] leading-[19px] text-ink-secondary";

/**
 * The first "Start recording" once a cloud service has a key (#141): where should meeting audio
 * be transcribed? A card per route, each with its own button, so one click chooses and starts.
 * This computer comes first and is recommended; a cloud card says, in body text, that meeting
 * audio, other people's voices included, goes to that service, so choosing it is the consent
 * (ADR-0022). Nothing is chosen for the User: Esc or Cancel records nothing. Asked once; the
 * choice is Settings › Meetings' "Transcribe on".
 */
export function TranscriptionSheet() {
  const t = useT();
  const gap = useGap();
  const foot = t("meetings.sheet.foot");
  const tell = t("meetings.notice.tell");
  const waiting = useMeetings((state) => state.sheetFor);
  const services = useTranscriptionServices((state) => state.services).filter(
    (service) => service.hasKey,
  );
  const { pickRoute, cancelSheet } = useMeetings.getState();
  const dialog = useModal(waiting !== null);

  return (
    <dialog
      ref={dialog}
      data-testid="transcription-sheet"
      aria-labelledby="transcription-sheet-title"
      onClose={cancelSheet}
      className={`${dialogClass} w-[35rem]`}
    >
      <div className="flex flex-col gap-3 px-6 pt-[22px] pb-4">
        <div className="flex flex-col gap-1">
          <h2 id="transcription-sheet-title" className={dialogTitleClass}>
            {t("meetings.sheet.title")}
          </h2>
          <p className="text-[13px] leading-[18px] text-ink-meta">{t("meetings.sheet.sub")}</p>
        </div>
        <div className="flex flex-col gap-2">
          <div data-testid="transcription-sheet-local" className={recommendedCardClass}>
            <RouteHead
              route="local"
              name={t("meetings.sheet.local")}
              icon={<ComputerLineIcon className="size-4 text-ink-meta" />}
              action={t("meetings.sheet.useLocal")}
              recommended
              onPick={pickRoute}
            />
            <p className={lineClass}>{t("meetings.route.localLine")}</p>
          </div>
          {services.map((service) => (
            <CloudCard key={service.route} service={service} onPick={pickRoute} />
          ))}
        </div>
        <div className="flex flex-wrap items-center gap-x-3 gap-y-2 pt-1">
          <p className="min-w-0 flex-1 text-[13px] leading-[18px] text-ink-meta">
            {foot}
            {gap(foot, tell)}
            {tell}
          </p>
          <button
            type="button"
            data-testid="transcription-sheet-cancel"
            onClick={cancelSheet}
            className={`${buttonStyle("ghost")} gap-1.5`}
          >
            {t("meetings.notice.cancel")}
            <kbd className="rounded-sm border border-rule px-1 font-sans text-[11px] leading-4 text-ink-meta">
              Esc
            </kbd>
          </button>
        </div>
      </div>
    </dialog>
  );
}

function CloudCard({
  service,
  onPick,
}: {
  service: TranscriptionServiceStatus;
  onPick: (route: TranscriptionRoute) => void;
}) {
  const t = useT();
  const language = useLanguage();
  return (
    <div data-testid={`transcription-sheet-${service.route}`} className={cardClass}>
      <RouteHead
        route={service.route}
        name={service.name}
        icon={<CloudLineIcon className="size-4 text-ink-meta" />}
        action={t("meetings.sheet.use", { service: service.name })}
        onPick={onPick}
      />
      <p className="text-[13px] leading-[18px] text-ink-meta tabular-nums">
        {serviceMeta(service, t, language)}
      </p>
      {/* The consent line is the point of the card: body text, not fine print. */}
      <p className={`${lineClass} text-ink`}>
        {t("meetings.route.data", { service: service.name })}
      </p>
    </div>
  );
}

function RouteHead({
  route,
  name,
  icon,
  action,
  recommended = false,
  onPick,
}: {
  route: TranscriptionRoute;
  name: string;
  icon: ReactNode;
  action: string;
  recommended?: boolean;
  onPick: (route: TranscriptionRoute) => void;
}) {
  const t = useT();
  return (
    <div className="flex min-h-8 flex-wrap items-center gap-2">
      <span className={nameClass}>
        {icon}
        {name}
      </span>
      {recommended && (
        <span className="rounded-sm bg-chip px-1.5 text-[11px] leading-[18px] font-semibold text-ink-secondary">
          {t("meetings.sheet.recommended")}
        </span>
      )}
      <span className="grow" />
      <button
        type="button"
        data-testid={`transcription-sheet-use-${route}`}
        onClick={() => onPick(route)}
        className={buttonStyle(recommended ? "primary" : "secondary")}
      >
        {action}
      </button>
    </div>
  );
}
