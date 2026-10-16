import type { Meeting, MeetingHelpKind } from "../../../../core/api";
import { helpKindsFor, helpLabel, isGeneralHelpKind } from "../../../../shared/meetingHelp";
import { useT } from "../../i18n";
import { helpKeysLabel, useMeetingHelp } from "../../meetingHelp";
import { useAppStore } from "../../store";

/**
 * The help buttons during a call: the general ones with their shortcuts, then the ones
 * the meeting's template adds. "grid" in the floating panel, two to a row with the keys on
 * them; "strip" at the top of the assistant, the keys in each tooltip. The one pressed last
 * is marked. `keys`: the shortcuts held now, which the panel shows; else the ones set.
 */
export function HelpButtons({
  meeting,
  layout,
  keys,
  onPress,
}: {
  meeting: Meeting;
  layout: "grid" | "strip";
  keys?: Partial<Record<MeetingHelpKind, string>>;
  onPress(kind: MeetingHelpKind): void;
}) {
  const t = useT();
  const settings = useAppStore((state) => state.settings);
  const pending = useMeetingHelp((state) => state.pending[meeting.id]);
  const pressed = pending ?? meeting.help?.kind;
  return (
    <fieldset
      aria-label={t("meetings.help.label")}
      data-testid="help-buttons"
      className={`help-buttons help-buttons--${layout}`}
    >
      {helpKindsFor(meeting.helpTemplate).map((kind) => {
        const name = t(helpLabel(kind));
        const shortcut = keys ? (keys[kind] ?? null) : helpKeysLabel(kind, settings);
        return (
          <button
            key={kind}
            type="button"
            data-testid="help-button"
            data-kind={kind}
            data-general={isGeneralHelpKind(kind)}
            aria-pressed={pressed === kind}
            title={shortcut ? t("meetings.help.button", { name, shortcut }) : name}
            onClick={() => onPress(kind)}
            className="help-button"
          >
            <span className="help-button-name">{name}</span>
            {layout === "grid" && shortcut && (
              <kbd data-testid="help-button-keys" className="help-button-keys">
                {shortcut}
              </kbd>
            )}
          </button>
        );
      })}
    </fieldset>
  );
}
