import { askHelpInAssistant } from "../../meetingHelp";
import { isLive, useMeetings } from "../../meetings";
import { HelpButtons } from "./HelpButtons";

/** The help buttons at the top of the assistant while the Mind it reads records. */
export function HelpStrip({ mindId }: { mindId: string | null }) {
  const meeting = useMeetings((state) =>
    state.meetings.find((each) => each.mindId === mindId && isLive(each)),
  );
  if (!meeting) return null;
  return (
    <div data-testid="help-strip" className="help-strip">
      <HelpButtons
        meeting={meeting}
        layout="strip"
        onPress={(kind) => void askHelpInAssistant(kind)}
      />
    </div>
  );
}
