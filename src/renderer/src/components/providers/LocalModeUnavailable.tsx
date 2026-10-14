import { useT } from "../../i18n";
import { turnOffLocalMode } from "../../localMode";
import { useAppStore } from "../../store";
import { ProblemLine } from "../ProblemLine";

/** A cloud choice while "Keep everything on this computer" is on: why it can't be used, and the way out. */
export function LocalModeUnavailable() {
  const t = useT();
  return (
    <ProblemLine
      role="status"
      testId="local-mode-unavailable"
      action={{
        label: t("localMode.turnOff"),
        testId: "local-mode-turn-off",
        onClick: () =>
          turnOffLocalMode().catch((error: unknown) => useAppStore.getState().reportError(error)),
      }}
    >
      {t("composer.model.localModeOff")}
    </ProblemLine>
  );
}
