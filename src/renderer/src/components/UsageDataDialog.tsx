import { useId, useState } from "react";
import type { MessageKey } from "../../../shared/i18n";
import { core } from "../core";
import { errorMessage } from "../errors";
import { useT } from "../i18n";
import { usageDataQuestionWaiting, useAppStore } from "../store";
import {
  buttonClass,
  dialogActionsClass,
  dialogBodyClass,
  dialogClass,
  dialogTextClass,
  dialogTitleClass,
  hintClass,
  primaryButtonClass,
  rowTitleClass,
  ruledListClass,
  ruledRowClass,
} from "./ui";
import { useModal } from "./useModal";

/** Every event and field usage data can hold, for people to read. */
export const USAGE_DATA_DETAILS_URL =
  "https://github.com/junruxiong/IncarnaMind/blob/main/docs/privacy.md#usage-data";

/** What usage data looks like, in words. */
const EXAMPLES: readonly MessageKey[] = [
  "usageData.example.question",
  "usageData.example.answer",
  "usageData.example.documents",
];

/**
 * The first run's question about usage data, shown once, only by a copy that
 * can send it. A release build asks, with an example of what is sent, and
 * sends nothing unless the User agrees ("Don't send" and Esc decline). A test
 * build (the alpha) says that it sends usage data, as testers agreed, with
 * the switch to turn it off. Either way the choice is kept in Settings → Privacy.
 */
export function UsageDataDialog() {
  const t = useT();
  const id = useId();
  const privacy = useAppStore((state) => state.privacy);
  const waiting = usageDataQuestionWaiting(privacy);
  const tester = privacy?.usageData.testerBuild === true;
  /** Closes as soon as the User answers, before the core confirms it. */
  const [answered, setAnswered] = useState(false);
  /** A test build's switch: on, as usage data is, until the User turns it off. */
  const [send, setSend] = useState(true);
  const open = waiting && !answered;
  // The dialog itself takes the focus, so neither answer looks chosen.
  const dialog = useModal(open, { focusDialog: true });

  const answer = async (agree: boolean) => {
    setAnswered(true);
    try {
      await core.updatePrivacySettings({ usageData: agree });
    } catch (failure) {
      // Not answered after all: ask again.
      setAnswered(false);
      useAppStore.setState({ actionError: errorMessage(failure) });
    }
  };

  return (
    <dialog
      ref={dialog}
      tabIndex={-1}
      data-testid="usage-data-dialog"
      data-tester={tester}
      aria-labelledby={`${id}-title`}
      aria-describedby={`${id}-body`}
      // Esc declines in a release build, and keeps the switch's state in a test build.
      onCancel={(event) => {
        event.preventDefault();
        void answer(tester ? send : false);
      }}
      className={`${dialogClass} w-[30rem] outline-none`}
    >
      {open && (
        <div className={dialogBodyClass}>
          <h2 id={`${id}-title`} className={dialogTitleClass}>
            {t(tester ? "usageData.dialog.tester.title" : "usageData.dialog.title")}
          </h2>
          <div className="flex flex-col gap-2">
            <p id={`${id}-body`} className={dialogTextClass}>
              {t(tester ? "usageData.dialog.tester.body" : "usageData.dialog.body")}
            </p>
            <p className={dialogTextClass}>{t("usageData.dialog.example")}</p>
            <ul data-testid="usage-data-examples" className={ruledListClass}>
              {EXAMPLES.map((key) => (
                <li key={key} className="py-2 text-ui text-ink">
                  {t(key)}
                </li>
              ))}
            </ul>
          </div>
          <p className={dialogTextClass}>{t("usageData.dialog.never")}</p>
          {tester && (
            <div className={ruledListClass}>
              <div className={ruledRowClass}>
                <label htmlFor={`${id}-switch`} className={rowTitleClass}>
                  {t("privacy.traffic.usage-data.toggle")}
                </label>
                <input
                  id={`${id}-switch`}
                  type="checkbox"
                  role="switch"
                  data-testid="usage-data-switch"
                  className="switch"
                  checked={send}
                  aria-checked={send}
                  onChange={(event) => setSend(event.target.checked)}
                />
              </div>
            </div>
          )}
          <p className={hintClass}>
            {t("usageData.dialog.change")}{" "}
            <a
              href={USAGE_DATA_DETAILS_URL}
              target="_blank"
              rel="noreferrer"
              className="text-accent underline-offset-2 hover:text-accent-strong hover:underline"
            >
              {t("usageData.dialog.details")}
            </a>
          </p>
          <div className={dialogActionsClass}>
            {tester ? (
              <button
                type="button"
                data-testid="usage-data-done"
                onClick={() => void answer(send)}
                className={primaryButtonClass}
              >
                {t("usageData.dialog.done")}
              </button>
            ) : (
              <>
                <button
                  type="button"
                  data-testid="usage-data-decline"
                  onClick={() => void answer(false)}
                  className={buttonClass}
                >
                  {t("usageData.dialog.decline")}
                </button>
                <button
                  type="button"
                  data-testid="usage-data-accept"
                  onClick={() => void answer(true)}
                  className={primaryButtonClass}
                >
                  {t("usageData.dialog.accept")}
                </button>
              </>
            )}
          </div>
        </div>
      )}
    </dialog>
  );
}
