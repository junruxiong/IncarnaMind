import { useT } from "../../i18n";
import { useAppStore } from "../../store";
import {
  buttonStyle,
  rowButtonsClass,
  rowTextClass,
  rowTitleClass,
  ruledListClass,
  ruledRowClass,
  sectionTitleClass,
} from "../ui";
import { useContinueWithChatGpt } from "./ChatGptSignIn";

const GOOGLE_CREDITS = "https://developers.google.com/program";

/**
 * Settings → Models → "Already pay for a plan?": the routes for a plan the User already has.
 * ChatGPT signs in; Claude and Google AI plans can't be used by other apps, so the plan pays
 * where the User uses IncarnaMind from (the Claude app, Antigravity: Settings → Agents).
 */
export function PlansSettings() {
  const t = useT();
  const showPage = useAppStore((state) => state.showSettingsPage);
  const sign = useContinueWithChatGpt({ keepDefault: true });
  const row = (title: string, line: string, action: React.ReactNode, extra?: React.ReactNode) => (
    <li className={ruledRowClass}>
      <div className="flex min-w-0 flex-col gap-0.5">
        <span className={rowTitleClass}>{title}</span>
        <span className={rowTextClass}>{line}</span>
        {extra}
      </div>
      <div className={rowButtonsClass}>{action}</div>
    </li>
  );
  return (
    <section data-testid="plans-settings">
      <h4 className={`mb-2 ${sectionTitleClass}`}>{t("plans.title")}</h4>
      <ul className={ruledListClass}>
        {row(
          t("plans.chatgpt.title"),
          t("plans.chatgpt.line"),
          <button
            type="button"
            data-testid="plans-chatgpt"
            onClick={() => void sign.run()}
            className={buttonStyle("secondary", "sm")}
          >
            {t("chatgpt.continue")}
          </button>,
        )}
        {row(
          t("plans.claude.title"),
          t("plans.claude.line"),
          <button
            type="button"
            data-testid="plans-claude"
            onClick={() => showPage("agents")}
            className={buttonStyle("secondary", "sm")}
          >
            {t("setup.claude.connect")}
          </button>,
        )}
        {row(
          t("plans.google.title"),
          t("plans.google.line"),
          <button
            type="button"
            data-testid="plans-google"
            onClick={() => showPage("agents")}
            className={buttonStyle("secondary", "sm")}
          >
            {t("plans.google.connect")}
          </button>,
          <a
            href={GOOGLE_CREDITS}
            target="_blank"
            rel="noreferrer"
            className="text-[13px] font-semibold text-accent underline-offset-2 hover:underline"
          >
            {t("plans.google.credits")} ↗
          </a>,
        )}
      </ul>
    </section>
  );
}
