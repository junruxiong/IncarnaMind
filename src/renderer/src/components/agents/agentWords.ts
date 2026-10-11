/**
 * Words for Settings → Agents (#351), shared by its parts: an agent's name, how
 * it is set up, what came of setting it up, and a call in its activity list,
 * all in plain words (no "MCP", "stdio" or "token" outside the details).
 */
import type {
  AgentActivity,
  AgentClient,
  AgentClientId,
  AgentSetupResult,
} from "../../../../core/api";
import type { Language } from "../../../../core/language";
import type { MessageKey, MessageParams } from "../../../../shared/i18n";

type Translate = (key: MessageKey, params?: MessageParams) => string;

/** An agent's name: its own, or "Another agent". */
export const clientName = (client: Pick<AgentClient, "id" | "name">, t: Translate) =>
  client.id === "other" ? t("agents.client.other") : client.name;

/** How connecting an agent goes, in a line under its name. */
export function setupWords(client: AgentClient, t: Translate): string {
  const agent = clientName(client, t);
  switch (client.setup) {
    case "bundle":
      return t("agents.setup.bundle");
    case "link":
      return t("agents.setup.link", { agent });
    case "file":
      return t("agents.setup.file", { agent });
    case "command":
      return t("agents.setup.command");
    case "manual":
      return t("agents.setup.manual", { agent });
  }
}

/** What came of setting an agent up, as one line. */
export function doneWords(result: AgentSetupResult, agent: string, t: Translate): string {
  switch (result.kind) {
    case "opened":
      return result.how === "bundle" ? t("agents.done.bundle") : t("agents.done.link", { agent });
    case "wrote":
      return t("agents.done.wrote", { agent });
    case "copy":
      if (result.path) return t("agents.done.unreadable", { agent, path: result.path });
      return result.how === "command"
        ? t("agents.done.command")
        : t("agents.done.settings", { agent });
  }
}

/** A call in the activity list: what it did, and how many results, or why it didn't. */
export function activityWords(activity: AgentActivity, t: Translate): string {
  const text = activity.summary;
  const what = t(`agents.activity.${activity.tool}`, { text });
  if (activity.outcome !== "done") {
    const reason = t(`agents.reason.${activity.problem ?? "failed"}`);
    return `${what} · ${t(
      activity.outcome === "refused" ? "agents.activity.refused" : "agents.activity.failed",
      { reason },
    )}`;
  }
  if (activity.count === null || activity.tool === "show_citation") return what;
  const count = t(
    activity.count === 1 ? "agents.activity.count.one" : "agents.activity.count.other",
    { count: activity.count },
  );
  return `${what} · ${count}`;
}

/** A time, as the activity list and "Last used" show it. */
export const timeWords = (at: string, language: Language) =>
  new Intl.DateTimeFormat(language, { dateStyle: "medium", timeStyle: "short" }).format(
    new Date(at),
  );

/** Where the step-by-step help for an agent is: the setup notes, in the interface language. */
export const helpUrl = (client: AgentClientId, language: Language) =>
  `https://github.com/junruxiong/IncarnaMind/blob/main/docs/${
    language === "zh-CN" ? "agents.zh-CN.md" : "agents.md"
  }#${client}`;
