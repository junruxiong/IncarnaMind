import type {
  ApprovalPolicy,
  ApprovalPolicyValue,
  ConnectorDirectoryEntry,
  ConnectorTool,
} from "../../../../core/api";
import { core } from "../../core";
import { useLanguage, useT } from "../../i18n";
import { useAppStore } from "../../store";
import { ruledListClass, segmentClass, segmentedClass } from "../ui";

/** What the User chooses for a Tool: ask before each call, run without asking, or leave it out. */
const choices = ["ask", "always", "off"] as const;
type Choice = (typeof choices)[number];

/**
 * Whether Answers ask before calling a Tool: the User's policy, or else the
 * default for what its Connector claims (a Tool that says it only reads doesn't ask).
 */
export function asksFirst(tool: ConnectorTool, policy: ApprovalPolicyValue | undefined): boolean {
  return policy === "ask" || (policy === undefined && !tool.readOnly);
}

/**
 * What a connected Connector can do: each Tool, what it claims (only reads,
 * or may change things, as the Connector says; IncarnaMind can't check), and
 * whether Answers ask before calling it, call it without asking, or don't
 * use it at all, which the User chooses.
 */
export function ConnectorTools({
  connectorId,
  tools,
  policies,
}: {
  connectorId: string;
  tools: ConnectorTool[];
  policies: ApprovalPolicy[];
}) {
  const t = useT();
  const policyOf = (tool: ConnectorTool) =>
    policies.find(
      (policy) =>
        policy.subject.kind === "tool" &&
        policy.subject.connectorId === connectorId &&
        policy.subject.tool === tool.name,
    )?.policy;
  const choose = (tool: ConnectorTool, choice: Choice) => {
    // The default for what the Tool claims needs no policy.
    const policy: ApprovalPolicyValue | null =
      choice === "off" ? "off" : (choice === "ask") === !tool.readOnly ? null : choice;
    core
      .setApprovalPolicy({ subject: { kind: "tool", connectorId, tool: tool.name }, policy })
      .catch((failure: unknown) => useAppStore.getState().reportError(failure));
  };
  if (tools.length === 0) {
    return <p className="text-[13px] leading-5 text-ink-meta">{t("connectors.tools.none")}</p>;
  }
  return (
    <div className="flex flex-col gap-2">
      <ul data-testid="connector-tools" className={ruledListClass}>
        {tools.map((tool) => {
          const policy = policyOf(tool);
          const ask = asksFirst(tool, policy);
          const chosen: Choice = policy === "off" ? "off" : ask ? "ask" : "always";
          const label = t("approvals.tools.select", { tool: tool.name });
          return (
            <li
              key={tool.name}
              data-testid="connector-tool"
              data-tool={tool.name}
              data-read-only={tool.readOnly}
              data-asks={ask}
              data-off={chosen === "off"}
              className="flex items-center gap-4 py-2.5"
            >
              <span className="flex min-w-0 flex-1 flex-col" title={tool.description}>
                <span
                  data-testid="connector-tool-name"
                  className="truncate text-[13px] leading-5 font-semibold text-ink"
                >
                  {tool.title ?? tool.name}
                </span>
                <span
                  data-testid="connector-tool-claim"
                  className="text-[12px] leading-[18px] text-ink-meta"
                >
                  {tool.readOnly
                    ? t("connectors.directory.tool.reads")
                    : t("connectors.directory.tool.changes")}
                </span>
              </span>
              <span
                role="radiogroup"
                aria-label={label}
                data-testid="connector-tool-approval"
                className={segmentedClass}
              >
                {choices.map((each) => (
                  <label key={each} className={segmentClass}>
                    <input
                      type="radio"
                      name={`${connectorId}:${tool.name}`}
                      value={each}
                      checked={chosen === each}
                      onChange={() => choose(tool, each)}
                      className="sr-only"
                    />
                    {t(`approvals.tools.${each}`)}
                  </label>
                ))}
              </span>
            </li>
          );
        })}
      </ul>
      <p className="text-[12px] leading-[18px] text-ink-meta">{t("connectors.tools.note")}</p>
    </div>
  );
}

/**
 * A directory entry's Tools as IncarnaMind reviewed them, before the User
 * connects it: each one's name and what it does, with no choice yet, since
 * the choices belong to the Tools the service lists once connected.
 */
export function ReviewedTools({ tools }: { tools: NonNullable<ConnectorDirectoryEntry["tools"]> }) {
  const t = useT();
  const language = useLanguage();
  return (
    <div className="flex flex-col gap-2">
      <ul data-testid="connector-reviewed-tools" className={ruledListClass}>
        {tools.map((tool) => (
          <li
            key={tool.title.en}
            data-testid="connector-reviewed-tool"
            className="flex flex-col py-2.5"
          >
            <span className="text-[13px] leading-5 font-semibold text-ink">
              {tool.title[language]}
            </span>
            <span className="text-[12px] leading-[18px] text-ink-meta">
              {tool.changes
                ? t("connectors.directory.tool.changes")
                : t("connectors.directory.tool.reads")}
            </span>
          </li>
        ))}
      </ul>
      <p className="text-[12px] leading-[18px] text-ink-meta">
        {t("connectors.directory.what.listed")}
      </p>
    </div>
  );
}
