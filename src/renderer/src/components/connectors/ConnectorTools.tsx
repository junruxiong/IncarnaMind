import type { ApprovalPolicy, ApprovalPolicyValue, ConnectorTool } from "../../../../core/api";
import { core } from "../../core";
import { useT } from "../../i18n";
import { useAppStore } from "../../store";
import { ruledListClass, segmentClass, segmentedClass } from "../ui";

/**
 * Whether Answers ask before calling a Tool: the User's policy, or else the
 * default for what its Connector claims (a Tool that says it only reads doesn't ask).
 */
export function asksFirst(tool: ConnectorTool, policy: ApprovalPolicyValue | undefined): boolean {
  return policy === "ask" || (policy !== "always" && !tool.readOnly);
}

/**
 * What a connected Connector can do: each Tool, what it claims (only reads,
 * or may change things, as the Connector says; IncarnaMind can't check), and
 * whether Answers ask before calling it, which the User can change.
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
  const choose = (tool: ConnectorTool, ask: boolean) => {
    // The default for what the Tool claims needs no policy.
    const policy: ApprovalPolicyValue | null =
      ask === !tool.readOnly ? null : ask ? "ask" : "always";
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
          const ask = asksFirst(tool, policyOf(tool));
          const label = t("approvals.tools.select", { tool: tool.name });
          return (
            <li
              key={tool.name}
              data-testid="connector-tool"
              data-tool={tool.name}
              data-read-only={tool.readOnly}
              data-asks={ask}
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
                {(["ask", "always"] as const).map((each) => (
                  <label key={each} className={segmentClass}>
                    <input
                      type="radio"
                      name={`${connectorId}:${tool.name}`}
                      value={each}
                      checked={ask === (each === "ask")}
                      onChange={() => choose(tool, each === "ask")}
                      className="sr-only"
                    />
                    {t(each === "ask" ? "approvals.tools.ask" : "approvals.tools.always")}
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
