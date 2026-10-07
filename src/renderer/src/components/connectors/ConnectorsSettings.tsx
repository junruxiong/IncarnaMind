import { useEffect, useRef, useState } from "react";
import type {
  ApprovalPolicy,
  ApprovalPolicyValue,
  Connector,
  ConnectorError,
  ConnectorState,
  ConnectorTool,
} from "../../../../core/api";
import { core } from "../../core";
import { errorMessage } from "../../errors";
import { useT } from "../../i18n";
import { useAppStore } from "../../store";
import { TrashIcon } from "../icons";
import { buttonClass } from "../providers/shared";
import { ConnectorForm } from "./ConnectorForm";
import { ConnectorImport } from "./ConnectorImport";
import { commandLine } from "./commandLine";

const stateClass: Record<ConnectorState, string> = {
  off: "bg-gray-100 text-gray-500",
  connecting: "bg-blue-50 text-blue-700",
  ready: "bg-green-50 text-green-700",
  error: "bg-red-50 text-red-700",
};

/** A Connector's Tools and what each claims, to notice when they change. */
const toolsSignature = (tools: readonly ConnectorTool[]) =>
  tools.map((tool) => `${tool.name}:${tool.readOnly}`).join("\n");

/**
 * Settings → Connectors: each Connector with its state, a switch to turn it
 * on or off, and its Tools, each with what it claims (read-only or not) and
 * whether Answers ask before calling it; adding one by its command, or
 * importing Claude Desktop's or Cursor's configuration. The Tools of a
 * Connector added, or whose Tools changed, while Settings is open are shown
 * unfolded, so the User sees which claim to be read-only.
 */
export function ConnectorsSettings() {
  const t = useT();
  const [connectors, setConnectors] = useState<Connector[] | null>(null);
  const [policies, setPolicies] = useState<ApprovalPolicy[]>([]);
  const [mode, setMode] = useState<"list" | "add" | "import">("list");
  /** Connectors whose Tools to show unfolded. */
  const [unfolded, setUnfolded] = useState<ReadonlySet<string>>(new Set());
  /** The Connectors there when Settings opened, and each one's Tools as last seen. */
  const seen = useRef<{ initial: ReadonlySet<string>; tools: Map<string, string> } | null>(null);

  useEffect(() => {
    const update = (list: Connector[]) => {
      setConnectors(list);
      const first = seen.current === null;
      const known = seen.current ?? {
        initial: new Set(list.map((each) => each.id)),
        tools: new Map<string, string>(),
      };
      seen.current = known;
      const open: string[] = [];
      for (const connector of list) {
        if (!connector.tools) continue;
        const signature = toolsSignature(connector.tools);
        const previous = known.tools.get(connector.id);
        const added = !first && !known.initial.has(connector.id) && previous === undefined;
        const changed = previous !== undefined && previous !== signature;
        if (added || changed) open.push(connector.id);
        known.tools.set(connector.id, signature);
      }
      if (open.length > 0) setUnfolded((current) => new Set([...current, ...open]));
    };
    core.listConnectors().then(update, () => undefined);
    core.listApprovalPolicies().then(setPolicies, () => undefined);
    const stops = [
      core.on("connectors.changed", update),
      core.on("approvals.changed", setPolicies),
    ];
    return () => {
      for (const stop of stops) stop();
    };
  }, []);

  if (!connectors) return null;
  return (
    <section data-testid="connectors-settings">
      <h3 className="mb-1 text-sm font-medium">{t("connectors.settings.title")}</h3>
      <p className="text-sm text-gray-600">{t("connectors.settings.body")}</p>

      {connectors.length === 0 && mode === "list" && (
        <p className="mt-2 text-sm text-gray-500">{t("connectors.settings.empty")}</p>
      )}
      {connectors.length > 0 && (
        <ul className="mt-2 flex flex-col gap-2">
          {connectors.map((connector) => (
            <ConnectorRow
              key={connector.id}
              connector={connector}
              policies={policies}
              unfolded={unfolded.has(connector.id)}
            />
          ))}
        </ul>
      )}

      {mode === "list" && (
        <div className="mt-2 flex flex-wrap gap-2">
          <button
            type="button"
            data-testid="connector-add"
            onClick={() => setMode("add")}
            className={buttonClass}
          >
            {t("connectors.settings.add")}
          </button>
          <button
            type="button"
            data-testid="connector-import"
            onClick={() => setMode("import")}
            className={buttonClass}
          >
            {t("connectors.settings.import")}
          </button>
        </div>
      )}
      {mode === "add" && <ConnectorForm onDone={() => setMode("list")} />}
      {mode === "import" && <ConnectorImport onDone={() => setMode("list")} />}
    </section>
  );
}

/** Runs an action on a Connector; a failure shows in the app's error banner. */
async function act(action: () => Promise<unknown>) {
  try {
    await action();
  } catch (failure) {
    useAppStore.setState({ actionError: errorMessage(failure) });
  }
}

function ConnectorRow({
  connector,
  policies,
  unfolded,
}: {
  connector: Connector;
  policies: ApprovalPolicy[];
  unfolded: boolean;
}) {
  const t = useT();
  const { id, name } = connector;
  return (
    <li
      data-testid="connector"
      data-connector-id={id}
      data-state={connector.state}
      className="rounded-[9px] border border-gray-200 px-3 py-2"
    >
      <div className="flex items-center gap-2">
        <span className="min-w-0 flex-1 truncate text-sm font-medium">{name}</span>
        <span
          data-testid="connector-state"
          className={`shrink-0 rounded-full px-2 py-0.5 text-xs ${stateClass[connector.state]}`}
        >
          {t(`connectors.state.${connector.state}`)}
        </span>
        <input
          type="checkbox"
          role="switch"
          aria-checked={connector.enabled}
          data-testid="connector-toggle"
          aria-label={t("connectors.toggle", { name })}
          checked={connector.enabled}
          onChange={(event) => void act(() => core.setConnectorEnabled(id, event.target.checked))}
          className="size-4 shrink-0 accent-gray-800"
        />
        <button
          type="button"
          aria-label={t("connectors.remove", { name })}
          title={t("connectors.remove", { name })}
          onClick={() => void act(() => core.deleteConnector(id))}
          className="shrink-0 rounded-[6px] p-1 text-gray-500 hover:bg-gray-100 hover:text-gray-800"
        >
          <TrashIcon className="size-4" />
        </button>
      </div>
      <p
        className="mt-0.5 truncate font-mono text-xs text-gray-500"
        title={commandLine(connector.command, connector.args)}
      >
        {commandLine(connector.command, connector.args)}
      </p>
      {connector.state === "error" && connector.error && (
        <ErrorNotice
          error={connector.error}
          onRetry={() => void act(() => core.restartConnector(id))}
        />
      )}
      {connector.state === "ready" && connector.tools && (
        <Tools
          // A new set of Tools unfolds afresh.
          key={toolsSignature(connector.tools)}
          connectorId={id}
          tools={connector.tools}
          policies={policies}
          unfolded={unfolded}
        />
      )}
    </li>
  );
}

/** What went wrong, in plain words, and what the User can do. */
function ErrorNotice({ error, onRetry }: { error: ConnectorError; onRetry(): void }) {
  const t = useT();
  const message =
    error.kind === "missing-command"
      ? error.install
        ? t("connectors.error.missing-command.install", {
            command: error.command ?? "",
            install: error.install,
          })
        : t("connectors.error.missing-command", { command: error.command ?? "" })
      : t(`connectors.error.${error.kind}`);
  return (
    <div
      role="alert"
      data-testid="connector-error"
      data-error-kind={error.kind}
      className="mt-1.5 rounded-[6px] bg-red-50 px-2 py-1.5 text-sm text-red-900"
    >
      <p>{message}</p>
      <div className="mt-1 flex flex-wrap items-center gap-2 text-xs">
        {error.retrying ? (
          <span className="text-red-800/80">{t("connectors.restarting")}</span>
        ) : (
          <button
            type="button"
            onClick={onRetry}
            className="rounded-[6px] border border-current/20 px-2 py-0.5 hover:bg-white/60"
          >
            {t("connectors.retry")}
          </button>
        )}
        {error.kind !== "missing-command" && error.message && (
          <details className="min-w-0 text-red-800/80">
            <summary className="cursor-pointer">{t("connectors.details")}</summary>
            <pre className="mt-1 max-h-32 overflow-auto font-mono whitespace-pre-wrap break-words">
              {error.message}
            </pre>
          </details>
        )}
      </div>
    </div>
  );
}

/**
 * Whether Answers ask before calling a Tool: the User's policy, or else the
 * default for what its Connector claims (a Tool that says it only reads doesn't ask).
 */
function asksFirst(tool: ConnectorTool, policy: ApprovalPolicyValue | undefined): boolean {
  return policy === "ask" || (policy !== "always" && !tool.readOnly);
}

/**
 * The Connector's Tools, folded: what each claims to do (read-only or not,
 * as the Connector says), and whether Answers ask before calling it, which
 * the User can change for any Tool.
 */
function Tools({
  connectorId,
  tools,
  policies,
  unfolded,
}: {
  connectorId: string;
  tools: ConnectorTool[];
  policies: ApprovalPolicy[];
  unfolded: boolean;
}) {
  const t = useT();
  const count =
    tools.length === 0
      ? t("connectors.tools.none")
      : tools.length === 1
        ? t("connectors.tools.count.one")
        : t("connectors.tools.count", { count: tools.length });
  if (tools.length === 0) return <p className="mt-1 text-xs text-gray-500">{count}</p>;
  const readOnly = tools.filter((tool) => tool.readOnly).length;
  const claims =
    readOnly === 0
      ? t("approvals.tools.readOnly.none")
      : readOnly === 1
        ? t("approvals.tools.readOnly.one")
        : t("approvals.tools.readOnly.count", { count: readOnly });
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
    void act(() =>
      core.setApprovalPolicy({
        subject: { kind: "tool", connectorId, tool: tool.name },
        policy,
      }),
    );
  };
  return (
    <details open={unfolded} data-testid="connector-tools" className="mt-1 text-xs text-gray-600">
      <summary className="cursor-pointer select-none">
        {count} · {claims}
      </summary>
      <ul className="mt-1 flex flex-col gap-1">
        {tools.map((tool) => {
          const ask = asksFirst(tool, policyOf(tool));
          return (
            <li
              key={tool.name}
              data-testid="connector-tool"
              data-tool={tool.name}
              data-read-only={tool.readOnly}
              data-asks={ask}
              className="flex items-center gap-2"
            >
              <span className="min-w-0 flex-1 truncate" title={tool.description}>
                <span data-testid="connector-tool-name" className="font-mono">
                  {tool.name}
                </span>
                <span
                  data-testid="connector-tool-claim"
                  className={tool.readOnly ? "text-green-700" : "text-gray-400"}
                >
                  {" · "}
                  {tool.readOnly ? t("connectors.tools.readOnly") : t("connectors.tools.changes")}
                </span>
              </span>
              <select
                data-testid="connector-tool-approval"
                aria-label={t("approvals.tools.select", { tool: tool.name })}
                value={ask ? "ask" : "always"}
                onChange={(event) => choose(tool, event.target.value === "ask")}
                className="shrink-0 rounded-[6px] border border-gray-300 bg-white px-1 py-0.5 text-xs"
              >
                <option value="ask">{t("approvals.tools.ask")}</option>
                <option value="always">{t("approvals.tools.always")}</option>
              </select>
            </li>
          );
        })}
      </ul>
      <p className="mt-1 text-gray-500">{t("connectors.tools.note")}</p>
    </details>
  );
}
