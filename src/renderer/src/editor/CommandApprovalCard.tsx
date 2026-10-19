import { useId } from "react";
import { COMMAND_AT_ONCE, type CommandApprovalRequest, type CommandAsk } from "../../../core/api";
import type { MessageKey } from "../../../shared/i18n";
import { useApprovals } from "../approvals";
import { FORMAT_LABELS } from "../exports";
import { useT } from "../i18n";
import { folderName } from "../linkedFolders";
import type { Translate } from "../objects";
import { ThirdPartyLine } from "./ThirdPartyLine";
import { withShortcut } from "./withShortcut";

type Change = Extract<CommandAsk, { kind: "change" }>;

/** A change's question, for several things, then for one. */
const CHANGE: Readonly<Record<Change["command"], [MessageKey, MessageKey]>> = {
  "document.rename": [
    "commands.ask.change.document.rename",
    "commands.ask.change.document.rename.one",
  ],
  "mind.rename": ["commands.ask.change.mind.rename", "commands.ask.change.mind.rename.one"],
  "document.moveTo": [
    "commands.ask.change.document.moveTo",
    "commands.ask.change.document.moveTo.one",
  ],
  "mind.moveTo": ["commands.ask.change.mind.moveTo", "commands.ask.change.mind.moveTo.one"],
  "document.tags": ["commands.ask.change.document.tags", "commands.ask.change.document.tags.one"],
  "document.delete": [
    "commands.ask.change.document.delete",
    "commands.ask.change.document.delete.one",
  ],
  "mind.delete": ["commands.ask.change.mind.delete", "commands.ask.change.mind.delete.one"],
};

const WHY: Readonly<Record<Change["reason"], MessageKey>> = {
  many: "commands.ask.why.many",
  afterReading: "commands.ask.why.afterReading",
  focusedMind: "commands.ask.why.focusedMind",
};

/** The card's question, why it asks, the path it reaches (if any) and the names it lists. */
function wordsOf(
  t: Translate,
  ask: CommandAsk,
): { title: string; why: string; path: string | null; names: string[] } {
  switch (ask.kind) {
    case "export":
      return {
        title: t("commands.ask.export.title", {
          mind: ask.mind.name,
          format: t(FORMAT_LABELS[ask.format]),
          folder: folderName(ask.folder),
        }),
        why: t("commands.ask.export.text"),
        path: ask.folder,
        names: [ask.fileName],
      };
    case "add":
      return {
        title:
          ask.files.length === 1
            ? t("commands.ask.add.title.one", { folder: folderName(ask.folder) })
            : t("commands.ask.add.title", {
                folder: folderName(ask.folder),
                count: ask.files.length,
              }),
        why: t("commands.ask.add.text"),
        path: ask.folder,
        names: ask.files,
      };
    case "change": {
      const [many, one] = CHANGE[ask.command];
      const into = ask.into ?? "";
      return {
        title: t(ask.count === 1 ? one : many, {
          count: ask.count,
          name: ask.names[0] ?? "",
          folder: into,
          tag: into,
        }),
        why: t(WHY[ask.reason], { max: COMMAND_AT_ONCE }),
        path: null,
        names: ask.count === 1 ? [] : ask.names,
      };
    }
  }
}

/**
 * A command of the assistant's waiting for the User (#1110), as the approval cards are: what it
 * would do, saving a Mind as a file outside the Output location, reading a folder's files to add
 * them, or a change inside IncarnaMind that needs their say, and why it asks; the folder and the
 * names it reaches; and Allow or Don't allow. Never "always": each such call asks.
 */
export function CommandApprovalCard({
  request,
  keys,
}: {
  request: CommandApprovalRequest;
  keys?: string;
}) {
  const t = useT();
  const respond = useApprovals((state) => state.respond);
  const titleId = useId();
  const { ask } = request;
  const { title, why, path, names } = wordsOf(t, ask);
  const allow = t("commands.ask.allow");
  return (
    <div
      contentEditable={false}
      role="alertdialog"
      aria-labelledby={titleId}
      data-testid="approval-card"
      data-tool={request.tool}
      data-kind="command"
      data-ask={ask.kind}
      className="approval-card"
    >
      <div className="approval-head">
        <p id={titleId} className="approval-title">
          <span aria-hidden="true" className="approval-dot" />
          <span className="approval-title-text">{title}</span>
        </p>
        <p className="approval-explanation">{why}</p>
        <ThirdPartyLine request={request} />
      </div>
      {(path !== null || names.length > 0) && (
        <div className="approval-details">
          {/* A path has no spaces to wrap at: it breaks anywhere rather than widen the card. */}
          {path !== null && (
            <p data-testid="approval-folder" className="approval-label break-all">
              {path}
            </p>
          )}
          <ul
            data-testid="approval-files"
            className="approval-args max-h-40 list-none overflow-y-auto"
          >
            {names.map((name, index) => (
              // biome-ignore lint/suspicious/noArrayIndexKey: two Documents may have the same name.
              <li key={index} data-testid="approval-file">
                {name}
              </li>
            ))}
          </ul>
        </div>
      )}
      <div className="approval-actions">
        <button
          type="button"
          data-testid="approval-allow-once"
          onClick={() => respond(request.requestId, "allow-once")}
          className="approval-button approval-button--primary"
          {...withShortcut(allow, keys)}
        >
          {allow}
          {keys && <span className="approval-keys">{keys}</span>}
        </button>
        <button
          type="button"
          data-testid="approval-deny"
          onClick={() => respond(request.requestId, "deny")}
          className="approval-button approval-button--ghost"
        >
          {t("commands.ask.deny")}
        </button>
      </div>
    </div>
  );
}
