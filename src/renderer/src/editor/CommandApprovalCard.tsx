import { useId } from "react";
import type { CommandApprovalRequest } from "../../../core/api";
import { useApprovals } from "../approvals";
import { FORMAT_LABELS } from "../exports";
import { useT } from "../i18n";
import { folderName } from "../linkedFolders";
import { ThirdPartyLine } from "./ThirdPartyLine";
import { withShortcut } from "./withShortcut";

/**
 * A command of the assistant's waiting for the User (#1110), as the approval cards are: what it
 * would do outside IncarnaMind, saving a Mind as a file outside the Output location or reading a
 * folder's files to add them; the file it would write, or the files it would read, by name; and
 * Allow or Don't allow. Never "always": each such call asks.
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
  const title =
    ask.kind === "export"
      ? t("commands.ask.export.title", {
          mind: ask.mind.name,
          format: t(FORMAT_LABELS[ask.format]),
          folder: folderName(ask.folder),
        })
      : ask.files.length === 1
        ? t("commands.ask.add.title.one", { folder: folderName(ask.folder) })
        : t("commands.ask.add.title", { folder: folderName(ask.folder), count: ask.files.length });
  const names = ask.kind === "export" ? [ask.fileName] : ask.files;
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
        <p className="approval-explanation">
          {t(ask.kind === "export" ? "commands.ask.export.text" : "commands.ask.add.text")}
        </p>
        <ThirdPartyLine request={request} />
      </div>
      <div className="approval-details">
        {/* A path has no spaces to wrap at: it breaks anywhere rather than widen the card. */}
        <p data-testid="approval-folder" className="approval-label break-all">
          {ask.folder}
        </p>
        <ul
          data-testid="approval-files"
          className="approval-args max-h-40 list-none overflow-y-auto"
        >
          {names.map((name) => (
            <li key={name} data-testid="approval-file">
              {name}
            </li>
          ))}
        </ul>
      </div>
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
