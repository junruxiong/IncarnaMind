import { useEffect, useId, useState } from "react";
import type {
  AgentClient,
  AgentSetupPreview,
  LibraryGroup,
  SecretStorageStatus,
} from "../../../../core/api";
import { clientInSentence } from "../../../../shared/agentNames";
import { core } from "../../core";
import { useT } from "../../i18n";
import { useAppStore } from "../../store";
import { InlinePath } from "../PathText";
import { SecretStorageNotice } from "../providers/ProviderForm";
import { SLOT, Slotted } from "../Slotted";
import {
  buttonClass,
  dialogActionsClass,
  dialogBodyClass,
  dialogClass,
  dialogTextClass,
  dialogTitleClass,
  errorTextClass,
  fieldLabelClass,
  hintClass,
  inputClass,
  primaryButtonClass,
} from "../ui";
import { useModal } from "../useModal";

/**
 * The one confirmation before an agent is connected (#351), which is also the
 * User's consent to its data flow: what it will be able to read, where that
 * goes (its own AI model, under its service's terms, which may train on it),
 * the Folders it may read (none chosen until the User picks), and, for an
 * agent whose settings file IncarnaMind writes, exactly what it adds there.
 * Open while `client` is set; `onAllow` gets the Folders chosen.
 */
export function ConnectAgentDialog({
  client,
  folders,
  busy,
  problem,
  onAllow,
  onClose,
}: {
  client: AgentClient | null;
  folders: readonly LibraryGroup[];
  busy: boolean;
  /** Why the last try to connect failed, shown here, where it happened. */
  problem: string | null;
  onAllow(folderIds: string[], name: string): void;
  onClose(): void;
}) {
  const t = useT();
  const dialog = useModal(client !== null, { focusDialog: true });
  const [chosen, setChosen] = useState<ReadonlySet<string>>(new Set());
  /** The name the User gives "Other agent". */
  const [name, setName] = useState("");
  const [preview, setPreview] = useState<AgentSetupPreview | null>(null);
  /** Where the agent's key can be kept: with no keyring (Linux), only once the User allows plain text. */
  const [storage, setStorage] = useState<SecretStorageStatus | null>(null);
  const reportError = useAppStore((state) => state.reportError);
  const titleId = useId();
  const clientId = client?.id ?? null;

  // A new agent starts with no Folder chosen, and its own preview.
  useEffect(() => {
    setChosen(new Set());
    setName("");
    setPreview(null);
    if (!clientId) return;
    let current = true;
    core.previewAgentSetup(clientId).then((value) => {
      if (current) setPreview(value);
    }, reportError);
    core.getSecretStorage().then((value) => {
      if (current) setStorage(value);
    }, reportError);
    return () => {
      current = false;
    };
  }, [clientId, reportError]);
  const canKeepKey = storage?.canSave !== false;

  const agent = client ? name.trim() || clientInSentence(client, t) : "";
  const toggle = (folderId: string, on: boolean) => {
    const next = new Set(chosen);
    if (on) next.add(folderId);
    else next.delete(folderId);
    setChosen(next);
  };

  return (
    <dialog
      ref={dialog}
      tabIndex={-1}
      onClose={(event) => {
        // React passes the event on up its tree, to Settings' own dialog, which would close too.
        event.stopPropagation();
        onClose();
      }}
      aria-labelledby={titleId}
      data-testid="connect-agent-dialog"
      className={`${dialogClass} w-[30rem]`}
    >
      <div className={dialogBodyClass}>
        <h2 id={titleId} className={dialogTitleClass}>
          {t("agents.connect.title", { agent })}
        </h2>
        <p className={dialogTextClass}>{t("agents.connect.body", { agent })}</p>
        {client?.id === "other" && (
          <label className="flex flex-col gap-1">
            <span className={fieldLabelClass}>{t("agents.connect.name")}</span>
            <input
              type="text"
              data-testid="connect-agent-name"
              value={name}
              placeholder={t("agents.connect.namePlaceholder")}
              onChange={(event) => setName(event.target.value)}
              className={inputClass}
            />
          </label>
        )}
        <fieldset className="flex flex-col gap-1">
          <legend className={fieldLabelClass}>{t("agents.connect.folders")}</legend>
          {folders.length === 0 ? (
            <p data-testid="connect-agent-no-folders" className={hintClass}>
              {t("agents.connect.noFolders")}
            </p>
          ) : (
            <div className="mt-1 flex max-h-48 flex-col overflow-y-auto">
              {folders.map((folder) => (
                <label
                  key={folder.id}
                  className="flex min-h-8 cursor-pointer items-center gap-2 text-ui text-ink"
                >
                  <input
                    type="checkbox"
                    data-testid="connect-agent-folder"
                    checked={chosen.has(folder.id)}
                    onChange={(event) => toggle(folder.id, event.target.checked)}
                    className="size-4 shrink-0 accent-ink"
                  />
                  <span className="min-w-0 truncate" title={folder.name}>
                    {folder.name}
                  </span>
                </label>
              ))}
            </div>
          )}
        </fieldset>
        {preview && (
          <div data-testid="connect-agent-change" className="flex flex-col gap-1">
            <p className={dialogTextClass}>
              <Slotted
                message={t("agents.connect.fileChange", { path: SLOT })}
                slot={<InlinePath path={preview.path} />}
              />
            </p>
            <details>
              <summary className="cursor-pointer text-[13px] leading-5 text-ink-secondary">
                {t("agents.connect.showChange")}
              </summary>
              <pre className="mt-1 max-h-48 overflow-auto rounded-md bg-frame p-2 font-mono text-[12px] leading-[18px] break-all whitespace-pre-wrap text-ink">
                {preview.change}
              </pre>
            </details>
          </div>
        )}
        {client?.skillPath && (
          <p data-testid="connect-agent-skill" className={dialogTextClass}>
            <Slotted
              message={t("agents.connect.skill", { path: SLOT, agent })}
              slot={<InlinePath path={client.skillPath} />}
            />
          </p>
        )}
        {folders.length > 0 && chosen.size === 0 && (
          <p className={hintClass}>{t("agents.connect.pickOne")}</p>
        )}
        {storage && !storage.canSave && (
          <SecretStorageNotice
            status={storage}
            onAccept={() => core.acceptPlainTextSecretStorage().then(setStorage, reportError)}
          />
        )}
        {problem && (
          <p data-testid="connect-agent-problem" role="alert" className={errorTextClass}>
            {problem}
          </p>
        )}
        <div className={dialogActionsClass}>
          <button type="button" onClick={onClose} className={buttonClass}>
            {t("agents.connect.cancel")}
          </button>
          <button
            type="button"
            data-testid="connect-agent-allow"
            disabled={chosen.size === 0 || busy || !canKeepKey}
            onClick={() => onAllow([...chosen], name)}
            className={primaryButtonClass}
          >
            {t("agents.connect.allow")}
          </button>
        </div>
      </div>
    </dialog>
  );
}
