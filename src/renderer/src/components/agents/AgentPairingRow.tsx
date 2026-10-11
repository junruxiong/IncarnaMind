import { useEffect, useRef, useState } from "react";
import type {
  AgentActivity,
  AgentClientId,
  AgentPairing,
  AgentSetupResult,
  LibraryGroup,
} from "../../../../core/api";
import { core } from "../../core";
import { useLanguage, useT } from "../../i18n";
import { useAppStore } from "../../store";
import {
  buttonStyle,
  ghostButtonClass,
  hintClass,
  inputClass,
  rowButtonsClass,
  rowTextClass,
  rowTitleClass,
  ruledRowClass,
} from "../ui";
import { activityWords, doneWords, helpUrl, timeWords } from "./agentWords";

/** What setting this agent up just did, shown under it until the page goes. */
export interface JustSetUp {
  pairingId: string;
  result: AgentSetupResult;
}

/**
 * One connected agent in Settings → Agents (#351): its name, the Folders it
 * may read (changed in place), when it was last used, its activity (every
 * call, newest first), the details a technical User needs to set it up again,
 * and Disconnect, which stops it at once, as deleting a Connector does.
 */
export function AgentPairingRow({
  pairing,
  folders,
  justSetUp,
  onUndone,
}: {
  pairing: AgentPairing;
  folders: readonly LibraryGroup[];
  justSetUp: JustSetUp | null;
  onUndone(): void;
}) {
  const t = useT();
  const language = useLanguage();
  const reportError = useAppStore((state) => state.reportError);
  const [open, setOpen] = useState<"folders" | "activity" | null>(null);
  const [activity, setActivity] = useState<AgentActivity[] | null>(null);
  const [copied, setCopied] = useState(false);
  const [renaming, setRenaming] = useState(false);

  // The activity list follows each call while it is open.
  useEffect(() => {
    if (open !== "activity") return;
    core.listAgentActivity(pairing.id).then(setActivity, reportError);
    return core.on("agents.activity", (call) => {
      if (call.pairingId === pairing.id) setActivity((list) => [call, ...(list ?? [])]);
    });
  }, [open, pairing.id, reportError]);

  // The Folders as the User just chose them, shown at once; the core's answer then replaces them.
  const [chosen, setChosen] = useState(pairing.folderIds);
  useEffect(() => setChosen(pairing.folderIds), [pairing.folderIds]);
  const names = chosen
    .map((id) => folders.find((folder) => folder.id === id)?.name)
    .filter((name): name is string => name !== undefined);
  const setFolder = (folderId: string, on: boolean) => {
    const next = on ? [...chosen, folderId] : chosen.filter((id) => id !== folderId);
    setChosen(next);
    core.updateAgent(pairing.id, { folderIds: next }).catch((error: unknown) => {
      setChosen(pairing.folderIds);
      reportError(error);
    });
  };
  const copySettings = async () => {
    try {
      const setup = await core.copyAgentSetup(pairing.id);
      await navigator.clipboard.writeText(setup.text);
      setCopied(true);
    } catch (error) {
      reportError(error);
    }
  };
  const done = justSetUp?.pairingId === pairing.id ? justSetUp.result : null;
  const toggle = (what: "folders" | "activity") => setOpen(open === what ? null : what);

  return (
    <li data-testid="agent-pairing" data-name={pairing.name} className={ruledRowClass}>
      <div className="flex min-w-0 flex-col gap-0.5">
        {renaming ? (
          <RenameAgent pairing={pairing} onDone={() => setRenaming(false)} />
        ) : (
          <button
            type="button"
            data-testid="agent-pairing-name"
            title={pairing.name}
            onDoubleClick={() => setRenaming(true)}
            onKeyDown={(event) => {
              if (event.key === "F2") {
                event.preventDefault();
                setRenaming(true);
              }
            }}
            className={`${rowTitleClass} cursor-default truncate text-left`}
          >
            {pairing.name}
          </button>
        )}
        <span data-testid="agent-pairing-reads" className={rowTextClass}>
          {names.length > 0
            ? t("agents.pairing.reads", { folders: names.join(", ") })
            : t("agents.pairing.readsNothing")}
        </span>
        <span className={hintClass}>
          {pairing.lastUsedAt
            ? t("agents.pairing.lastUsed", { when: timeWords(pairing.lastUsedAt, language) })
            : t("agents.pairing.notUsed")}
        </span>
        {done && (
          <p data-testid="agent-setup-done" role="status" className={`${rowTextClass} mt-1`}>
            {doneWords(done, pairing.name, t)}
            {done.kind === "wrote" && (
              <>
                <span aria-hidden="true"> · </span>
                <button
                  type="button"
                  data-testid="agent-setup-undo"
                  onClick={() => core.undoAgentSetup(pairing.id).then(onUndone, reportError)}
                  className="rounded-sm font-semibold text-accent underline-offset-2 hover:underline"
                >
                  {t("agents.done.undo")}
                </button>
              </>
            )}
            {pairing.client && done.kind === "copy" && (
              <>
                <span aria-hidden="true"> · </span>
                <a
                  href={helpUrl(pairing.client as AgentClientId, language)}
                  target="_blank"
                  rel="noreferrer"
                  className="font-semibold text-accent underline-offset-2 hover:underline"
                >
                  {t("agents.settings.help")}
                </a>
              </>
            )}
          </p>
        )}
        {open === "folders" && (
          <fieldset data-testid="agent-pairing-folders" className="mt-2 flex flex-col">
            <legend className="text-[13px] leading-5 font-semibold text-ink">
              {t("agents.pairing.foldersTitle", { agent: pairing.name })}
            </legend>
            {folders.map((folder) => (
              <label
                key={folder.id}
                className="flex min-h-8 cursor-pointer items-center gap-2 text-ui text-ink"
              >
                <input
                  type="checkbox"
                  checked={chosen.includes(folder.id)}
                  onChange={(event) => setFolder(folder.id, event.target.checked)}
                  className="size-4 shrink-0 accent-ink"
                />
                <span className="min-w-0 truncate" title={folder.name}>
                  {folder.name}
                </span>
              </label>
            ))}
          </fieldset>
        )}
        {open === "activity" && (
          <ol
            data-testid="agent-activity"
            className="mt-2 flex max-h-64 flex-col gap-1 overflow-y-auto"
          >
            {activity?.length === 0 && <li className={hintClass}>{t("agents.activity.empty")}</li>}
            {activity?.map((call) => (
              <li
                key={call.id}
                data-testid="agent-activity-call"
                data-outcome={call.outcome}
                className="grid grid-cols-[auto_minmax(0,1fr)] gap-x-3 text-[13px] leading-5"
              >
                <span className="text-ink-meta">{timeWords(call.at, language)}</span>
                <span className="break-words text-ink-secondary">{activityWords(call, t)}</span>
              </li>
            ))}
          </ol>
        )}
        <details className="mt-1">
          <summary className="cursor-pointer text-[12px] leading-[18px] text-ink-meta">
            {t("agents.pairing.details")}
          </summary>
          <button
            type="button"
            data-testid="agent-copy-settings"
            onClick={() => void copySettings()}
            className={`${ghostButtonClass} mt-1`}
          >
            {copied ? t("agents.pairing.copied") : t("agents.pairing.copySettings")}
          </button>
        </details>
      </div>
      <div className={rowButtonsClass}>
        <button
          type="button"
          data-testid="agent-folders"
          aria-expanded={open === "folders"}
          onClick={() => toggle("folders")}
          className={buttonStyle("secondary", "sm")}
        >
          {t("agents.pairing.folders")}
        </button>
        <button
          type="button"
          data-testid="agent-activity-toggle"
          aria-expanded={open === "activity"}
          onClick={() => toggle("activity")}
          className={buttonStyle("secondary", "sm")}
        >
          {t("agents.pairing.activity")}
        </button>
        <button
          type="button"
          data-testid="agent-disconnect"
          onClick={() => core.disconnectAgent(pairing.id).catch(reportError)}
          className={buttonStyle("ghost", "sm")}
        >
          {t("agents.pairing.disconnect")}
        </button>
      </div>
    </li>
  );
}

/**
 * An agent's name, renamed in place as everywhere (a double-click or F2):
 * Enter or leaving the field saves, Esc cancels, and an empty name changes
 * nothing. Enter while a Chinese input method is composing only composes.
 */
function RenameAgent({ pairing, onDone }: { pairing: AgentPairing; onDone(): void }) {
  const t = useT();
  const reportError = useAppStore((state) => state.reportError);
  const [value, setValue] = useState(pairing.name);
  const field = useRef<HTMLInputElement>(null);
  const finished = useRef(false);

  useEffect(() => {
    field.current?.focus();
    field.current?.select();
  }, []);

  const finish = (save: boolean) => {
    if (finished.current) return;
    finished.current = true;
    const name = value.trim();
    if (save && name && name !== pairing.name) {
      core.updateAgent(pairing.id, { name }).catch(reportError);
    }
    onDone();
  };

  return (
    <input
      ref={field}
      value={value}
      data-testid="agent-pairing-rename"
      aria-label={t("agents.pairing.renameLabel", { agent: pairing.name })}
      onChange={(event) => setValue(event.target.value)}
      onKeyDown={(event) => {
        if (event.nativeEvent.isComposing) return;
        if (event.key === "Enter" || event.key === "Escape") {
          // Esc ends the renaming, not Settings.
          event.preventDefault();
          event.stopPropagation();
          finish(event.key === "Enter");
        }
      }}
      onBlur={() => finish(true)}
      className={inputClass}
    />
  );
}
