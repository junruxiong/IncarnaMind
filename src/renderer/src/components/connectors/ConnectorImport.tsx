import { type ChangeEvent, useEffect, useState } from "react";
import type { ConnectorImportEntry, SecretStorageStatus } from "../../../../core/api";
import { core } from "../../core";
import { errorMessage } from "../../errors";
import { useT } from "../../i18n";
import { SecretStorageNotice } from "../providers/ProviderForm";
import { buttonClass, inputClass, primaryButtonClass } from "../providers/shared";
import { commandLine } from "./commandLine";

/**
 * Imports Connectors from Claude Desktop's or Cursor's `mcpServers` JSON,
 * pasted or read from a file. It shows what will be added, and what won't
 * and why, before anything is added.
 */
export function ConnectorImport({ onDone }: { onDone(): void }) {
  const t = useT();
  const [json, setJson] = useState("");
  const [preview, setPreview] = useState<ConnectorImportEntry[] | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [secretStorage, setSecretStorage] = useState<SecretStorageStatus | null>(null);

  useEffect(() => {
    core.getSecretStorage().then(setSecretStorage, () => undefined);
  }, []);

  const showPreview = async (text: string) => {
    setError(null);
    setPreview(null);
    try {
      setPreview(await core.previewConnectorImport(text));
    } catch (failure) {
      setError(errorMessage(failure));
    }
  };

  const pickFile = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file) return;
    const text = await file.text();
    setJson(text);
    await showPreview(text);
  };

  const adding = preview?.filter((entry) => entry.action === "add") ?? [];
  const needsKeychain = adding.some((entry) => entry.env.length > 0);
  const blocked = needsKeychain && secretStorage !== null && !secretStorage.canSave;

  const confirm = async () => {
    setBusy(true);
    setError(null);
    try {
      await core.importConnectors(json);
      onDone();
    } catch (failure) {
      setError(errorMessage(failure));
    } finally {
      setBusy(false);
    }
  };

  const acceptPlainText = async () => {
    try {
      setSecretStorage(await core.acceptPlainTextSecretStorage());
    } catch (failure) {
      setError(errorMessage(failure));
    }
  };

  return (
    <div data-testid="connector-import-form" className="mt-3 flex flex-col gap-3">
      <p className="text-sm text-gray-600">{t("connectors.import.body")}</p>
      <label className="text-sm text-gray-600">
        {t("connectors.import.json")}
        <textarea
          value={json}
          onChange={(event) => {
            setJson(event.target.value);
            setPreview(null);
          }}
          rows={6}
          spellCheck={false}
          placeholder={
            '{\n  "mcpServers": {\n    "github": { "command": "npx", "args": ["-y", "…"] }\n  }\n}'
          }
          className={`${inputClass} font-mono text-xs`}
        />
      </label>
      <div className="flex flex-wrap items-center gap-2">
        <label className={`${buttonClass} cursor-pointer`}>
          {t("connectors.import.file")}
          <input
            type="file"
            accept=".json,application/json"
            data-testid="connector-import-file"
            onChange={(event) => void pickFile(event)}
            className="sr-only"
          />
        </label>
        <button
          type="button"
          disabled={!json.trim()}
          onClick={() => void showPreview(json)}
          className={buttonClass}
        >
          {t("connectors.import.preview")}
        </button>
        <button type="button" onClick={onDone} className={buttonClass}>
          {t("connectors.form.cancel")}
        </button>
      </div>

      {preview && (
        <>
          <ul data-testid="connector-import-preview" className="flex flex-col gap-1.5">
            {preview.map((entry) => (
              <li
                key={entry.name}
                data-action={entry.action}
                className="rounded-[9px] border border-gray-200 px-3 py-1.5 text-sm"
              >
                <div className="flex items-center gap-2">
                  <span className="min-w-0 flex-1 truncate font-medium">{entry.name}</span>
                  <span
                    className={`shrink-0 text-xs ${entry.action === "add" ? "text-green-700" : "text-gray-500"}`}
                  >
                    {t(`connectors.import.action.${entry.action}`)}
                  </span>
                </div>
                {entry.command && (
                  <p className="truncate font-mono text-xs text-gray-500">
                    {commandLine(entry.command, entry.args)}
                  </p>
                )}
                {entry.env.length > 0 && (
                  <p className="text-xs text-gray-500">
                    {t("connectors.import.env", { names: entry.env.join(", ") })}
                  </p>
                )}
              </li>
            ))}
          </ul>
          {needsKeychain && secretStorage && !secretStorage.canSave && (
            <SecretStorageNotice status={secretStorage} onAccept={() => void acceptPlainText()} />
          )}
          {adding.length === 0 ? (
            <p className="text-sm text-gray-600">{t("connectors.import.nothing")}</p>
          ) : (
            <div>
              <button
                type="button"
                data-testid="connector-import-confirm"
                disabled={busy || blocked}
                onClick={() => void confirm()}
                className={primaryButtonClass}
              >
                {busy
                  ? t("connectors.import.adding")
                  : adding.length === 1
                    ? t("connectors.import.confirm.one")
                    : t("connectors.import.confirm", { count: adding.length })}
              </button>
            </div>
          )}
        </>
      )}
      {error && (
        <p role="alert" className="text-sm text-red-700">
          {error}
        </p>
      )}
    </div>
  );
}
