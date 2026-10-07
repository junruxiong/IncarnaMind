import { type FormEvent, useEffect, useState } from "react";
import type { SecretStorageStatus } from "../../../../core/api";
import { core } from "../../core";
import { errorMessage } from "../../errors";
import { useT } from "../../i18n";
import { SecretStorageNotice } from "../providers/ProviderForm";
import { buttonClass, inputClass, primaryButtonClass } from "../providers/shared";

/** Non-empty lines, trimmed. */
const lines = (text: string) =>
  text
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean);

/** "NAME=value" lines as variables, or null if a line has no name. */
export function parseEnvLines(text: string): Record<string, string> | null {
  const env: Record<string, string> = {};
  for (const line of text.split("\n")) {
    if (!line.trim()) continue;
    const equals = line.indexOf("=");
    const name = line.slice(0, Math.max(0, equals)).trim();
    if (!name) return null;
    env[name] = line.slice(equals + 1).replace(/\r$/, "");
  }
  return env;
}

/** Adds a local Connector: its name, command, arguments and environment. */
export function ConnectorForm({ onDone }: { onDone(): void }) {
  const t = useT();
  const [name, setName] = useState("");
  const [command, setCommand] = useState("");
  const [args, setArgs] = useState("");
  const [env, setEnv] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [secretStorage, setSecretStorage] = useState<SecretStorageStatus | null>(null);

  useEffect(() => {
    core.getSecretStorage().then(setSecretStorage, () => undefined);
  }, []);

  const variables = parseEnvLines(env);
  const hasSecrets = variables !== null && Object.keys(variables).length > 0;
  const blocked = hasSecrets && secretStorage !== null && !secretStorage.canSave;

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    if (!variables) return;
    setBusy(true);
    setError(null);
    try {
      await core.addConnector({ name, command, args: lines(args), env: variables });
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
    <form
      data-testid="connector-form"
      onSubmit={(event) => void submit(event)}
      className="mt-3 flex flex-col gap-3"
    >
      <label className="text-sm text-gray-600">
        {t("connectors.form.name")}
        <input
          required
          value={name}
          onChange={(event) => setName(event.target.value)}
          placeholder="GitHub"
          className={inputClass}
        />
      </label>
      <label className="text-sm text-gray-600">
        {t("connectors.form.command")}
        <input
          required
          value={command}
          onChange={(event) => setCommand(event.target.value)}
          placeholder="npx"
          spellCheck={false}
          className={`${inputClass} font-mono`}
        />
        <span className="mt-1 block text-xs text-gray-500">{t("connectors.form.commandHint")}</span>
      </label>
      <label className="text-sm text-gray-600">
        {t("connectors.form.args")}
        <textarea
          value={args}
          onChange={(event) => setArgs(event.target.value)}
          rows={3}
          placeholder={"-y\n@modelcontextprotocol/server-github"}
          spellCheck={false}
          className={`${inputClass} font-mono`}
        />
      </label>
      <label className="text-sm text-gray-600">
        {t("connectors.form.env")}
        <textarea
          value={env}
          onChange={(event) => setEnv(event.target.value)}
          rows={2}
          placeholder="GITHUB_PERSONAL_ACCESS_TOKEN=…"
          spellCheck={false}
          autoComplete="off"
          className={`${inputClass} font-mono`}
        />
        <span className="mt-1 block text-xs text-gray-500">{t("connectors.form.envHint")}</span>
        {variables === null && (
          <span role="alert" className="mt-1 block text-xs text-red-700">
            {t("connectors.form.envInvalid")}
          </span>
        )}
      </label>

      {hasSecrets && secretStorage && !secretStorage.canSave && (
        <SecretStorageNotice status={secretStorage} onAccept={() => void acceptPlainText()} />
      )}

      <div className="flex flex-wrap items-center gap-2">
        <button
          type="submit"
          disabled={busy || variables === null || blocked}
          className={primaryButtonClass}
        >
          {busy ? t("connectors.form.adding") : t("connectors.form.add")}
        </button>
        <button type="button" onClick={onDone} className={buttonClass}>
          {t("connectors.form.cancel")}
        </button>
      </div>
      {error && (
        <p role="alert" className="text-sm text-red-700">
          {error}
        </p>
      )}
    </form>
  );
}
