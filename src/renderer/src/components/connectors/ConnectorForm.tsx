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

/**
 * Adds a Connector: a local one by its name, command, arguments and
 * environment, or a remote one by its name and URL (and, for a service that
 * can't register IncarnaMind by itself, the User's own OAuth app).
 */
export function ConnectorForm({ onDone }: { onDone(): void }) {
  const t = useT();
  const [kind, setKind] = useState<"local" | "remote">("local");
  const [name, setName] = useState("");
  const [command, setCommand] = useState("");
  const [args, setArgs] = useState("");
  const [env, setEnv] = useState("");
  const [url, setUrl] = useState("");
  const [clientId, setClientId] = useState("");
  const [clientSecret, setClientSecret] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [secretStorage, setSecretStorage] = useState<SecretStorageStatus | null>(null);

  useEffect(() => {
    core.getSecretStorage().then(setSecretStorage, () => undefined);
  }, []);

  const remote = kind === "remote";
  const variables = parseEnvLines(env);
  const hasSecrets = remote
    ? clientSecret.trim() !== ""
    : variables !== null && Object.keys(variables).length > 0;
  const blocked = hasSecrets && secretStorage !== null && !secretStorage.canSave;
  const invalid = !remote && variables === null;

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    if (invalid) return;
    setBusy(true);
    setError(null);
    try {
      if (remote) {
        const client = clientId.trim()
          ? { clientId, ...(clientSecret.trim() ? { clientSecret } : {}) }
          : undefined;
        await core.addConnector({ name, url, ...(client ? { client } : {}) });
      } else {
        await core.addConnector({ name, command, args: lines(args), env: variables ?? {} });
      }
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
      <fieldset className="text-sm text-gray-600">
        <legend>{t("remoteConnectors.form.kind")}</legend>
        <div className="mt-1 flex flex-wrap gap-4">
          {(["local", "remote"] as const).map((each) => (
            <label key={each} className="flex items-center gap-1.5 text-gray-800">
              <input
                type="radio"
                name="connector-kind"
                value={each}
                checked={kind === each}
                onChange={() => setKind(each)}
                className="accent-gray-800"
              />
              {t(`remoteConnectors.form.kind.${each}`)}
            </label>
          ))}
        </div>
      </fieldset>
      <label className="text-sm text-gray-600">
        {t("connectors.form.name")}
        <input
          required
          value={name}
          onChange={(event) => setName(event.target.value)}
          placeholder={remote ? "Linear" : "GitHub"}
          className={inputClass}
        />
      </label>
      {remote ? (
        <>
          <label className="text-sm text-gray-600">
            {t("remoteConnectors.form.url")}
            <input
              required
              type="url"
              value={url}
              onChange={(event) => setUrl(event.target.value)}
              placeholder="https://mcp.example.com/mcp"
              spellCheck={false}
              className={`${inputClass} font-mono`}
            />
            <span className="mt-1 block text-xs text-gray-500">
              {t("remoteConnectors.form.urlHint")}
            </span>
          </label>
          <details className="text-sm text-gray-600">
            <summary className="cursor-pointer select-none">
              {t("remoteConnectors.form.client")}
            </summary>
            <p className="mt-1 text-xs text-gray-500">{t("remoteConnectors.client.hint")}</p>
            <label className="mt-2 block">
              {t("remoteConnectors.client.id")}
              <input
                value={clientId}
                onChange={(event) => setClientId(event.target.value)}
                spellCheck={false}
                autoComplete="off"
                className={`${inputClass} font-mono`}
              />
            </label>
            <label className="mt-2 block">
              {t("remoteConnectors.client.secret")}
              <input
                type="password"
                value={clientSecret}
                onChange={(event) => setClientSecret(event.target.value)}
                autoComplete="off"
                className={`${inputClass} font-mono`}
              />
              <span className="mt-1 block text-xs text-gray-500">
                {t("remoteConnectors.client.secretHint")}
              </span>
            </label>
          </details>
        </>
      ) : (
        <LocalFields
          command={command}
          setCommand={setCommand}
          args={args}
          setArgs={setArgs}
          env={env}
          setEnv={setEnv}
          envInvalid={variables === null}
        />
      )}

      {hasSecrets && secretStorage && !secretStorage.canSave && (
        <SecretStorageNotice status={secretStorage} onAccept={() => void acceptPlainText()} />
      )}

      <div className="flex flex-wrap items-center gap-2">
        <button type="submit" disabled={busy || invalid || blocked} className={primaryButtonClass}>
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

/** A local Connector's command, arguments and environment. */
function LocalFields(props: {
  command: string;
  setCommand(value: string): void;
  args: string;
  setArgs(value: string): void;
  env: string;
  setEnv(value: string): void;
  envInvalid: boolean;
}) {
  const t = useT();
  const { command, setCommand, args, setArgs, env, setEnv } = props;
  return (
    <>
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
        {props.envInvalid && (
          <span role="alert" className="mt-1 block text-xs text-red-700">
            {t("connectors.form.envInvalid")}
          </span>
        )}
      </label>
    </>
  );
}
