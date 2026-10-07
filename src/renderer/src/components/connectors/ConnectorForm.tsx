import { type FormEvent, useEffect, useState } from "react";
import type { SecretStorageStatus } from "../../../../core/api";
import { core } from "../../core";
import { errorMessage } from "../../errors";
import { useT } from "../../i18n";
import { SecretStorageNotice } from "../providers/ProviderForm";
import {
  choiceListClass,
  compactChoiceRadioClass,
  compactChoiceRowClass,
  errorTextClass,
  fieldLabelClass,
  ghostButtonClass,
  hintClass,
  inputClass,
  primaryButtonClass,
} from "../ui";

/** Non-empty lines, trimmed. */
const lines = (text: string) =>
  text
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean);

/** "NAME=value" lines as variables, or null if a line has no name. */
function parseEnvLines(text: string): Record<string, string> | null {
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
      className="flex flex-col gap-3"
    >
      <fieldset>
        <legend className={`mb-1.5 ${fieldLabelClass}`}>{t("remoteConnectors.form.kind")}</legend>
        <div className={choiceListClass}>
          {(["local", "remote"] as const).map((each) => (
            <label key={each} className={compactChoiceRowClass}>
              <input
                type="radio"
                name="connector-kind"
                value={each}
                checked={kind === each}
                onChange={() => setKind(each)}
                className={compactChoiceRadioClass}
              />
              {t(`remoteConnectors.form.kind.${each}`)}
            </label>
          ))}
        </div>
      </fieldset>
      <label className={fieldLabelClass}>
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
          <label className={fieldLabelClass}>
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
            <span className={hintClass}>{t("remoteConnectors.form.urlHint")}</span>
          </label>
          <details className="text-[13px] leading-5 text-ink-secondary">
            <summary className="cursor-pointer select-none hover:text-ink">
              {t("remoteConnectors.form.client")}
            </summary>
            <p className={hintClass}>{t("remoteConnectors.client.hint")}</p>
            <label className={`mt-2 ${fieldLabelClass}`}>
              {t("remoteConnectors.client.id")}
              <input
                value={clientId}
                onChange={(event) => setClientId(event.target.value)}
                spellCheck={false}
                autoComplete="off"
                className={`${inputClass} font-mono`}
              />
            </label>
            <label className={`mt-2 ${fieldLabelClass}`}>
              {t("remoteConnectors.client.secret")}
              <input
                type="password"
                value={clientSecret}
                onChange={(event) => setClientSecret(event.target.value)}
                autoComplete="off"
                className={`${inputClass} font-mono`}
              />
              <span className={hintClass}>{t("remoteConnectors.client.secretHint")}</span>
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
        <button type="button" onClick={onDone} className={ghostButtonClass}>
          {t("connectors.form.cancel")}
        </button>
      </div>
      {error && (
        <p role="alert" className={errorTextClass}>
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
      <label className={fieldLabelClass}>
        {t("connectors.form.command")}
        <input
          required
          value={command}
          onChange={(event) => setCommand(event.target.value)}
          placeholder="npx"
          spellCheck={false}
          className={`${inputClass} font-mono`}
        />
        <span className={hintClass}>{t("connectors.form.commandHint")}</span>
      </label>
      <label className={fieldLabelClass}>
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
      <label className={fieldLabelClass}>
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
        <span className={hintClass}>{t("connectors.form.envHint")}</span>
        {props.envInvalid && (
          <span role="alert" className="mt-1 block text-[12px] leading-[18px] text-danger">
            {t("connectors.form.envInvalid")}
          </span>
        )}
      </label>
    </>
  );
}
