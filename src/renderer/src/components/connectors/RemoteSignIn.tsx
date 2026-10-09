import { type FormEvent, useState } from "react";
import type { RemoteConnector } from "../../../../core/api";
import { core } from "../../core";
import { errorMessage } from "../../errors";
import { useT } from "../../i18n";
import { useAppStore } from "../../store";
import { buttonStyle, fieldLabelClass, hintClass, inputClass, noticeClass } from "../ui";

/** A text button inside a line, e.g. "Sign out". */
const linkButtonClass =
  "rounded-sm text-accent underline-offset-2 hover:text-accent-strong hover:underline";

/** Runs an action; a failure shows in the app's error banner. */
async function act(action: () => Promise<unknown>) {
  try {
    await action();
  } catch (failure) {
    useAppStore.setState({ actionError: errorMessage(failure) });
  }
}

/**
 * A remote Connector's sign-in, under its row: signing in (in the browser)
 * when the server requires it, waiting for the browser, signing out, and the
 * OAuth app it signs in with when the service can't register IncarnaMind.
 */
export function RemoteSignIn({ connector }: { connector: RemoteConnector }) {
  const t = useT();
  const { id, state, signIn, clientId } = connector;
  const [editingClient, setEditingClient] = useState(false);
  const needsClient = signIn.error?.kind === "client-id-required";

  if (state === "signing-in") {
    return (
      <div
        data-testid="connector-signing-in"
        className={`mt-1 flex flex-wrap items-center gap-2 ${noticeClass}`}
      >
        <span className="flex-1">{t("remoteConnectors.signingIn")}</span>
        <button
          type="button"
          data-testid="connector-sign-in-cancel"
          onClick={() => void act(() => core.cancelConnectorSignIn(id))}
          className={buttonStyle("secondary", "sm")}
        >
          {t("remoteConnectors.cancel")}
        </button>
      </div>
    );
  }

  const clientLine = clientId && (
    <p className="mt-1 flex flex-wrap items-center gap-2 text-[12px] leading-[18px] text-ink-meta">
      <span>{t("remoteConnectors.client.using", { clientId })}</span>
      <button
        type="button"
        onClick={() => void act(() => core.setConnectorClient(id, null))}
        className={linkButtonClass}
      >
        {t("remoteConnectors.client.forget")}
      </button>
    </p>
  );

  if (state === "needs-sign-in") {
    return (
      <div data-testid="connector-sign-in-notice" className={`mt-1 ${noticeClass}`}>
        <p>
          {signIn.expired
            ? t("remoteConnectors.signIn.expired")
            : t("remoteConnectors.signIn.required")}
        </p>
        {signIn.error && (
          <p role="alert" data-testid="connector-sign-in-error" className="mt-1 text-danger">
            {t(`remoteConnectors.signIn.error.${signIn.error.kind}`)}
          </p>
        )}
        {needsClient || editingClient ? (
          <ClientForm
            connector={connector}
            onDone={() => setEditingClient(false)}
            cancellable={!needsClient}
          />
        ) : (
          <div className="mt-2 flex flex-wrap items-center gap-3">
            <button
              type="button"
              data-testid="connector-sign-in"
              onClick={() => void act(() => core.signInToConnector(id))}
              className={buttonStyle("primary", "sm")}
            >
              {signIn.expired ? t("remoteConnectors.signInAgain") : t("remoteConnectors.signIn")}
            </button>
            {!clientId && (
              <button
                type="button"
                onClick={() => setEditingClient(true)}
                className={`text-[12px] ${linkButtonClass}`}
              >
                {t("remoteConnectors.form.client")}
              </button>
            )}
          </div>
        )}
        {clientLine}
      </div>
    );
  }

  if (!signIn.signedIn && !clientId) return null;
  return (
    <div className="text-[12px] leading-[18px] text-ink-meta">
      {signIn.signedIn && (
        <p className="flex flex-wrap items-center gap-2">
          <span>{t("remoteConnectors.signedIn")}</span>
          <button
            type="button"
            data-testid="connector-sign-out"
            onClick={() => void act(() => core.signOutOfConnector(id))}
            className={linkButtonClass}
          >
            {t("remoteConnectors.signOut")}
          </button>
        </p>
      )}
      {clientLine}
    </div>
  );
}

/** The OAuth app the User registered with the service: its client ID, and secret if it has one. */
function ClientForm({
  connector,
  onDone,
  cancellable,
}: {
  connector: RemoteConnector;
  onDone(): void;
  /** False when the service can't be signed in to without one. */
  cancellable: boolean;
}) {
  const t = useT();
  const [clientId, setClientId] = useState(connector.clientId ?? "");
  const [clientSecret, setClientSecret] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    setBusy(true);
    setError(null);
    try {
      await core.setConnectorClient(connector.id, {
        clientId,
        ...(clientSecret.trim() ? { clientSecret } : {}),
      });
      onDone();
      await core.signInToConnector(connector.id);
    } catch (failure) {
      setError(errorMessage(failure));
    } finally {
      setBusy(false);
    }
  };

  return (
    <form
      data-testid="connector-client-form"
      onSubmit={(event) => void submit(event)}
      className="mt-2 flex flex-col gap-2"
    >
      <p className={hintClass}>{t("remoteConnectors.client.hint")}</p>
      <label className={fieldLabelClass}>
        {t("remoteConnectors.client.id")}
        <input
          required
          value={clientId}
          onChange={(event) => setClientId(event.target.value)}
          spellCheck={false}
          autoComplete="off"
          className={`${inputClass} font-mono`}
        />
      </label>
      <label className={fieldLabelClass}>
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
      <div className="flex flex-wrap gap-2">
        <button
          type="submit"
          disabled={busy || !clientId.trim()}
          className={buttonStyle("primary", "sm")}
        >
          {t("remoteConnectors.client.save")}
        </button>
        {cancellable && (
          <button type="button" onClick={onDone} className={buttonStyle("ghost", "sm")}>
            {t("remoteConnectors.cancel")}
          </button>
        )}
      </div>
      {error && (
        <p role="alert" className="text-[12px] leading-[18px] text-danger">
          {error}
        </p>
      )}
    </form>
  );
}
