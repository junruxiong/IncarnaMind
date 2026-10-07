import { type FormEvent, useState } from "react";
import type { RemoteConnector } from "../../../../core/api";
import { core } from "../../core";
import { errorMessage } from "../../errors";
import { useT } from "../../i18n";
import { useAppStore } from "../../store";
import { inputClass } from "../providers/shared";

const smallButton =
  "rounded-[6px] border border-current/20 px-2 py-0.5 hover:bg-white/60 disabled:opacity-50";

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
        className="mt-1.5 flex flex-wrap items-center gap-2 rounded-[6px] bg-blue-50 px-2 py-1.5 text-sm text-blue-900"
      >
        <span className="flex-1">{t("remoteConnectors.signingIn")}</span>
        <button
          type="button"
          data-testid="connector-sign-in-cancel"
          onClick={() => void act(() => core.cancelConnectorSignIn(id))}
          className={`${smallButton} text-xs`}
        >
          {t("remoteConnectors.cancel")}
        </button>
      </div>
    );
  }

  const clientLine = clientId && (
    <p className="mt-1 flex flex-wrap items-center gap-2 text-xs text-gray-500">
      <span>{t("remoteConnectors.client.using", { clientId })}</span>
      <button
        type="button"
        onClick={() => void act(() => core.setConnectorClient(id, null))}
        className="underline hover:text-gray-800"
      >
        {t("remoteConnectors.client.forget")}
      </button>
    </p>
  );

  if (state === "needs-sign-in") {
    return (
      <div
        data-testid="connector-sign-in-notice"
        className="mt-1.5 rounded-[6px] bg-amber-50 px-2 py-1.5 text-sm text-amber-950"
      >
        <p>
          {signIn.expired
            ? t("remoteConnectors.signIn.expired")
            : t("remoteConnectors.signIn.required")}
        </p>
        {signIn.error && (
          <p role="alert" data-testid="connector-sign-in-error" className="mt-1 text-amber-900">
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
          <div className="mt-1 flex flex-wrap items-center gap-2 text-xs">
            <button
              type="button"
              data-testid="connector-sign-in"
              onClick={() => void act(() => core.signInToConnector(id))}
              className={smallButton}
            >
              {signIn.expired ? t("remoteConnectors.signInAgain") : t("remoteConnectors.signIn")}
            </button>
            {!clientId && (
              <button
                type="button"
                onClick={() => setEditingClient(true)}
                className="underline hover:text-amber-700"
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
    <div className="mt-1 text-xs text-gray-500">
      {signIn.signedIn && (
        <p className="flex flex-wrap items-center gap-2">
          <span>{t("remoteConnectors.signedIn")}</span>
          <button
            type="button"
            data-testid="connector-sign-out"
            onClick={() => void act(() => core.signOutOfConnector(id))}
            className="underline hover:text-gray-800"
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
      className="mt-1.5 flex flex-col gap-2"
    >
      <p className="text-xs text-amber-900">{t("remoteConnectors.client.hint")}</p>
      <label className="text-xs text-gray-700">
        {t("remoteConnectors.client.id")}
        <input
          required
          value={clientId}
          onChange={(event) => setClientId(event.target.value)}
          spellCheck={false}
          autoComplete="off"
          className={`${inputClass} bg-white font-mono`}
        />
      </label>
      <label className="text-xs text-gray-700">
        {t("remoteConnectors.client.secret")}
        <input
          type="password"
          value={clientSecret}
          onChange={(event) => setClientSecret(event.target.value)}
          autoComplete="off"
          className={`${inputClass} bg-white font-mono`}
        />
        <span className="mt-1 block text-gray-500">{t("remoteConnectors.client.secretHint")}</span>
      </label>
      <div className="flex flex-wrap gap-2 text-xs">
        <button type="submit" disabled={busy || !clientId.trim()} className={smallButton}>
          {t("remoteConnectors.client.save")}
        </button>
        {cancellable && (
          <button type="button" onClick={onDone} className={smallButton}>
            {t("remoteConnectors.cancel")}
          </button>
        )}
      </div>
      {error && (
        <p role="alert" className="text-xs text-red-700">
          {error}
        </p>
      )}
    </form>
  );
}
