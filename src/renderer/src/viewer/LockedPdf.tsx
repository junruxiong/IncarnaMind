import { type FormEvent, useEffect, useRef, useState } from "react";
import type { SecretStorageStatus } from "../../../core/api";
import { SecretStorageNotice } from "../components/providers/ProviderForm";
import { buttonStyle, errorTextClass, fieldClass } from "../components/ui";
import { core } from "../core";
import { errorMessage } from "../errors";
import { useT } from "../i18n";

/**
 * A password-protected PDF the viewer can't open yet (#763): its password is
 * asked for in place. Unlocking keeps it, by the core, and the PDF opens; a
 * wrong one says so under the field, which keeps the text, selected, to try
 * again. The field is uncontrolled, so the password stays out of React state.
 * The password is kept with the system's keyring: on Linux with none running,
 * the User first accepts keeping it without one, as for an API key.
 */
export function LockedPdf({ documentId, onUnlocked }: { documentId: string; onUnlocked(): void }) {
  const t = useT();
  const field = useRef<HTMLInputElement>(null);
  const [wrong, setWrong] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [secretStorage, setSecretStorage] = useState<SecretStorageStatus | null>(null);
  const blocked = secretStorage !== null && !secretStorage.canSave;

  useEffect(() => field.current?.focus(), []);
  useEffect(() => {
    core.getSecretStorage().then(setSecretStorage, (failure: unknown) => {
      setError(errorMessage(failure, t));
    });
  }, [t]);

  const acceptPlainText = async () => {
    try {
      setSecretStorage(await core.acceptPlainTextSecretStorage());
      field.current?.focus();
    } catch (failure) {
      setError(errorMessage(failure, t));
    }
  };

  const unlock = async (event: FormEvent) => {
    event.preventDefault();
    const input = field.current;
    if (!input || input.value === "" || busy || blocked) return;
    setBusy(true);
    setError(null);
    try {
      const result = await core.unlockDocument({ documentId, password: input.value });
      if (result.kind === "wrong-password") {
        setWrong(true);
        input.focus();
        input.select();
        return;
      }
      input.value = "";
      onUnlocked();
    } catch (failure) {
      setError(errorMessage(failure, t));
    } finally {
      setBusy(false);
    }
  };

  return (
    <form
      data-testid="pdf-locked"
      onSubmit={(event) => void unlock(event)}
      className="mx-auto flex w-full max-w-sm flex-col gap-3 px-6 py-10 text-ui"
    >
      <h2 className="font-semibold text-ink">{t("viewer.locked.title")}</h2>
      <p className="text-ink-secondary">{t("viewer.locked.body")}</p>
      <div className="flex gap-2">
        <input
          ref={field}
          type="password"
          data-testid="pdf-password"
          aria-label={t("viewer.locked.password")}
          placeholder={t("viewer.locked.password")}
          autoComplete="off"
          aria-invalid={wrong || undefined}
          aria-describedby={wrong ? "pdf-password-wrong" : undefined}
          onInput={() => {
            setWrong(false);
            setError(null);
          }}
          className={`${fieldClass} min-w-0 flex-1`}
        />
        <button
          type="submit"
          data-testid="pdf-unlock"
          disabled={busy || blocked}
          className={buttonStyle("primary")}
        >
          {t("viewer.locked.unlock")}
        </button>
      </div>
      {wrong && (
        <p
          id="pdf-password-wrong"
          role="alert"
          data-testid="pdf-password-wrong"
          className="text-ink-meta"
        >
          {t("viewer.locked.wrong")}
        </p>
      )}
      {error !== null && (
        <p role="alert" data-testid="pdf-unlock-error" className={errorTextClass}>
          {error}
        </p>
      )}
      {secretStorage && blocked && (
        <SecretStorageNotice status={secretStorage} onAccept={() => void acceptPlainText()} />
      )}
    </form>
  );
}
