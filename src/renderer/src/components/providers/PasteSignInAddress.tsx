import { type KeyboardEvent, useEffect, useId, useRef, useState } from "react";
import type { PastedAddressProblem } from "../../../../core/api";
import { core } from "../../core";
import { useGap, useT } from "../../i18n";
import { useAppStore } from "../../store";
import { ProblemMark } from "../ProblemLine";
import { buttonStyle, errorTextClass, fieldClass } from "../ui";

/**
 * "Didn't come back? Paste the address the browser shows", under a ChatGPT
 * sign-in that waits for the browser: when the browser can't hand the
 * sign-in back (a blocked port, another browser profile, a proxy), the
 * address it ended on finishes the sign-in. It opens in place; Esc closes it.
 */
export function PasteSignInAddress() {
  const t = useT();
  const gap = useGap();
  const id = useId();
  const [open, setOpen] = useState(false);
  const [address, setAddress] = useState("");
  const [problem, setProblem] = useState<PastedAddressProblem | null>(null);
  const [sending, setSending] = useState(false);
  const field = useRef<HTMLInputElement>(null);
  const opener = useRef<HTMLButtonElement>(null);
  const closedWithEsc = useRef(false);

  useEffect(() => {
    if (open) field.current?.focus();
    else if (closedWithEsc.current) {
      closedWithEsc.current = false;
      opener.current?.focus();
    }
  }, [open]);

  const question = t("chatgpt.paste.question");
  const action = t("chatgpt.paste.open");
  const between = gap(question, action);

  if (!open) {
    return (
      <p className="text-[13px] leading-5 text-ink-secondary">
        {question}
        {between}
        <button
          ref={opener}
          type="button"
          data-testid="chatgpt-paste-open"
          onClick={() => setOpen(true)}
          className="font-semibold text-accent underline-offset-2 hover:underline"
        >
          {action}
        </button>
      </p>
    );
  }

  const empty = address.trim() === "";

  const submit = async () => {
    if (sending || empty) return;
    setSending(true);
    setProblem(null);
    try {
      const pasted = await core.pasteChatGptSignInAddress(address);
      // Taken: the sign-in finishes, and this line goes with the wait it belongs to.
      if (pasted.ok) return;
      setProblem(pasted.problem);
    } catch (error) {
      useAppStore.getState().reportError(error);
    }
    setSending(false);
    field.current?.focus();
  };

  const onKeyDown = (event: KeyboardEvent<HTMLInputElement>) => {
    if (event.nativeEvent.isComposing) return;
    if (event.key === "Enter") {
      event.preventDefault();
      void submit();
    } else if (event.key === "Escape") {
      // Closes the field only, not the Settings dialog or menu it sits in.
      event.preventDefault();
      event.stopPropagation();
      closedWithEsc.current = true;
      setOpen(false);
      setAddress("");
      setProblem(null);
    }
  };

  return (
    <div data-testid="chatgpt-paste" className="flex w-full flex-col gap-1">
      <label htmlFor={id} className="text-[13px] leading-5 text-ink-secondary">
        {question}
        {between}
        {action}
      </label>
      <div className="flex w-full items-center gap-2">
        <input
          ref={field}
          id={id}
          type="text"
          data-testid="chatgpt-paste-address"
          value={address}
          readOnly={sending}
          placeholder="http://127.0.0.1:…"
          spellCheck={false}
          autoComplete="off"
          aria-invalid={problem ? true : undefined}
          aria-describedby={problem ? `${id}-problem` : undefined}
          onChange={(event) => {
            setAddress(event.target.value);
            setProblem(null);
          }}
          onKeyDown={onKeyDown}
          className={`${fieldClass} h-7 min-w-0 flex-1 py-0 font-mono text-[13px]`}
        />
        <button
          type="button"
          data-testid="chatgpt-paste-continue"
          disabled={sending || empty}
          onClick={() => void submit()}
          className={buttonStyle("secondary", "sm")}
        >
          {t("chatgpt.paste.continue")}
        </button>
      </div>
      {problem && (
        <p
          id={`${id}-problem`}
          role="alert"
          data-testid="chatgpt-paste-problem"
          className={errorTextClass}
        >
          <ProblemMark />
          {t(`chatgpt.paste.problem.${problem}`)}
        </p>
      )}
    </div>
  );
}
