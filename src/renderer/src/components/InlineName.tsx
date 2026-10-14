import { useLayoutEffect, useRef, useState } from "react";
import { nameKeyAction } from "../nameKeys";
import { useAppStore } from "../store";
import { rowInputClass } from "./sidebarRows";

/**
 * A name typed where it is shown (docs/agents/interaction.md, rule 1), for
 * every object that has one. Everything is selected at the start. Enter or
 * leaving the field saves; Esc cancels and keeps the old name; a blank or
 * unchanged name changes nothing and counts as a cancel. Enter while an input
 * method is composing only confirms the composition.
 */
export function InlineName({
  value,
  label,
  placeholder,
  maxLength,
  testId,
  className = rowInputClass,
  onCommit,
  onCancel,
}: {
  value: string;
  /** What a screen reader calls the field: "Rename Board minutes". */
  label: string;
  placeholder?: string;
  maxLength?: number;
  testId?: string;
  className?: string;
  /** The new name, trimmed; its failure is reported as a problem. */
  onCommit(name: string): void | Promise<void>;
  onCancel(): void;
}) {
  const [draft, setDraft] = useState(value);
  const field = useRef<HTMLInputElement>(null);
  const finished = useRef(false);

  useLayoutEffect(() => {
    field.current?.focus();
    field.current?.select();
  }, []);

  const finish = (save: boolean) => {
    if (finished.current) return;
    finished.current = true;
    const name = draft.trim();
    if (!save || name === "" || name === value) {
      onCancel();
      return;
    }
    Promise.resolve(onCommit(name)).catch(useAppStore.getState().reportError);
  };

  return (
    <input
      ref={field}
      value={draft}
      aria-label={label}
      placeholder={placeholder}
      maxLength={maxLength}
      data-testid={testId}
      onChange={(event) => setDraft(event.target.value)}
      onKeyDown={(event) => {
        const action = nameKeyAction({
          key: event.key,
          keyCode: event.keyCode,
          isComposing: event.nativeEvent.isComposing,
        });
        if (!action) return;
        event.preventDefault();
        // Esc cancels here, and closes nothing else.
        event.stopPropagation();
        finish(action === "save");
      }}
      onBlur={() => finish(true)}
      className={className}
    />
  );
}
