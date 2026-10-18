import { useEffect, useRef } from "react";
import { ChevronLeftLineIcon } from "./lineIcons";

/**
 * A Settings sub-page's way up to its page, named by that page ("‹ Connectors"), as interaction
 * rule 3 asks. The page opened from a card or a row, so the keyboard's
 * place starts here, unless the page puts it where the User acts first (`focused` false).
 */
export function SettingsBackLink({
  label,
  testId,
  onBack,
  focused = true,
}: {
  label: string;
  testId: string;
  onBack(): void;
  focused?: boolean;
}) {
  const button = useRef<HTMLButtonElement>(null);
  // biome-ignore lint/correctness/useExhaustiveDependencies: once, as the page opens.
  useEffect(() => {
    if (focused) button.current?.focus();
  }, []);
  return (
    <button
      ref={button}
      type="button"
      data-testid={testId}
      onClick={onBack}
      className="inline-flex items-center gap-1 self-start rounded-sm text-[12px] leading-[18px] text-ink-meta hover:text-ink"
    >
      <ChevronLeftLineIcon className="size-3" />
      {label}
    </button>
  );
}
