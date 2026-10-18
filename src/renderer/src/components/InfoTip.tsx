import { InfoLineIcon } from "./lineIcons";

/**
 * What would be a helper line under a control, as a tooltip beside its label
 * (DESIGN.md, Words on screen): the ⓘ says there is one, and the words are
 * its accessible name too.
 */
export function InfoTip({ text, id, testId }: { text: string; id?: string; testId?: string }) {
  return (
    <span
      id={id}
      role="img"
      aria-label={text}
      title={text}
      data-testid={testId}
      className="ml-1.5 inline-flex shrink-0 cursor-default align-[-2px] text-ink-placeholder hover:text-ink-secondary"
    >
      <InfoLineIcon className="size-3.5" />
    </span>
  );
}
