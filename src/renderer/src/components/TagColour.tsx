import { useId } from "react";
import { isTagColour, TAG_COLOURS, type TagColour } from "../../../shared/tagColours";
import { useT } from "../i18n";
import { CheckLineIcon } from "./lineIcons";

/** A chip's fill and text in a Tag's colour. Literal classes, so Tailwind finds them. */
const CHIP: Record<TagColour, string> = {
  stone: "bg-tag-stone text-tag-stone-ink",
  taupe: "bg-tag-taupe text-tag-taupe-ink",
  brick: "bg-tag-brick text-tag-brick-ink",
  rose: "bg-tag-rose text-tag-rose-ink",
  orchid: "bg-tag-orchid text-tag-orchid-ink",
  violet: "bg-tag-violet text-tag-violet-ink",
  indigo: "bg-tag-indigo text-tag-indigo-ink",
  petrol: "bg-tag-petrol text-tag-petrol-ink",
};

/** A dot's colour: the brighter tone, as Finder's. */
const DOT: Record<TagColour, string> = {
  stone: "bg-tag-stone-dot",
  taupe: "bg-tag-taupe-dot",
  brick: "bg-tag-brick-dot",
  rose: "bg-tag-rose-dot",
  orchid: "bg-tag-orchid-dot",
  violet: "bg-tag-violet-dot",
  indigo: "bg-tag-indigo-dot",
  petrol: "bg-tag-petrol-dot",
};

const known = (colour: string | undefined): TagColour => (isTagColour(colour) ? colour : "stone");

/** The classes that colour a Tag's chip. */
export const tagChipColour = (colour: string | undefined) => CHIP[known(colour)];

/**
 * A Tag as a solid round dot in its colour, as Finder shows one where there
 * is no room for its name: 10px on a row, in a menu or the picker; 11px in
 * the sidebar's Tags view. "Needs review" is a hollow ring (`ReviewRing`), so
 * it never reads as a Tag.
 */
export function TagDot({
  colour,
  size = "sm",
  className = "",
}: {
  colour: string | undefined;
  size?: "sm" | "md";
  className?: string;
}) {
  return (
    <span
      aria-hidden="true"
      data-testid="tag-dot"
      data-colour={known(colour)}
      className={`inline-block shrink-0 rounded-full ${DOT[known(colour)]} ${
        size === "md" ? "size-[11px]" : "size-2.5"
      } ${className}`}
    />
  );
}

/**
 * A Tag awaiting the User's review, where Tags are dots (the sidebar, filter
 * menus, picker options): a hollow `attention` ring, 9px with a 1.5px stroke,
 * in a dot's 10px place. Inside a named chip it is the 6px amber dot (`ReviewDot`).
 */
export function ReviewRing() {
  return (
    <span
      aria-hidden="true"
      data-testid="review-ring"
      className="flex size-2.5 shrink-0 items-center justify-center"
    >
      <span className="size-[9px] rounded-full border-[1.5px] border-attention" />
    </span>
  );
}

/** Choosing a Tag's colour: a row of its dots, one choice, arrow keys between them. */
export function TagColourPicker({
  value,
  onChange,
}: {
  value: TagColour;
  onChange(colour: TagColour): void;
}) {
  const t = useT();
  const name = useId();
  return (
    <fieldset data-testid="tag-colour-picker" className="flex flex-col gap-1">
      <legend className="mb-1 text-[13px] leading-5 text-ink-secondary">
        {t("tags.colour.label")}
      </legend>
      <div className="flex flex-wrap gap-1.5">
        {TAG_COLOURS.map((colour) => (
          <label
            key={colour}
            title={t(`tags.colour.${colour}`)}
            className="relative cursor-pointer"
          >
            <input
              type="radio"
              name={name}
              value={colour}
              checked={value === colour}
              aria-label={t(`tags.colour.${colour}`)}
              data-testid="tag-colour"
              data-colour={colour}
              onChange={() => onChange(colour)}
              className="peer sr-only"
            />
            <span
              className={`flex size-6 items-center justify-center rounded-full text-sheet ${DOT[colour]} peer-checked:shadow-[0_0_0_2px_var(--color-sheet),0_0_0_3px_var(--color-ink-meta)] peer-focus-visible:outline-2 peer-focus-visible:outline-offset-2 peer-focus-visible:outline-accent`}
            >
              {value === colour && <CheckLineIcon className="size-3.5" />}
            </span>
          </label>
        ))}
      </div>
    </fieldset>
  );
}
