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

/** A swatch: the fill, edged in its text colour. */
const SWATCH: Record<TagColour, string> = {
  stone: "bg-tag-stone border-tag-stone-ink/40",
  taupe: "bg-tag-taupe border-tag-taupe-ink/40",
  brick: "bg-tag-brick border-tag-brick-ink/40",
  rose: "bg-tag-rose border-tag-rose-ink/40",
  orchid: "bg-tag-orchid border-tag-orchid-ink/40",
  violet: "bg-tag-violet border-tag-violet-ink/40",
  indigo: "bg-tag-indigo border-tag-indigo-ink/40",
  petrol: "bg-tag-petrol border-tag-petrol-ink/40",
};

const known = (colour: string | undefined): TagColour => (isTagColour(colour) ? colour : "stone");

/** The classes that colour a Tag's chip. */
export const tagChipColour = (colour: string | undefined) => CHIP[known(colour)];

/**
 * A Tag's colour as a small square, beside its name in a list. Square, so it
 * never reads as the round amber "needs review" dot.
 */
export function TagSwatch({
  colour,
  size = "sm",
}: {
  colour: string | undefined;
  size?: "sm" | "md";
}) {
  return (
    <span
      aria-hidden="true"
      data-testid="tag-swatch"
      data-colour={known(colour)}
      className={`inline-block shrink-0 border ${SWATCH[known(colour)]} ${
        size === "md" ? "size-3 rounded-[3px]" : "size-2.5 rounded-[2px]"
      }`}
    />
  );
}

/** Choosing a Tag's colour: a row of swatches, one choice, arrow keys between them. */
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
              className={`flex size-6 items-center justify-center rounded-sm border ${SWATCH[colour]} ${CHIP[colour]} peer-checked:shadow-[0_0_0_2px_var(--color-sheet),0_0_0_3px_var(--color-ink-meta)] peer-focus-visible:outline-2 peer-focus-visible:outline-offset-2 peer-focus-visible:outline-accent`}
            >
              {value === colour && <CheckLineIcon className="size-3.5" />}
            </span>
          </label>
        ))}
      </div>
    </fieldset>
  );
}
