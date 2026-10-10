import { type Ref, useEffect, useImperativeHandle, useRef, useState } from "react";
import type { Skill } from "../../../core/api";
import { SkillIcon } from "../components/icons";
import { useT } from "../i18n";
import type { ChoiceListHandle } from "./ScopePicker";
import { matchSlashItems, skillSlashItems } from "./slashItems";

/**
 * The Skills "/" can choose in the composer for the next Question, filtered
 * by what follows the slash as the slash menu filters them. The first is
 * chosen; arrows move and Enter or Tab chooses (see `ChoiceListHandle`).
 */
export function SkillChoiceList({
  skills,
  describe,
  query,
  onChoose,
  id,
  onSelect,
  ref,
}: {
  /** The enabled Skills. */
  skills: readonly Skill[];
  describe(skill: Skill): string;
  query: string;
  onChoose(name: string): void;
  /** The listbox's id, and its options' (`${id}-${index}`). */
  id: string;
  /** Hears which option is chosen, by its index (-1: none). */
  onSelect(index: number): void;
  ref?: Ref<ChoiceListHandle>;
}) {
  const t = useT();
  const items = skillSlashItems(skills, describe);
  const shown = matchSlashItems(items, query, (item) =>
    typeof item.label === "string" ? t(item.label) : item.label.text,
  );
  // The choice belongs to a query: typing more starts again from the top.
  const [choice, setChoice] = useState({ query, index: 0 });
  const selected = choice.query === query ? Math.min(choice.index, shown.length - 1) : 0;
  const setSelected = (index: number) => setChoice({ query, index });
  const options = useRef<(HTMLButtonElement | null)[]>([]);
  const reportSelected = useRef(onSelect);
  reportSelected.current = onSelect;
  const nameOf = (index: number) => {
    const label = shown[index]?.label;
    return label && typeof label !== "string" ? label.text : null;
  };

  useEffect(() => {
    options.current[selected]?.scrollIntoView({ block: "nearest" });
    reportSelected.current(selected);
  }, [selected]);

  useImperativeHandle(ref, () => ({
    onKeyDown(event) {
      if (event.isComposing || shown.length === 0) return false;
      if (event.key === "ArrowDown" || event.key === "ArrowUp") {
        const step = event.key === "ArrowDown" ? 1 : -1;
        setSelected((selected + step + shown.length) % shown.length);
        return true;
      }
      if (event.key === "Enter" || event.key === "Tab") {
        const name = nameOf(selected);
        if (name) onChoose(name);
        return true;
      }
      return false;
    },
  }));

  return (
    <div
      id={id}
      role="listbox"
      data-testid="composer-skill-picker"
      aria-label={t("composer.skills.label")}
      className="editor-menu max-h-[320px] w-[260px] overflow-y-auto"
    >
      {shown.length === 0 && <p className="editor-menu-empty">{t("editor.slash.empty")}</p>}
      {shown.map((item, index) => (
        <button
          key={item.id}
          ref={(element) => {
            options.current[index] = element;
          }}
          id={`${id}-${index}`}
          type="button"
          role="option"
          aria-selected={index === selected}
          data-testid={`slash-item-${item.id}`}
          // The composer keeps the focus.
          onMouseDown={(event) => event.preventDefault()}
          onMouseEnter={() => setSelected(index)}
          onClick={() => {
            const name = nameOf(index);
            if (name) onChoose(name);
          }}
          title={item.hint}
          className="editor-menu-item"
        >
          <span className="editor-menu-icon">
            <SkillIcon className="size-3.5" />
          </span>
          <span className="truncate">{nameOf(index)}</span>
        </button>
      ))}
    </div>
  );
}
