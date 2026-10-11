import { Fragment, type Ref, useEffect, useImperativeHandle, useRef, useState } from "react";
import type { Skill } from "../../../core/api";
import { menuTitleClass } from "../components/ui";
import { useT } from "../i18n";
import type { ChoiceListHandle } from "./ScopePicker";
import { matchSlashItems, type SlashItem, skillSlashItems } from "./slashItems";

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
  onManage,
  id,
  onSelect,
  ref,
}: {
  /** The enabled Skills. */
  skills: readonly Skill[];
  describe(skill: Skill): string;
  query: string;
  onChoose(name: string): void;
  /** "Manage Skills…": Settings → Skills. */
  onManage(): void;
  /** The listbox's id, and its options' (`${id}-${index}`). */
  id: string;
  /** Hears which option is chosen, by its index (-1: none). */
  onSelect(index: number): void;
  ref?: Ref<ChoiceListHandle>;
}) {
  const t = useT();
  const items = skillSlashItems(skills, describe);
  const isBuiltIn = (item: SlashItem) =>
    skills.some((skill) => `skill-${skill.name}` === item.id && skill.builtIn);
  // The User's own first, then the built-in ones, as Settings lists them.
  const shown = matchSlashItems(items, query, (item) =>
    typeof item.label === "string" ? t(item.label) : item.label.text,
  ).sort((a, b) => Number(isBuiltIn(a)) - Number(isBuiltIn(b)));
  const grouped = shown.some(isBuiltIn) && shown.some((item) => !isBuiltIn(item));
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
      className="editor-menu skill-choices"
    >
      {shown.length === 0 && <p className="editor-menu-empty">{t("editor.slash.empty")}</p>}
      {shown.map((item, index) => (
        <Fragment key={item.id}>
          {grouped &&
            (index === 0 || isBuiltIn(item) !== isBuiltIn(shown[index - 1] as SlashItem)) && (
              <p className={menuTitleClass}>
                {t(isBuiltIn(item) ? "skills.settings.builtIn" : "skills.settings.yours")}
              </p>
            )}
          <button
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
            className="editor-menu-item skill-choice"
          >
            <span className="truncate">{nameOf(index)}</span>
            <span className="skill-choice-description truncate">{item.hint}</span>
          </button>
        </Fragment>
      ))}
      <button
        type="button"
        data-testid="composer-skills-manage"
        // The composer keeps the focus until the Settings page opens.
        onMouseDown={(event) => event.preventDefault()}
        onClick={onManage}
        className="editor-menu-item skill-choice-manage"
      >
        {t("composer.skills.manage")}
      </button>
    </div>
  );
}
