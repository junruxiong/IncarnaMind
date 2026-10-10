import { type KeyboardEvent, useId, useRef, useState } from "react";
import type { Mind } from "../../../core/api";
import { SearchIcon } from "../components/icons";
import { MindLineIcon, PlusLineIcon } from "../components/lineIcons";
import { useLanguage, useT } from "../i18n";
import { useAppStore } from "../store";
import { Popover } from "./Popover";

/** Where "Add to note ▾" puts an Answer. */
export type AddTarget =
  | { kind: "here" }
  | { kind: "mind"; mind: Mind }
  /** A new Mind, with this title. */
  | { kind: "new"; title: string };

/** Under "Recent", this many Minds; a search shows up to this many more. */
const RECENT = 3;
const FOUND = 8;

interface Option {
  id: string;
  target: AddTarget;
  title: string;
  detail: string;
  icon: "mind" | "new";
}

/**
 * The ▾ beside "Add to note" (board Chat-C2): "Find a Mind…", then This Mind
 * (at the cursor, Enter), the most recently edited Minds, and "New Mind…".
 * Typing finds any Mind by its title, and a title no Mind has names the new
 * one. ↑↓ move, Enter adds, Esc closes.
 */
export function AddToMindPicker({
  anchor,
  mindId,
  newTitle,
  hereAt,
  onChoose,
  onClose,
}: {
  anchor: HTMLElement;
  /** The Mind the thread is in. */
  mindId: string;
  /** What a new Mind is called when nothing is typed: the chat's first Question. */
  newTitle: string;
  /** Where in this Mind it goes: at the cursor, or at the end when the cursor isn't in it. */
  hereAt: "cursor" | "end";
  onChoose(target: AddTarget): void;
  onClose(): void;
}) {
  const t = useT();
  const language = useLanguage();
  const minds = useAppStore((state) => state.minds);
  const tabs = useAppStore((state) => state.tabs);
  const [query, setQuery] = useState("");
  const [active, setActive] = useState(0);
  const field = useRef<HTMLInputElement>(null);
  const listId = useId();

  const here = minds.find((mind) => mind.id === mindId);
  const others = minds.filter((mind) => mind.id !== mindId && mind.kind === "mind");
  const typed = query.trim();
  const matches = (mind: Mind) =>
    (mind.title || t("mind.untitled")).toLocaleLowerCase().includes(typed.toLocaleLowerCase());
  const shown = typed ? others.filter(matches).slice(0, FOUND) : others.slice(0, RECENT);
  const edited = (mind: Mind) =>
    tabs.includes(mind.id)
      ? t("thread.pick.openInTab")
      : t("thread.pick.edited", {
          date: new Date(mind.updatedAt).toLocaleDateString(language, {
            day: "numeric",
            month: "short",
          }),
        });

  const options: Option[] = [];
  if (!typed || (here && matches(here))) {
    options.push({
      id: `${listId}-here`,
      target: { kind: "here" },
      title: t("thread.pick.here"),
      detail: t(hereAt === "cursor" ? "thread.pick.hereCursor" : "thread.pick.hereEnd", {
        title: here?.title || t("mind.untitled"),
      }),
      icon: "mind",
    });
  }
  const recent = options.length;
  for (const mind of shown) {
    options.push({
      id: `${listId}-${mind.id}`,
      target: { kind: "mind", mind },
      title: mind.title || t("mind.untitled"),
      detail: edited(mind),
      icon: "mind",
    });
  }
  const fresh = typed || newTitle;
  options.push({
    id: `${listId}-new`,
    target: { kind: "new", title: fresh },
    title: typed ? t("thread.pick.newNamed", { title: typed }) : t("thread.pick.new"),
    detail: t("thread.pick.newDetail"),
    icon: "new",
  });
  const current = Math.min(active, options.length - 1);

  const onKeyDown = (event: KeyboardEvent<HTMLInputElement>) => {
    if (event.nativeEvent.isComposing) return;
    if (event.key === "ArrowDown" || event.key === "ArrowUp") {
      event.preventDefault();
      const step = event.key === "ArrowDown" ? 1 : -1;
      setActive((current + step + options.length) % options.length);
    } else if (event.key === "Enter") {
      event.preventDefault();
      const option = options[current];
      if (option) onChoose(option.target);
    }
  };

  const row = (option: Option, index: number) => (
    // biome-ignore lint/a11y/useKeyWithClickEvents: the search field owns the keys (aria-activedescendant).
    <div
      key={option.id}
      id={option.id}
      role="option"
      tabIndex={-1}
      aria-selected={index === current}
      data-testid="add-to-option"
      data-target={option.target.kind}
      onMouseMove={() => setActive(index)}
      onMouseDown={(event) => event.preventDefault()}
      onClick={() => onChoose(option.target)}
      className="add-to-option"
    >
      <span className="add-to-option-icon">
        {option.icon === "new" ? (
          <PlusLineIcon className="size-4" />
        ) : (
          <MindLineIcon className="size-4" />
        )}
      </span>
      <span className="add-to-option-text">
        <span className="add-to-option-title" title={option.title}>
          {option.title}
        </span>
        <span className="add-to-option-detail">{option.detail}</span>
      </span>
      {index === current && option.target.kind === "here" && (
        <span aria-hidden="true" className="add-to-option-key">
          {t("thread.pick.enter")}
        </span>
      )}
    </div>
  );

  return (
    <Popover
      anchor={anchor}
      placement="top-start"
      onDismiss={onClose}
      initialFocus={field}
      aria-label={t("thread.pick.label")}
      data-testid="add-to-picker"
      className="add-to-picker"
    >
      <div className="add-to-search">
        <label className="add-to-field">
          <SearchIcon className="size-3.5 shrink-0" />
          <span className="sr-only">{t("thread.pick.find")}</span>
          <input
            ref={field}
            type="text"
            role="combobox"
            aria-expanded="true"
            aria-controls={listId}
            aria-activedescendant={options[current]?.id}
            aria-autocomplete="list"
            data-testid="add-to-search"
            placeholder={t("thread.pick.find")}
            value={query}
            onChange={(event) => {
              setQuery(event.target.value);
              setActive(0);
            }}
            onKeyDown={onKeyDown}
          />
        </label>
      </div>
      <div id={listId} role="listbox" aria-label={t("thread.pick.minds")} className="add-to-list">
        {options.slice(0, recent).map(row)}
        {shown.length > 0 && (
          // biome-ignore lint/a11y/useSemanticElements: a group of a listbox's options, which a fieldset can't hold.
          <div role="group" aria-label={typed ? t("thread.pick.found") : t("thread.pick.recent")}>
            <p aria-hidden="true" className="add-to-heading">
              {typed ? t("thread.pick.found") : t("thread.pick.recent")}
            </p>
            {options
              .slice(recent, recent + shown.length)
              .map((option, index) => row(option, recent + index))}
          </div>
        )}
        {typed && shown.length === 0 && recent === 0 && (
          <p className="add-to-empty">{t("thread.pick.none")}</p>
        )}
        {/* biome-ignore lint/a11y/useSemanticElements: a rule between a listbox's options, where an hr can't go. */}
        <div role="separator" className="add-to-separator" />
        {row(options[options.length - 1] as Option, options.length - 1)}
      </div>
    </Popover>
  );
}
