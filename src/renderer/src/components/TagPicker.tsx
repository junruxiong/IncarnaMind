import {
  type KeyboardEvent,
  type ReactNode,
  type ToggleEvent,
  useEffect,
  useId,
  useMemo,
  useRef,
  useState,
} from "react";
import { useShallow } from "zustand/react/shallow";
import type { Document, DocumentTag, Tag } from "../../../core/api";
import { errorMessage } from "../errors";
import { useT } from "../i18n";
import { useAppStore } from "../store";
import { actionFor, pickerOptions, type TagOnDocuments, tagsOnDocuments } from "../tagEditing";
import { CheckLineIcon, CloseLineIcon, PlusLineIcon, SparkLineIcon } from "./lineIcons";
import { buttonStyle, errorTextClass, hintClass, inputClass, menuRuleClass } from "./ui";

const GAP_PX = 4;
const EDGE_PX = 8;

/** Below its button, or above it if the window is too short; kept inside the window. */
function place(button: HTMLElement | null, popover: HTMLElement | null): void {
  const anchor = button?.getBoundingClientRect();
  if (!anchor || !popover) return;
  popover.style.left = `${Math.max(EDGE_PX, Math.min(anchor.left, window.innerWidth - popover.offsetWidth - EDGE_PX))}px`;
  const below = anchor.bottom + GAP_PX;
  const height = popover.offsetHeight;
  popover.style.top =
    below + height > window.innerHeight - EDGE_PX
      ? `${Math.max(EDGE_PX, anchor.top - GAP_PX - height)}px`
      : `${below}px`;
}

/** Focuses the picker's field (or the description field while describing a new Tag). */
const focusFieldIn = (popover: HTMLElement | null) =>
  popover?.querySelector<HTMLElement>("[data-picker-field]")?.focus();

/**
 * A popover that opens from a button, like `usePopoverMenu`, for the Tag
 * picker: it takes the focus into its field, Esc closes only it, and a
 * click outside closes it.
 */
export function useTagPopover() {
  const id = useId();
  const button = useRef<HTMLButtonElement>(null);
  const popover = useRef<HTMLDivElement>(null);
  const [open, setOpen] = useState(false);

  useEffect(() => {
    if (!open || !popover.current?.matches(":popover-open")) return;
    place(button.current, popover.current);
    focusFieldIn(popover.current);
  }, [open]);

  return {
    open,
    close() {
      popover.current?.hidePopover();
      button.current?.focus();
    },
    /** Places it again, e.g. after its height changed. */
    reposition: () => place(button.current, popover.current),
    buttonProps: {
      ref: button,
      popoverTarget: id,
      "aria-haspopup": "dialog" as const,
      "aria-expanded": open,
    },
    popoverProps: {
      ref: popover,
      id,
      popover: "auto" as const,
      onBeforeToggle: (event: ToggleEvent<HTMLDivElement>) => {
        if (event.newState === "open") place(button.current, popover.current);
        setOpen(event.newState === "open");
      },
      onToggle: (event: ToggleEvent<HTMLDivElement>) => {
        if (event.newState !== "open") return;
        place(button.current, popover.current);
        focusFieldIn(popover.current);
      },
      onKeyDown: (event: KeyboardEvent) => {
        if (event.key !== "Escape" || event.defaultPrevented) return;
        // Here, so the Document viewer's Esc doesn't close it too.
        event.preventDefault();
        popover.current?.hidePopover();
        button.current?.focus();
      },
    },
  };
}

/** The popover's look: a menu's, a little wider, with room for the field. */
export const tagPopoverClass =
  "inset-auto m-0 w-80 max-w-[calc(100vw-16px)] overflow-visible rounded-lg border-0 bg-sheet p-0 text-ui font-normal text-ink shadow-popover";

/** A chip's look in the picker's field and on a row: 20px, radius 4, the chip fill. */
export const chipClass =
  "inline-flex h-5 max-w-[10rem] shrink-0 items-center gap-1 rounded-sm bg-chip px-1.5 text-[12px] leading-5 text-ink-secondary";

/**
 * A Tag's tooltip on one Document: who added it and, when automatic tagging
 * said, how likely it is, or that it awaits review.
 */
export function chipTitle(
  link: Pick<DocumentTag, "source" | "confidence" | "needsReview">,
  tag: Pick<Tag, "name">,
  t: ReturnType<typeof useT>,
): string {
  const percent = link.confidence === null ? null : Math.round(link.confidence * 100);
  if (link.needsReview)
    return percent === null
      ? t("tags.chip.reviewTitle", { tag: tag.name })
      : t("jev.chip.review", { tag: tag.name, percent });
  if (link.source === "user") return t("tags.chip.user", { tag: tag.name });
  return percent === null
    ? t("tags.chip.automatic", { tag: tag.name })
    : t("jev.chip.likely", { tag: tag.name, percent });
}

/** The tooltip of a Tag in the picker's field, on one Document. */
const stateTitle = (state: TagOnDocuments, t: ReturnType<typeof useT>) =>
  chipTitle(
    {
      source: state.automatic ? "automatic" : "user",
      confidence: state.confidence,
      needsReview: state.needsReview,
    },
    state.tag,
    t,
  );

/** The amber dot of a Tag awaiting the User's review (DESIGN.md: `attention`). */
export function ReviewDot() {
  return <span aria-hidden="true" className="size-1.5 shrink-0 rounded-full bg-attention" />;
}

/**
 * Edits the Tags of one Document or several, keyboard first: type to filter,
 * ↑↓ to move, Enter to add (or, on one Document, to confirm a Tag awaiting
 * review, or to take off one it has), Backspace in the empty field to pick
 * the last Tag and again to take it off, × on a Tag to take it off. A name
 * no Tag has is created, optionally with a description (Shift+Enter), and
 * added. With several Documents, a Tag some of them have is added to all.
 */
export function TagPicker({
  documentIds,
  label,
  footer,
  onReposition,
}: {
  documentIds: readonly string[];
  /** Says what is edited, for screen readers. */
  label: string;
  /** More actions under the list (re-tag, manage). */
  footer?: ReactNode;
  /** The popover's height changed. */
  onReposition?(): void;
}) {
  const t = useT();
  const listId = useId();
  const tags = useAppStore((state) => state.tags);
  const documents = useAppStore(
    useShallow((state) =>
      documentIds.flatMap((id) => state.documents.find((each) => each.id === id) ?? []),
    ),
  );
  const [query, setQuery] = useState("");
  const [active, setActive] = useState(0);
  const [picked, setPicked] = useState<string | null>(null);
  const [describing, setDescribing] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const field = useRef<HTMLInputElement>(null);
  const states = useMemo(() => tagsOnDocuments(tags, documents), [tags, documents]);
  const { matches, create } = pickerOptions(states, query);
  const options: ({ kind: "tag"; state: TagOnDocuments } | { kind: "create"; name: string })[] = [
    ...matches.map((state) => ({ kind: "tag" as const, state })),
    ...(create ? [{ kind: "create" as const, name: create }] : []),
  ];
  const applied = states.filter((state) => state.coverage === "all");
  const ids = documents.map((document) => document.id);
  const single = ids.length === 1;
  const names = documents.map((document) => document.name).join(", ");
  const current = Math.min(active, options.length - 1);

  // The list's length changes the popover's height: place it again.
  // biome-ignore lint/correctness/useExhaustiveDependencies: placing follows the visible rows.
  useEffect(() => onReposition?.(), [options.length, describing, applied.length, error]);

  // Back from describing a new Tag (created or not): the field has the keys again.
  useEffect(() => {
    if (describing === null) field.current?.focus();
  }, [describing]);

  const run = async (work: () => Promise<unknown>) => {
    setError(null);
    try {
      await work();
    } catch (failure) {
      setError(errorMessage(failure));
    }
  };
  const store = useAppStore.getState;
  const add = (tagId: string) =>
    single
      ? store().addDocumentTag(ids[0] as string, tagId)
      : store().addTagToDocuments(ids, tagId);
  const remove = (tagId: string) =>
    single
      ? store().removeDocumentTag(ids[0] as string, tagId)
      : store().removeTagFromDocuments(ids, tagId);

  // The field empties at once, so keys typed while the change is saved go to the next Tag.
  const choose = (state: TagOnDocuments) => {
    setQuery("");
    setPicked(null);
    return run(async () => {
      if (actionFor(state) === "remove") await remove(state.tag.id);
      else await add(state.tag.id);
      field.current?.focus();
    });
  };
  const createTag = (name: string, description = "") => {
    setQuery("");
    setActive(0);
    return run(async () => {
      try {
        const tag = await store().createTag(name, description);
        await add(tag.id);
      } catch (failure) {
        // Not created (e.g. the name is taken): the name comes back to correct.
        setQuery(name);
        throw failure;
      }
      setDescribing(null);
      field.current?.focus();
    });
  };

  const onKeyDown = (event: KeyboardEvent<HTMLInputElement>) => {
    if (event.key === "ArrowDown" || event.key === "ArrowUp") {
      event.preventDefault();
      if (options.length === 0) return;
      const step = event.key === "ArrowDown" ? 1 : -1;
      setActive((current + step + options.length) % options.length);
      return;
    }
    if (event.key === "Enter") {
      event.preventDefault();
      const option = options[current];
      if (!option) return;
      if (option.kind === "create") {
        if (event.shiftKey) setDescribing(option.name);
        else void createTag(option.name);
      } else void choose(option.state);
      return;
    }
    if (event.key === "Backspace" && query === "") {
      const last = applied.at(-1);
      if (!last) return;
      event.preventDefault();
      if (picked === last.tag.id) {
        void run(async () => {
          await remove(last.tag.id);
          setPicked(null);
        });
      } else setPicked(last.tag.id);
      return;
    }
    if (event.key !== "Shift") setPicked(null);
  };

  if (describing !== null)
    return (
      <DescribeTag
        name={describing}
        error={error}
        onCreate={(description) => void createTag(describing, description)}
        onBack={() => {
          setDescribing(null);
          setError(null);
        }}
      />
    );

  const activeId = options[current] ? `${listId}-${current}` : undefined;
  return (
    <div data-testid="tag-picker" className="flex flex-col">
      {/* The Tags on every Document edited, then the field. */}
      {/* biome-ignore lint/a11y/noStaticElementInteractions lint/a11y/useKeyWithClickEvents: a click on the field's empty space focuses the field, which has the keys. */}
      <div
        className="flex max-h-28 min-h-10 flex-wrap items-center gap-1 overflow-y-auto border-b border-rule px-2 py-1.5"
        onClick={() => field.current?.focus()}
      >
        {applied.map((state) => (
          <span
            key={state.tag.id}
            data-testid="tag-token"
            data-tag-id={state.tag.id}
            data-source={state.automatic ? "automatic" : "user"}
            data-needs-review={state.needsReview && single ? "true" : undefined}
            title={single ? stateTitle(state, t) : state.tag.description || undefined}
            className={`${chipClass} pr-0.5 ${
              picked === state.tag.id ? "shadow-[inset_0_0_0_1px_var(--color-ink-meta)]" : ""
            }`}
          >
            {state.needsReview && single && <ReviewDot />}
            {state.automatic && !(state.needsReview && single) && (
              <SparkLineIcon aria-hidden="true" className="size-2.5 shrink-0 text-ink-meta" />
            )}
            <span className="truncate">{state.tag.name}</span>
            {state.needsReview && single && (
              <span className="sr-only">{t("jev.chip.needsReview")}</span>
            )}
            {state.needsReview && single && (
              <button
                type="button"
                tabIndex={-1}
                data-testid="confirm-document-tag"
                aria-label={t("jev.chip.confirm", { tag: state.tag.name, name: names })}
                title={t("tags.review.confirm", { tag: state.tag.name })}
                onMouseDown={(event) => event.preventDefault()}
                onClick={() => void run(() => add(state.tag.id))}
                className="inline-flex size-4 items-center justify-center rounded-sm text-ink-secondary hover:bg-hover hover:text-ink"
              >
                <CheckLineIcon className="size-3" />
              </button>
            )}
            <button
              type="button"
              tabIndex={-1}
              data-testid="tag-token-remove"
              aria-label={t("tags.picker.remove", { tag: state.tag.name })}
              title={t("tags.picker.remove", { tag: state.tag.name })}
              onMouseDown={(event) => event.preventDefault()}
              onClick={() => void run(() => remove(state.tag.id))}
              className="inline-flex size-4 items-center justify-center rounded-sm text-ink-meta hover:bg-hover hover:text-ink"
            >
              <CloseLineIcon className="size-2.5" />
            </button>
          </span>
        ))}
        <input
          ref={field}
          data-picker-field
          data-testid="tag-picker-input"
          role="combobox"
          aria-label={label}
          aria-expanded="true"
          aria-controls={listId}
          aria-autocomplete="list"
          aria-activedescendant={activeId}
          placeholder={applied.length ? "" : t("tags.picker.placeholder")}
          title={t("tags.picker.hint")}
          value={query}
          maxLength={100}
          onChange={(event) => {
            setQuery(event.target.value);
            setActive(0);
            setPicked(null);
          }}
          onKeyDown={onKeyDown}
          className="h-7 min-w-24 flex-1 bg-transparent text-ui text-ink outline-none placeholder:text-ink-placeholder"
        />
      </div>
      <div
        id={listId}
        role="listbox"
        aria-multiselectable="true"
        aria-label={label}
        data-testid="tag-picker-list"
        className="max-h-64 overflow-y-auto p-1"
      >
        {options.length === 0 && (
          <p className="px-2 py-1 text-[13px] text-ink-meta">{t("tags.picker.none")}</p>
        )}
        {options.map((option, index) =>
          option.kind === "create" ? (
            // biome-ignore lint/a11y/useKeyWithClickEvents: the field owns the keys (aria-activedescendant).
            <div
              key="create"
              id={`${listId}-${index}`}
              role="option"
              tabIndex={-1}
              aria-selected="false"
              data-testid="tag-option-create"
              data-active={index === current ? "true" : undefined}
              onMouseDown={(event) => event.preventDefault()}
              onMouseMove={() => setActive(index)}
              onClick={() => void createTag(option.name)}
              className={`flex h-7 cursor-default items-center gap-2 rounded-md pr-1 pl-2 ${
                index === current ? "bg-hover" : ""
              }`}
            >
              <PlusLineIcon className="size-3.5 shrink-0 text-ink-meta" />
              <span className="min-w-0 flex-1 truncate">
                {t("tags.picker.create", { name: option.name })}
              </span>
              <button
                type="button"
                tabIndex={-1}
                data-testid="tag-option-describe"
                onClick={(event) => {
                  event.stopPropagation();
                  setDescribing(option.name);
                }}
                className="shrink-0 rounded-sm px-1 text-[12px] text-ink-meta hover:bg-chip hover:text-ink"
              >
                {t("tags.picker.describe")}
              </button>
            </div>
          ) : (
            <TagOption
              key={option.state.tag.id}
              id={`${listId}-${index}`}
              state={option.state}
              single={single}
              active={index === current}
              onHover={() => setActive(index)}
              onChoose={() => void choose(option.state)}
            />
          ),
        )}
      </div>
      {error && (
        <p role="alert" className={`${errorTextClass} px-3 pb-2`}>
          {error}
        </p>
      )}
      <div className={menuRuleClass} />
      {/* The keys, unless the row has actions to offer instead; the field's own placeholder starts it. */}
      <div className="flex min-h-8 items-center justify-between gap-2 px-1 pb-1">
        {footer ? (
          <div className="flex min-w-0 items-center">{footer}</div>
        ) : (
          <span className="truncate px-2 text-[12px] text-ink-meta">{t("tags.picker.hint")}</span>
        )}
      </div>
    </div>
  );
}

/** One Tag in the list: a mark for on all, some or none; its name; review or automatic; its description. */
function TagOption({
  id,
  state,
  single,
  active,
  onHover,
  onChoose,
}: {
  id: string;
  state: TagOnDocuments;
  single: boolean;
  active: boolean;
  onHover(): void;
  onChoose(): void;
}) {
  const t = useT();
  const { tag, coverage } = state;
  const review = single && state.needsReview;
  const notes = [
    review ? t("jev.chip.needsReview") : null,
    state.automatic && coverage !== "none" && !review ? t("tags.picker.automatic") : null,
    coverage === "some" ? t("tags.picker.some", { count: state.count, total: state.total }) : null,
    tag.description || null,
  ].filter((note): note is string => note !== null);
  return (
    // biome-ignore lint/a11y/useKeyWithClickEvents: the field owns the keys (aria-activedescendant).
    <div
      id={id}
      role="option"
      tabIndex={-1}
      aria-selected={coverage === "all"}
      aria-label={tag.name}
      aria-description={notes.join(" · ") || undefined}
      data-testid="tag-option"
      data-tag-id={tag.id}
      data-coverage={coverage}
      data-needs-review={review ? "true" : undefined}
      data-active={active ? "true" : undefined}
      title={tag.description || undefined}
      onMouseDown={(event) => event.preventDefault()}
      onMouseMove={onHover}
      onClick={onChoose}
      className={`flex h-7 cursor-default items-center gap-2 rounded-md pr-2 pl-2 ${active ? "bg-hover" : ""}`}
    >
      <span aria-hidden="true" className="flex size-3.5 shrink-0 items-center justify-center">
        {coverage === "all" && !review && <CheckLineIcon className="size-3.5 text-ink" />}
        {coverage === "some" && <span className="h-0.5 w-2.5 rounded-full bg-ink-secondary" />}
        {review && <ReviewDot />}
      </span>
      <span className="min-w-0 shrink-0 truncate">{tag.name}</span>
      {state.automatic && coverage !== "none" && !review && (
        <SparkLineIcon aria-hidden="true" className="size-2.5 shrink-0 text-ink-meta" />
      )}
      <span aria-hidden="true" className="min-w-0 flex-1 truncate text-[12px] text-ink-meta">
        {review ? t("jev.chip.needsReview") : tag.description}
      </span>
    </div>
  );
}

/** Creating a Tag with a description, which automatic tagging reads. */
function DescribeTag({
  name,
  error,
  onCreate,
  onBack,
}: {
  name: string;
  error: string | null;
  onCreate(description: string): void;
  onBack(): void;
}) {
  const t = useT();
  const [description, setDescription] = useState("");
  return (
    <form
      data-testid="tag-describe"
      className="flex flex-col gap-2 p-3"
      onSubmit={(event) => {
        event.preventDefault();
        onCreate(description);
      }}
      onKeyDown={(event) => {
        if (event.key === "Escape") {
          // Back to the list, not closed.
          event.preventDefault();
          event.stopPropagation();
          onBack();
        }
      }}
    >
      <p className="text-ui font-semibold break-words text-ink">
        {t("tags.picker.create", { name })}
      </p>
      <label className="block text-[13px] leading-5 text-ink-secondary">
        {t("tags.picker.descriptionLabel")}
        <textarea
          data-picker-field
          data-testid="tag-describe-input"
          // biome-ignore lint/a11y/noAutofocus: the form replaces the list the focus was in.
          autoFocus
          rows={2}
          maxLength={500}
          value={description}
          placeholder={t("tags.dialog.descriptionPlaceholder")}
          onChange={(event) => setDescription(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter" && !event.shiftKey) {
              event.preventDefault();
              onCreate(description);
            }
          }}
          className={`${inputClass} resize-none`}
        />
      </label>
      <p className={`${hintClass} mt-0`}>{t("tags.picker.descriptionHint")}</p>
      {error && (
        <p role="alert" className={errorTextClass}>
          {error}
        </p>
      )}
      <div className="flex justify-end gap-2">
        <button type="button" onClick={onBack} className={buttonStyle("ghost", "sm")}>
          {t("tags.picker.back")}
        </button>
        <button
          type="submit"
          data-testid="tag-describe-create"
          className={buttonStyle("primary", "sm")}
        >
          {t("tags.picker.createConfirm")}
        </button>
      </div>
    </form>
  );
}

/** The Documents a picker edits, for its label: "Tags of X" or "Tags of 3 Documents". */
export function pickerLabel(
  t: ReturnType<typeof useT>,
  documents: readonly Pick<Document, "name">[],
): string {
  return documents.length === 1
    ? t("tags.menu.open", { name: documents[0]?.name ?? "" })
    : t("tags.picker.labelMany", { count: documents.length });
}
