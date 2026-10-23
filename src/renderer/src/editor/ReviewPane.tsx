import type { Editor } from "@tiptap/core";
import { useEditorState } from "@tiptap/react";
import { Fragment, type KeyboardEvent, useEffect, useLayoutEffect, useMemo, useRef } from "react";
import {
  ChatLineIcon,
  CheckLineIcon,
  ChevronDownLineIcon,
  CloseLineIcon,
  MoreLineIcon,
} from "../components/lineIcons";
import {
  iconButtonClass,
  menuClass,
  menuItemClass,
  menuRuleClass,
  segmentClass,
  segmentedClass,
} from "../components/ui";
import { usePopoverMenu } from "../components/usePopoverMenu";
import { resolveProposalsIn } from "../editWithAi";
import { useLanguage, useT } from "../i18n";
import { authorKey, type ReviewFilter, useReviewPanes } from "../reviewPane";
import { editorDoc, pendingIn } from "./proposalReview";
import { type PickedRange, pickInMind, scrollMindTo } from "./reviewPick";
import {
  type ChangeItem,
  type CommentItem,
  type ExcerptPart,
  initialsOf,
  madeByLine,
  type ReviewItem,
  reviewItemsIn,
  waitingChanges,
} from "./theirChanges";

type Shown = Omit<ChangeItem, "from" | "to"> | Omit<CommentItem, "from" | "to">;

// Places move with every key typed above them: without them, the cards redraw only when they change.
const withoutPlace = ({ from: _from, to: _to, ...rest }: ReviewItem): Shown => rest;

const cardIn = (list: HTMLElement | null, id: string) =>
  list?.querySelector<HTMLButtonElement>(`[data-review-id="${CSS.escape(id)}"]`) ?? null;

const itemsOf = (editor: Editor) => reviewItemsIn(editor.state.doc, editorDoc(editor));

const decided = (item: Shown) =>
  item.kind === "change" && (item.state === "accepted" || item.state === "rejected");

/**
 * Someone else's changes and comments from the Word file, beside the Mind
 * inside its tab (#697, the board Final-10-Redlines): a card each in the
 * Mind's order; All · Unresolved and a menu of who made them; the selected
 * card opens to the whole change, and its change is washed in the text and
 * scrolled to. A change or comment clicked in the text selects its card.
 */
export function ReviewPane({
  editor,
  mindId,
  over,
}: {
  editor: Editor;
  mindId: string;
  over: boolean;
}) {
  const t = useT();
  const pane = useReviewPanes((state) => state.panes[mindId]);
  const items = useEditorState({
    editor,
    selector: ({ editor: current }) => itemsOf(current).map(withoutPlace),
  });
  const anyDecided = items.some(decided);
  const filter: ReviewFilter = pane?.filter ?? (anyDecided ? "unresolved" : "all");
  const who = pane?.who ?? null;
  const update = (patch: Parameters<ReturnType<typeof useReviewPanes.getState>["update"]>[1]) =>
    useReviewPanes.getState().update(mindId, patch);

  // The default is kept once the pane opens, so deciding a change doesn't switch the list under the User.
  useEffect(() => {
    if (pane && pane.filter === null) {
      useReviewPanes.getState().update(mindId, { filter: anyDecided ? "unresolved" : "all" });
    }
  }, [pane, anyDecided, mindId]);

  const inFilter = (item: Shown) => filter === "all" || !decided(item);
  const shown = items.filter(
    (item) => inFilter(item) && (who === null || authorKey(item.author) === who),
  );
  const selected = shown.some((item) => item.id === pane?.selected)
    ? (pane?.selected ?? null)
    : null;

  const list = useRef<HTMLUListElement>(null);
  const cardButton = (id: string) => cardIn(list.current, id);

  const select = (id: string, from: "pane" | "mind") => {
    update({ selected: id });
    const item = itemsOf(editor).find((each) => each.id === id);
    if (!item) return;
    pickInMind(editor, pickedRanges(editor, item));
    if (from === "pane") scrollMindTo(editor, item.from);
  };

  const opened = pane?.opened ?? 0;
  // biome-ignore lint/correctness/useExhaustiveDependencies: `opened` is what asks for it.
  useLayoutEffect(() => {
    const { from } = editor.state.selection;
    const atCursor = itemsOf(editor).find(
      (each) =>
        each.kind === "change" && each.state === "waiting" && each.from <= from && from <= each.to,
    );
    if (atCursor && atCursor.id !== selected) update({ selected: atCursor.id });
    const target =
      (atCursor ? cardButton(atCursor.id) : null) ??
      (selected ? cardButton(selected) : null) ??
      list.current?.querySelector<HTMLButtonElement>("[data-review-id]") ??
      null;
    target?.focus({ preventScroll: true });
    target?.scrollIntoView({ block: "nearest" });
  }, [opened]);

  // biome-ignore lint/correctness/useExhaustiveDependencies: new `items` mean the change may have moved: washed again where it is.
  useEffect(() => {
    const item = selected ? itemsOf(editor).find((each) => each.id === selected) : undefined;
    pickInMind(editor, item ? pickedRanges(editor, item) : []);
  }, [editor, selected, items]);
  useEffect(() => () => pickInMind(editor, []), [editor]);

  useEffect(() => {
    const showCard = (id: string) => {
      useReviewPanes.getState().update(mindId, { selected: id });
      requestAnimationFrame(() => cardIn(list.current, id)?.scrollIntoView({ block: "nearest" }));
    };
    const onSelection = () => {
      if (!editor.isFocused) return;
      const { from } = editor.state.selection;
      const change = pendingIn(editor.state).find(
        (each) =>
          each.theirs && each.ranges.some((range) => range.from <= from && from <= range.to),
      );
      if (change) showCard(change.id);
    };
    const onClick = (event: MouseEvent) => {
      const mark = event.target instanceof Element ? event.target.closest(".word-atom") : null;
      if (!mark || editor.isDestroyed) return;
      const pos = editor.view.posAtDOM(mark, 0);
      const item = itemsOf(editor).find(
        (each) =>
          (each.kind === "comment" || each.state === "kept") && each.from <= pos && pos < each.to,
      );
      if (item) showCard(item.id);
    };
    const dom = editor.view.dom;
    editor.on("selectionUpdate", onSelection);
    dom.addEventListener("click", onClick);
    return () => {
      editor.off("selectionUpdate", onSelection);
      dom.removeEventListener("click", onClick);
    };
  }, [editor, mindId]);

  const close = () => {
    useReviewPanes.getState().close(mindId);
    if (!editor.isDestroyed) editor.commands.focus(null, { scrollIntoView: false });
  };

  const onListKey = (event: KeyboardEvent<HTMLUListElement>) => {
    if (event.nativeEvent.isComposing) return;
    const at = shown.findIndex((item) => item.id === selected);
    const next =
      event.key === "ArrowDown"
        ? shown[Math.min(at + 1, shown.length - 1)]
        : event.key === "ArrowUp"
          ? shown[Math.max(at - 1, 0)]
          : event.key === "Home"
            ? shown[0]
            : event.key === "End"
              ? shown.at(-1)
              : undefined;
    if (!next) return;
    event.preventDefault();
    select(next.id, "pane");
    const button = cardButton(next.id);
    button?.focus({ preventScroll: true });
    button?.scrollIntoView({ block: "nearest" });
  };

  return (
    <aside
      aria-label={t("review.title")}
      data-testid="review-pane"
      data-placement={over ? "over" : "beside"}
      className={`review-pane ${over ? "review-pane--over" : ""}`}
      onKeyDown={(event) => {
        if (event.key !== "Escape" || event.defaultPrevented || event.nativeEvent.isComposing)
          return;
        event.preventDefault();
        close();
      }}
    >
      <div className="review-pane-head">
        <FilterSwitch
          mindId={mindId}
          filter={filter}
          onChange={(next) => update({ filter: next })}
        />
        <AuthorMenu
          items={items.filter(inFilter)}
          who={who}
          onChange={(next) => update({ who: next })}
        />
        <PaneMenu editor={editor} mindId={mindId} />
        <button
          type="button"
          data-testid="review-close"
          aria-label={t("review.close")}
          title={t("review.close")}
          onClick={close}
          className={iconButtonClass}
        >
          <CloseLineIcon className="size-4" />
        </button>
      </div>
      {shown.length === 0 ? (
        <p data-testid="review-empty" className="review-pane-empty">
          {t("review.empty")}
        </p>
      ) : (
        <ul
          ref={list}
          aria-label={t("review.list")}
          data-testid="review-list"
          className="review-pane-list"
          onKeyDown={onListKey}
        >
          {shown.map((item, index) => (
            <Card
              key={item.id}
              item={item}
              selected={item.id === selected}
              tabbable={selected ? item.id === selected : index === 0}
              onSelect={() => select(item.id, "pane")}
            />
          ))}
        </ul>
      )}
    </aside>
  );
}

function pickedRanges(editor: Editor, item: ReviewItem): PickedRange[] {
  if (item.kind === "comment" || item.state === "kept") {
    return [{ from: item.from, to: item.to, node: true }];
  }
  if (item.state === "waiting") {
    const proposal = pendingIn(editor.state).find((each) => each.id === item.id);
    return (
      proposal?.ranges.map((range) =>
        range.node
          ? { from: range.from, to: range.to, node: true }
          : { from: range.from, to: range.to },
      ) ?? []
    );
  }
  if (item.state !== "accepted") return [];
  const added = item.excerpt
    .filter((part) => part.kind === "insert")
    .map((part) => part.text)
    .join("");
  const size = editor.state.doc.content.size;
  const to = Math.min(item.from + added.length, size);
  return added !== "" && editor.state.doc.textBetween(item.from, to) === added
    ? [{ from: item.from, to }]
    : [];
}

function FilterSwitch({
  mindId,
  filter,
  onChange,
}: {
  mindId: string;
  filter: ReviewFilter;
  onChange(next: ReviewFilter): void;
}) {
  const t = useT();
  const name = `review-filter-${mindId}`;
  const option = (value: ReviewFilter, label: string) => (
    <label className={segmentClass}>
      <input
        type="radio"
        name={name}
        value={value}
        data-testid={`review-filter-${value}`}
        checked={filter === value}
        onChange={() => onChange(value)}
        className="sr-only"
      />
      {label}
    </label>
  );
  return (
    <div role="radiogroup" aria-label={t("review.filter")} className={segmentedClass}>
      {option("all", t("review.all"))}
      {option("unresolved", t("review.unresolved"))}
    </div>
  );
}

function AuthorMenu({
  items,
  who,
  onChange,
}: {
  items: readonly Shown[];
  who: string | null;
  onChange(next: string | null): void;
}) {
  const t = useT();
  const menu = usePopoverMenu({ focus: "checked", align: "end" });
  const authors = useMemo(() => {
    const counts = new Map<string, { name: string; count: number }>();
    for (const item of items) {
      const key = authorKey(item.author);
      const entry = counts.get(key) ?? {
        name: item.author ?? t("proposal.unknownAuthor"),
        count: 0,
      };
      entry.count++;
      counts.set(key, entry);
    }
    return [...counts.entries()];
  }, [items, t]);
  const current = who === null ? null : authors.find(([key]) => key === who)?.[1].name;
  const choose = (next: string | null) => {
    onChange(next);
    menu.close();
  };
  const option = (key: string | null, label: string) => (
    <button
      key={key ?? "\u0000everyone"}
      type="button"
      role="menuitemradio"
      aria-checked={who === key}
      data-testid={key === null ? "review-author-everyone" : "review-author"}
      onClick={() => choose(key)}
      className={`${menuItemClass} pl-2`}
    >
      <span className="flex size-4 shrink-0 items-center justify-center">
        {who === key && <CheckLineIcon className="size-4" />}
      </span>
      <span className="min-w-0 flex-1 truncate">{label}</span>
    </button>
  );
  return (
    <>
      <button
        {...menu.buttonProps}
        type="button"
        data-testid="review-author-menu"
        aria-label={t("review.authors", { name: current ?? t("review.everyone") })}
        title={current ?? undefined}
        className="review-pane-who"
      >
        <span className="min-w-0 truncate">{current ?? t("review.everyone")}</span>
        <ChevronDownLineIcon className="size-3 shrink-0" />
      </button>
      <div
        {...menu.menuProps}
        role="menu"
        aria-label={t("review.authors.menu")}
        data-testid="review-author-list"
        className={menuClass}
      >
        {menu.open && (
          <>
            {option(null, t("review.everyone"))}
            <div className={menuRuleClass} />
            {authors.map(([key, { name, count }]) =>
              option(key, t("review.author", { name, count })),
            )}
          </>
        )}
      </div>
    </>
  );
}

function PaneMenu({ editor, mindId }: { editor: Editor; mindId: string }) {
  const t = useT();
  const menu = usePopoverMenu({ focus: "first", align: "end" });
  const waiting = useEditorState({
    editor,
    selector: ({ editor: current }) =>
      waitingChanges(itemsOf(current)).map(({ id, author }) => ({ id, author })),
  });
  if (waiting.length === 0) return null;
  const authors = [...new Set(waiting.map((each) => authorKey(each.author)))];
  const decide = (ids: string[], outcome: "accept" | "reject") => {
    menu.close();
    resolveProposalsIn(editor, mindId, ids, outcome);
  };
  const item = (testId: string, label: string, ids: string[], outcome: "accept" | "reject") => (
    <button
      key={testId}
      type="button"
      role="menuitem"
      data-testid={testId}
      onClick={() => decide(ids, outcome)}
      className={`${menuItemClass} pl-2`}
    >
      <span className="min-w-0 flex-1 truncate">{label}</span>
    </button>
  );
  const all = waiting.map((each) => each.id);
  return (
    <>
      <button
        {...menu.buttonProps}
        type="button"
        data-testid="review-more"
        aria-label={t("review.more")}
        title={t("review.more")}
        className={iconButtonClass}
      >
        <MoreLineIcon className="size-4" />
      </button>
      <div
        {...menu.menuProps}
        role="menu"
        aria-label={t("review.more")}
        data-testid="review-more-menu"
        className={menuClass}
      >
        {menu.open && (
          <>
            {item("review-accept-all", t("proposal.acceptAll"), all, "accept")}
            {item("review-reject-all", t("proposal.rejectAll"), all, "reject")}
            {authors.length > 1 &&
              authors.map((key) => {
                const name = key === "" ? t("proposal.unknownAuthor") : key;
                const ids = waiting
                  .filter((each) => authorKey(each.author) === key)
                  .map((each) => each.id);
                return (
                  <Fragment key={key}>
                    <div className={menuRuleClass} />
                    {item(
                      "review-accept-author",
                      t("proposal.theirs.acceptAll", { name }),
                      ids,
                      "accept",
                    )}
                    {item(
                      "review-reject-author",
                      t("proposal.theirs.rejectAll", { name }),
                      ids,
                      "reject",
                    )}
                  </Fragment>
                );
              })}
          </>
        )}
      </div>
    </>
  );
}

function Excerpt({ parts, testId }: { parts: readonly ExcerptPart[]; testId: string }) {
  return (
    <span data-testid={testId} className="review-card-excerpt">
      {parts.map((part, index) =>
        part.kind === "insert" ? (
          // biome-ignore lint/suspicious/noArrayIndexKey: the parts of one excerpt, in order.
          <ins key={index}>{part.text}</ins>
        ) : part.kind === "delete" ? (
          // biome-ignore lint/suspicious/noArrayIndexKey: the parts of one excerpt, in order.
          <del key={index}>{part.text}</del>
        ) : (
          // biome-ignore lint/suspicious/noArrayIndexKey: the parts of one excerpt, in order.
          <span key={index}>{part.text}</span>
        ),
      )}
    </span>
  );
}

const spoken = (parts: readonly ExcerptPart[], kind: ExcerptPart["kind"]) =>
  parts
    .filter((part) => part.kind === kind)
    .map((part) => part.text)
    .join(" ")
    .replace(/\s+/g, " ")
    .trim();

function Card({
  item,
  selected,
  tabbable,
  onSelect,
}: {
  item: Shown;
  selected: boolean;
  tabbable: boolean;
  onSelect(): void;
}) {
  const t = useT();
  const language = useLanguage();
  const made = madeByLine(item, t, language);
  if (item.kind === "comment") {
    return (
      <li
        data-testid="review-comment"
        className="review-card"
        data-selected={selected || undefined}
      >
        <button
          type="button"
          data-review-id={item.id}
          tabIndex={tabbable ? 0 : -1}
          aria-current={selected || undefined}
          aria-label={t("review.comment.label", { made, text: item.text })}
          title={made}
          onClick={onSelect}
          className="review-card-main"
        >
          <span className="review-card-mark" aria-hidden="true">
            <ChatLineIcon className="size-3.5" />
          </span>
          <span className="review-card-body">
            <span className="review-card-head">
              <b className="review-card-name">{item.author ?? t("proposal.unknownAuthor")}</b>
            </span>
            <span
              data-testid="review-comment-text"
              className="review-card-excerpt review-card-comment"
            >
              {item.text}
            </span>
          </span>
        </button>
      </li>
    );
  }
  const removed = spoken(item.excerpt, "delete");
  const added = spoken(item.excerpt, "insert");
  // A field or a picture someone added or removed has no words to show: what it is instead.
  const wordless =
    removed === "" && added === ""
      ? t(
          item.excerpt.some((part) => part.kind === "delete")
            ? "wordAtom.kind.deletion"
            : "wordAtom.kind.insertion",
        )
      : null;
  const what = wordless
    ? wordless
    : removed && added
      ? t("proposal.announce.replace", { removed, added })
      : added
        ? t("proposal.announce.add", { added })
        : t("proposal.announce.remove", { removed });
  const status =
    item.state === "accepted"
      ? t("review.accepted")
      : item.state === "rejected"
        ? t("review.rejected")
        : item.state === "kept"
          ? t("review.kept")
          : item.outOfDate
            ? t("proposal.outOfDate")
            : null;
  return (
    <li
      data-testid="review-change"
      data-state={item.state}
      className="review-card"
      data-selected={selected || undefined}
    >
      <button
        type="button"
        data-review-id={item.id}
        tabIndex={tabbable ? 0 : -1}
        aria-current={selected || undefined}
        aria-label={[made, item.clause, what, status].filter(Boolean).join(", ")}
        title={made}
        onClick={onSelect}
        className="review-card-main"
      >
        <span className="review-card-initials" aria-hidden="true">
          {initialsOf(item.author)}
        </span>
        <span className="review-card-body">
          <span className="review-card-head">
            {item.clause && (
              <b data-testid="review-clause" className="review-card-clause">
                {item.clause}
              </b>
            )}
            {status && (
              <span
                data-testid="review-status"
                title={item.state === "kept" ? t("review.kept.why") : undefined}
                className="review-card-status"
              >
                {item.state === "accepted" && <CheckLineIcon className="size-3" />}
                {item.state === "rejected" && <CloseLineIcon className="size-3" />}
                {status}
              </span>
            )}
          </span>
          {wordless ? (
            <span data-testid="review-excerpt" className="review-card-excerpt review-card-what">
              {wordless}
            </span>
          ) : (
            <Excerpt parts={item.excerpt} testId="review-excerpt" />
          )}
        </span>
      </button>
    </li>
  );
}

/**
 * "Changes ({n})" at the page's top right, left of its ⋯ button, while the
 * Mind is too narrow for the pane beside it: it opens the pane over the
 * Mind's right edge, and folds it again.
 */
export function ReviewFold({ editor, mindId }: { editor: Editor; mindId: string }) {
  const t = useT();
  const over = useReviewPanes((state) => state.panes[mindId]?.over ?? false);
  const count = useEditorState({
    editor,
    selector: ({ editor: current }) =>
      pendingIn(current.state).filter((each) => each.theirs).length,
  });
  return (
    <button
      type="button"
      data-testid="review-fold"
      aria-expanded={over}
      onClick={() => {
        if (over) useReviewPanes.getState().update(mindId, { over: false });
        else useReviewPanes.getState().open(mindId);
      }}
      className="review-fold"
    >
      {t("review.fold", { count })}
    </button>
  );
}
