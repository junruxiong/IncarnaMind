import {
  type KeyboardEvent,
  type ReactNode,
  useEffect,
  useId,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import type { DocumentTextMatch, MindTextMatch, TextRange } from "../../../core/api";
import type { MessageKey } from "../../../shared/i18n";
import { keysHere, pressesShortcut } from "../commands/keymap";
import { commands } from "../commands/registry";
import { core } from "../core";
import { errorMessage } from "../errors";
import { translateNow, useT } from "../i18n";
import { enter, exit } from "../motion";
import {
  actionGroup,
  documentTextGroup,
  type NamedThings,
  nameGroups,
  noteGroup,
  PALETTE_CONTEXT,
  type PaletteGroup,
  type PaletteGroupId,
  type PaletteItem,
  readQuery,
  recentGroup,
  runItem,
  searchesWords,
  usePalette,
} from "../palette";
import { useAppStore } from "../store";
import { DocumentLineIcon, FolderLineIcon, MindLineIcon, SearchLineIcon } from "./lineIcons";
import { TagDot } from "./TagColour";

/** How long typing pauses before the words inside are searched for. */
const WORDS_DELAY_MS = 120;

const GROUP_TITLES: Record<PaletteGroupId, MessageKey> = {
  recent: "palette.group.recent",
  minds: "palette.group.minds",
  documents: "palette.group.documents",
  foldersAndTags: "palette.group.foldersAndTags",
  actions: "palette.group.actions",
  notes: "palette.group.notes",
  documentText: "palette.group.documentText",
};

const NO_GROUPS: NamedThings["folders"] = [];
const NO_ASSIGNMENTS: NamedThings["assignments"] = [];

/** What the core found for one query; `failed` says why it couldn't search. */
interface Found<T> {
  query: string;
  items: readonly T[];
  failed?: string;
}

/**
 * ⌘K, a native modal <dialog> opened by ⌘K (Ctrl+K) or the sidebar's Search.
 * The field keeps the focus and owns the keys; Esc closes and gives the focus
 * back where it was.
 */
export function CommandPalette() {
  const open = usePalette((state) => state.open);
  return open ? <Palette /> : null;
}

function Palette() {
  const t = useT();
  const id = useId();
  const dialog = useRef<HTMLDialogElement>(null);
  const field = useRef<HTMLInputElement>(null);
  const list = useRef<HTMLDivElement>(null);
  const [query, setQuery] = useState("");
  const [activeKey, setActiveKey] = useState<string | null>(null);
  const [notes, setNotes] = useState<Found<MindTextMatch> | null>(null);
  const [passages, setPassages] = useState<Found<DocumentTextMatch> | null>(null);

  const minds = useAppStore((state) => state.minds);
  const documents = useAppStore((state) => state.documents);
  const tags = useAppStore((state) => state.tags);
  const folders = useAppStore((state) => state.library?.groups ?? NO_GROUPS);
  const assignments = useAppStore((state) => state.library?.assignments ?? NO_ASSIGNMENTS);

  useLayoutEffect(() => {
    const element = dialog.current;
    if (!element) return;
    element.showModal();
    field.current?.focus();
    enter(element, {});
  }, []);

  const { text, words, actionsOnly } = useMemo(() => readQuery(query), [query]);
  const searching = !actionsOnly && searchesWords(text);
  // Asked for once typing pauses; each answer stays until the next arrives.
  useEffect(() => {
    if (!searching) return;
    let current = true;
    const timer = setTimeout(() => {
      core.searchMinds(text).then(
        (items) => current && setNotes({ query: text, items }),
        (error: unknown) =>
          current &&
          setNotes({ query: text, items: [], failed: errorMessage(error, translateNow) }),
      );
      core.searchDocumentText(text).then(
        (items) => current && setPassages({ query: text, items }),
        (error: unknown) =>
          current &&
          setPassages({ query: text, items: [], failed: errorMessage(error, translateNow) }),
      );
    }, WORDS_DELAY_MS);
    return () => {
      current = false;
      clearTimeout(timer);
    };
  }, [searching, text]);

  const things = useMemo(
    () => ({ minds, documents, tags, folders, assignments }),
    [minds, documents, tags, folders, assignments],
  );
  const groups: PaletteGroup[] = useMemo(() => {
    if (actionsOnly) return actionGroup(words, PALETTE_CONTEXT, t);
    if (words.length === 0) {
      return [...recentGroup(things, t), ...actionGroup([], PALETTE_CONTEXT, t)];
    }
    return [
      ...nameGroups(words, things, t),
      ...actionGroup(words, PALETTE_CONTEXT, t),
      ...(searching && notes ? noteGroup(notes.items, minds, t) : []),
      ...(searching && passages ? documentTextGroup(passages.items, documents, t) : []),
    ];
  }, [actionsOnly, words, things, t, searching, notes, passages, minds, documents]);

  const items = groups.flatMap((group) => group.items);
  const active = items.find((item) => item.key === activeKey) ?? items[0];
  const waiting = searching && (notes?.query !== text || passages?.query !== text);
  const problem = searching ? (notes?.failed ?? passages?.failed) : undefined;

  // biome-ignore lint/correctness/useExhaustiveDependencies: the option chosen now is when to scroll, not what is read.
  useLayoutEffect(() => {
    list.current
      ?.querySelector('[role="option"][aria-selected="true"]')
      ?.scrollIntoView({ block: "nearest" });
  }, [active?.key]);

  const close = () => {
    const element = dialog.current;
    if (!element?.open) return;
    exit(element, {});
    element.close();
  };

  const choose = (item: PaletteItem, inTab = false) => {
    // Closed first, so the focus is back where it was before what was chosen moves it.
    close();
    runItem(item, { inTab }).catch((error: unknown) => useAppStore.getState().reportError(error));
  };

  const onKeyDown = (event: KeyboardEvent<HTMLDialogElement>) => {
    if (event.nativeEvent.isComposing) return;
    if (event.key === "Escape" || pressesShortcut("palette.open", event.nativeEvent)) {
      // Here rather than the dialog's cancel, which Chromium may refuse to stop a second time.
      event.preventDefault();
      event.stopPropagation();
      close();
    } else if (event.key === "ArrowDown" || event.key === "ArrowUp") {
      event.preventDefault();
      if (items.length === 0 || !active) return;
      const at = items.indexOf(active);
      const step = event.key === "ArrowDown" ? 1 : -1;
      setActiveKey(items[(at + step + items.length) % items.length]?.key ?? null);
    } else if (event.key === "Enter") {
      event.preventDefault();
      if (active) choose(active, event.metaKey || event.ctrlKey);
    }
  };

  const optionId = (item: PaletteItem) => `${id}-${items.indexOf(item)}`;

  return (
    <dialog
      ref={dialog}
      data-testid="palette"
      aria-label={t("palette.label")}
      onKeyDown={onKeyDown}
      onClose={() => usePalette.getState().hide()}
      className="mx-auto mt-16 mb-auto flex max-h-[calc(100vh-96px)] w-[640px] max-w-[calc(100vw-32px)] flex-col overflow-hidden rounded-xl bg-sheet p-0 text-ink shadow-dialog backdrop:bg-scrim"
    >
      <div className="flex h-[52px] shrink-0 items-center gap-2.5 border-b border-rule px-4">
        <SearchLineIcon className="size-4 shrink-0 text-ink-meta" />
        <input
          ref={field}
          value={query}
          data-testid="palette-search"
          role="combobox"
          aria-expanded="true"
          aria-controls={`${id}-list`}
          aria-activedescendant={active ? optionId(active) : undefined}
          aria-label={t("palette.label")}
          placeholder={t("palette.placeholder")}
          spellCheck={false}
          onChange={(event) => {
            setQuery(event.target.value);
            setActiveKey(null);
          }}
          className="h-full min-w-0 flex-1 bg-transparent text-[15px] text-ink outline-none placeholder:text-ink-placeholder"
        />
        <Key>esc</Key>
      </div>
      <div
        ref={list}
        id={`${id}-list`}
        role="listbox"
        aria-label={t("palette.results")}
        data-testid="palette-results"
        className="thin-scrollbar min-h-0 flex-1 overflow-y-auto px-2 pb-2"
      >
        {groups.map((group) => (
          // biome-ignore lint/a11y/useSemanticElements: options in a listbox are grouped by role, which a fieldset can't be.
          <div
            key={group.id}
            role="group"
            aria-labelledby={`${id}-${group.id}`}
            data-testid="palette-group"
            data-group={group.id}
          >
            <div
              id={`${id}-${group.id}`}
              className="px-2 pt-3 pb-1 text-[12px] leading-[18px] font-semibold text-ink-meta"
            >
              {t(GROUP_TITLES[group.id])}
            </div>
            {group.items.map((item) => (
              <Option
                key={item.key}
                id={optionId(item)}
                item={item}
                active={item === active}
                onPoint={() => item.key !== active?.key && setActiveKey(item.key)}
                onChoose={(inTab) => choose(item, inTab)}
              />
            ))}
          </div>
        ))}
        {problem !== undefined ? (
          <p role="alert" className="px-2 pt-3 text-[13px] leading-5 break-words text-danger">
            {problem}
          </p>
        ) : (
          items.length === 0 &&
          query.trim() !== "" && (
            <p
              role="status"
              data-testid={waiting ? "palette-searching" : "palette-empty"}
              className="px-2 pt-4 pb-2 text-[13px] leading-5 text-ink-meta"
            >
              {t(waiting ? "palette.searching" : "palette.empty")}
            </p>
          )
        )}
      </div>
      <div className="flex h-9 shrink-0 items-center gap-4 border-t border-rule px-4 text-[12px] whitespace-nowrap text-ink-meta">
        <Hint keys="↑↓" label={t("palette.hint.move")} />
        <Hint keys={keysHere("Enter")} label={t("palette.hint.open")} />
        <Hint keys={keysHere("CmdOrCtrl+Enter")} label={t("palette.hint.openTab")} />
        <span className="ml-auto min-w-0 truncate">{t("palette.hint.actionsOnly")}</span>
      </div>
    </dialog>
  );
}

function Hint({ keys, label }: { keys: string; label: string }) {
  return (
    <span>
      <span aria-hidden="true">{keys} </span>
      {label}
    </span>
  );
}

function Key({ children }: { children: ReactNode }) {
  return (
    <kbd className="shrink-0 rounded-sm border border-rule px-1 font-sans text-[11px] leading-4 text-ink-meta">
      {children}
    </kbd>
  );
}

/** Text with the words found in it in bold. */
function Marked({ text, ranges }: { text: string; ranges: readonly TextRange[] }) {
  const parts: ReactNode[] = [];
  let at = 0;
  for (const range of ranges) {
    if (range.start < at) continue;
    if (range.start > at) parts.push(text.slice(at, range.start));
    parts.push(
      <b key={range.start} className="font-semibold text-ink">
        {text.slice(range.start, range.end)}
      </b>,
    );
    at = range.end;
  }
  parts.push(text.slice(at));
  return <>{parts}</>;
}

function ItemIcon({ item }: { item: PaletteItem }) {
  const className = "size-4 shrink-0 text-ink-meta";
  if (item.kind === "note") return <MindLineIcon className={className} />;
  if (item.kind === "passage")
    return <DocumentLineIcon kind={item.documentKind} className={className} />;
  if (item.kind === "command") {
    const Icon = commands.get(item.id)?.icon;
    return Icon ? (
      <Icon className={className} />
    ) : (
      <span aria-hidden="true" className="size-4 shrink-0" />
    );
  }
  switch (item.ref.kind) {
    case "mind":
      return <MindLineIcon className={className} />;
    case "document":
      return <DocumentLineIcon kind={item.documentKind ?? "text"} className={className} />;
    case "folder":
      return <FolderLineIcon className={className} />;
    default:
      return (
        <span aria-hidden="true" className="flex size-4 shrink-0 items-center justify-center">
          <TagDot colour={item.tagColour} />
        </span>
      );
  }
}

function Option({
  id,
  item,
  active,
  onPoint,
  onChoose,
}: {
  id: string;
  item: PaletteItem;
  active: boolean;
  onPoint(): void;
  onChoose(inTab: boolean): void;
}) {
  const words = item.kind === "note" || item.kind === "passage";
  const ranges = item.kind === "object" || item.kind === "command" ? item.ranges : [];
  const meta = item.kind === "command" ? null : item.meta;
  return (
    // biome-ignore lint/a11y/useKeyWithClickEvents: the field owns the keys (aria-activedescendant).
    <div
      id={id}
      role="option"
      tabIndex={-1}
      aria-selected={active}
      data-testid="palette-item"
      data-kind={item.kind === "object" ? item.ref.kind : item.kind}
      // Choosing with the pointer keeps the focus in the field until it closes.
      onMouseDown={(event) => event.preventDefault()}
      onMouseMove={onPoint}
      onClick={(event) => onChoose(event.metaKey || event.ctrlKey)}
      className={`flex cursor-default gap-2.5 rounded-md px-2 ${
        words ? "items-start py-1.5" : "h-9 items-center"
      } ${active ? "bg-hover" : ""}`}
    >
      <span className={`flex shrink-0 ${words ? "pt-0.5" : ""}`}>
        <ItemIcon item={item} />
      </span>
      <span className="min-w-0 flex-1">
        <span className="flex min-w-0 items-baseline gap-2">
          <span className="truncate text-ui text-ink" title={item.name}>
            <Marked text={item.name} ranges={ranges} />
          </span>
          {meta && (
            <span className="min-w-0 shrink truncate text-[12px] text-ink-meta" title={meta}>
              {meta}
            </span>
          )}
        </span>
        {words && (
          <span className="block truncate text-[13px] leading-5 text-ink-secondary">
            <Marked text={item.excerpt} ranges={item.highlights} />
          </span>
        )}
      </span>
      {item.kind === "command" && item.shortcut && (
        <span className="shrink-0 text-[12px] text-ink-meta">{item.shortcut}</span>
      )}
      {active && item.kind !== "command" && (
        <span className={`flex shrink-0 ${words ? "pt-0.5" : ""}`}>
          <Key>↵</Key>
        </span>
      )}
    </div>
  );
}
