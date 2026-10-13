import { type RefObject, useEffect } from "react";
import { useLanguage, useT } from "../../i18n";
import { formatCount } from "../../linkedFolders";
import { typesText } from "../../undo";
import { AddDocumentsMenu } from "../AddDocumentsMenu";
import { SearchIcon } from "../icons";
import { CloseLineIcon } from "../lineIcons";

/**
 * The page's head: "Library" and how many Documents it holds, as the
 * sidebar says, the one primary "Add Documents ▾", and under them the wide
 * search field, which `/` focuses from anywhere on the page but a field.
 */
export function LibraryHeader({
  count,
  search,
  onSearch,
  field,
}: {
  count: number;
  search: string;
  onSearch(search: string): void;
  field: RefObject<HTMLInputElement | null>;
}) {
  const t = useT();
  const language = useLanguage();

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key !== "/" || event.metaKey || event.ctrlKey || event.altKey) return;
      if (event.defaultPrevented || typesText(event.target) || !field.current) return;
      if (document.querySelector("dialog[open], :popover-open")) return;
      event.preventDefault();
      field.current.focus();
      field.current.select();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [field]);

  return (
    <header>
      <div className="flex min-w-0 items-center gap-3">
        <h1
          data-testid="library-heading"
          className="flex min-w-0 flex-1 items-baseline gap-2 font-serif text-heading font-semibold text-ink"
        >
          <span className="truncate">{t("library.title")}</span>
          <span
            data-testid="library-count"
            className="shrink-0 font-sans text-[13px] font-normal tabular-nums text-ink-meta"
          >
            {formatCount(count, language)}
          </span>
        </h1>
        <AddDocumentsMenu />
      </div>
      <label
        title={t("library.search.hint")}
        className="mt-3 flex h-10 min-w-0 cursor-text items-center gap-2.5 rounded-md border border-rule-strong bg-sheet px-3 focus-within:border-accent"
      >
        <SearchIcon className="size-4 shrink-0 text-ink-meta" />
        <input
          ref={field}
          data-testid="library-search"
          aria-label={t("library.search.label")}
          placeholder={t("library.search")}
          type="search"
          value={search}
          onChange={(event) => onSearch(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Escape" && search !== "") {
              event.preventDefault();
              event.stopPropagation();
              onSearch("");
            }
          }}
          className="min-w-0 flex-1 bg-transparent text-ui text-ink outline-0 placeholder:text-ink-placeholder [&::-webkit-search-cancel-button]:hidden"
        />
        {search === "" ? (
          <kbd className="shrink-0 rounded-sm bg-chip px-1.5 font-sans text-[11px] leading-4 text-ink-meta">
            /
          </kbd>
        ) : (
          <button
            type="button"
            data-testid="library-search-clear"
            aria-label={t("library.search.clear")}
            title={t("library.search.clear")}
            onClick={() => {
              onSearch("");
              field.current?.focus();
            }}
            className="-mr-1 inline-flex size-6 shrink-0 items-center justify-center rounded-sm text-ink-meta hover:bg-hover hover:text-ink"
          >
            <CloseLineIcon className="size-3.5" />
          </button>
        )}
      </label>
    </header>
  );
}
