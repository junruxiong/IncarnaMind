import { useLanguage, useT } from "../i18n";
import type { FacetOption, FilterSelection, LibraryFacet } from "../libraryFilters";
import { formatCount } from "../linkedFolders";
import { CheckLineIcon, ChevronDownLineIcon } from "./lineIcons";
import { TagSwatch } from "./TagColour";
import { buttonStyle, menuClass, menuItemClass, menuTitleClass } from "./ui";
import { usePopoverMenu } from "./usePopoverMenu";

/**
 * The Library's filters, in one row above its Documents: a button per filter
 * (year, format, status, or any other in `facets`), each opening a menu of its
 * options with how many Documents each holds, then "Clear filters" while any
 * is chosen. A button says what it keeps once something in it is chosen.
 */
export function LibraryFilterBar<T>({
  facets,
  options,
  selection,
  onToggle,
  onClear,
}: {
  facets: readonly LibraryFacet<T>[];
  options: ReadonlyMap<string, FacetOption[]>;
  selection: FilterSelection;
  onToggle(facetId: string, value: string): void;
  onClear(): void;
}) {
  const t = useT();
  const filtering = facets.some((facet) => (selection[facet.id]?.length ?? 0) > 0);
  return (
    <fieldset
      aria-label={t("library.filter.label")}
      data-testid="library-filters"
      className="mb-3 flex min-w-0 flex-wrap items-center gap-1"
    >
      {facets.map((facet) => (
        <FilterMenu
          key={facet.id}
          facet={facet}
          options={options.get(facet.id) ?? []}
          onToggle={(value) => onToggle(facet.id, value)}
        />
      ))}
      {filtering && (
        <button
          type="button"
          data-testid="library-filters-clear"
          onClick={onClear}
          className="ml-1 h-7 rounded-md px-1.5 text-[13px] text-ink-meta hover:text-ink hover:underline focus-visible:outline-offset-0"
        >
          {t("library.filter.clear")}
        </button>
      )}
    </fieldset>
  );
}

/** One filter: its button, which names what it keeps once chosen, and its menu of options. */
function FilterMenu<T>({
  facet,
  options,
  onToggle,
}: {
  facet: LibraryFacet<T>;
  options: readonly FacetOption[];
  onToggle(value: string): void;
}) {
  const t = useT();
  const language = useLanguage();
  const menu = usePopoverMenu();
  const chosen = options.filter((option) => option.selected);
  const title = t(facet.title);
  const first = chosen[0];
  const label = !first
    ? title
    : chosen.length === 1
      ? t("library.filter.chosen", { filter: title, value: facet.label(first.value, t) })
      : t("library.filter.chosenMore", {
          filter: title,
          value: facet.label(first.value, t),
          more: chosen.length - 1,
        });
  return (
    <>
      <button
        {...menu.buttonProps}
        type="button"
        data-testid="library-filter"
        data-facet={facet.id}
        data-active={first ? "true" : undefined}
        className={`${buttonStyle("ghost", "sm")} max-w-64 ${
          first ? "text-ink shadow-[inset_0_0_0_1px_var(--color-rule-strong)]" : ""
        }`}
      >
        <span className="truncate">{label}</span>
        <ChevronDownLineIcon className="size-3 shrink-0 text-ink-meta" />
      </button>
      <div
        {...menu.menuProps}
        role="menu"
        aria-label={title}
        data-testid="library-filter-menu"
        data-facet={facet.id}
        className={menuClass}
      >
        {menu.open && (
          <>
            <p className={menuTitleClass}>{title}</p>
            {options.map(({ value, count, selected }) => {
              const name = facet.label(value, t);
              return (
                <button
                  key={value}
                  type="button"
                  role="menuitemcheckbox"
                  aria-checked={selected}
                  aria-label={t(
                    count === 1 ? "library.filter.option.one" : "library.filter.option",
                    { value: name, count: formatCount(count, language) },
                  )}
                  data-testid="library-filter-option"
                  data-value={value}
                  data-count={count}
                  onClick={() => onToggle(value)}
                  className={`${menuItemClass} pl-2 ${count === 0 && !selected ? "text-ink-meta" : ""}`}
                >
                  <CheckLineIcon
                    className={`size-3.5 shrink-0 ${selected ? "text-ink" : "invisible"}`}
                  />
                  {facet.colour?.(value) && <TagSwatch colour={facet.colour(value) ?? undefined} />}
                  <span className="min-w-0 flex-1 truncate">{name}</span>
                  <span
                    aria-hidden="true"
                    className="shrink-0 text-[12px] tabular-nums text-ink-meta"
                  >
                    {formatCount(count, language)}
                  </span>
                </button>
              );
            })}
          </>
        )}
      </div>
    </>
  );
}
