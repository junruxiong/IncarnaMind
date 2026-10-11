import { useT } from "../i18n";
import { useAppStore } from "../store";
import { FolderLineIcon } from "./lineIcons";
import { SearchPicker } from "./SearchPicker";

/**
 * "Move to…": a Folder picker that opens beside what is moved (`anchor`),
 * with a field to find a Folder by name. "Not in a Folder" first, then every
 * Folder by name; the one it is in now is ticked. Typing filters, the arrows
 * move, Enter moves it there, Esc closes. Shared by a Mind's menu and (#212)
 * the Documents, the Library's selection and ⌘⇧M.
 */
export function MoveToPicker({
  anchor,
  name,
  current,
  onChoose,
  onClose,
}: {
  /** What it opens beside; it takes the focus back once closed. Null: closed. */
  anchor: HTMLElement | null;
  /** What is moved: a Mind's title, or "3 Documents". */
  name: string;
  /** Where it is now (null: Not in a Folder), or undefined for things in different places. */
  current: string | null | undefined;
  onChoose(folderId: string | null): void;
  onClose(): void;
}) {
  const t = useT();
  const groups = useAppStore((state) => state.library?.groups) ?? [];
  const options = [
    { id: null, label: t("library.unsorted") },
    ...groups.map((group) => ({ id: group.id, label: group.name })),
  ].map((option) => ({
    ...option,
    data: { "data-folder-id": option.id ?? "" },
    icon: (
      <FolderLineIcon
        className={`size-4 shrink-0 text-ink-meta ${option.id === null ? "opacity-60" : ""}`}
      />
    ),
  }));
  return (
    <SearchPicker<string | null>
      anchor={anchor}
      title={t("move.title", { name })}
      searchLabel={t("move.search")}
      noMatch={() => t("move.none")}
      options={options}
      current={current}
      currentLabel={t("move.here")}
      onChoose={onChoose}
      onClose={onClose}
      testId="move-to"
    />
  );
}
