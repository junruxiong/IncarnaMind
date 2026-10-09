import { useEffect, useRef, useState } from "react";
import { useShallow } from "zustand/react/shallow";
import type { LibraryGroup } from "../../../core/api";
import { libraryPresetKeys } from "../../../shared/libraryPresets";
import { core } from "../core";
import { errorMessage as messageOf } from "../errors";
import { useT } from "../i18n";
import { selectVisibleDocuments, useAppStore } from "../store";
import { ActiveTagFilter, DocumentTagMenu } from "./DocumentTags";
import { DocumentLineIcon, SettingsLineIcon } from "./lineIcons";
import { buttonStyle, errorTextClass, fieldLabelClass, inputClass } from "./ui";

type Action = (work: () => Promise<unknown>) => Promise<boolean>;

/** The sidebar chooses a folder; this sheet shows its documents and editable tags. */
export function LibraryPane() {
  const t = useT();
  const documents = useAppStore(useShallow(selectVisibleDocuments));
  const hasDocuments = useAppStore((state) => state.documents.length > 0);
  const snapshot = useAppStore((state) => state.library);
  const filter = useAppStore((state) => state.libraryFilter);
  const tags = useAppStore((state) => state.tags);
  const [search, setSearch] = useState("");
  const [limit, setLimit] = useState(100);
  const [editing, setEditing] = useState<LibraryGroup | "new" | null>(null);
  const [starters, setStarters] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    setLimit(100);
    setSearch("");
    setDeleting(false);
    setEditing(filter === "new" ? "new" : null);
  }, [filter]);
  const act: Action = async (work) => {
    setError(null);
    setBusy(true);
    try {
      await work();
      await useAppStore.getState().refreshLibrary();
      return true;
    } catch (failure) {
      setError(messageOf(failure));
      return false;
    } finally {
      setBusy(false);
    }
  };
  const groups = snapshot?.groups ?? [];
  const assignments = new Map(snapshot?.assignments.map((item) => [item.documentId, item]));
  const selected = groups.find((group) => group.id === filter);
  const shown = documents.filter((doc) => {
    const groupId = assignments.get(doc.id)?.groupId ?? null;
    const terms = [
      doc.name,
      ...doc.tags.map((link) => tags.find((tag) => tag.id === link.tagId)?.name ?? ""),
    ].join(" ");
    return (
      ((!selected && filter !== "unsorted") ||
        (filter === "unsorted" ? groupId === null : groupId === selected?.id)) &&
      terms.toLocaleLowerCase().includes(search.toLocaleLowerCase())
    );
  });
  const pending =
    snapshot?.assignments.filter(
      (item) => item.status === "pending" || item.status === "classifying",
    ).length ?? 0;
  const waiting = snapshot?.assignments.filter((item) => item.status === "waiting").length ?? 0;
  const failed = snapshot?.assignments.filter((item) => item.status === "failed").length ?? 0;
  const needsOrganization = shown.filter((doc) => {
    const item = assignments.get(doc.id);
    return item?.status !== "classified" || doc.tagging !== "tagged";
  });
  const organize = () => {
    if (!snapshot?.settings.classifier) {
      useAppStore.getState().openSettings("organization");
      return;
    }
    void act(() =>
      core.classifyDocuments(
        (needsOrganization.length ? needsOrganization : shown).map((doc) => doc.id),
      ),
    );
  };
  const closeForm = () => {
    setEditing(null);
    if (filter === "new") useAppStore.getState().openLibrary();
  };
  return (
    <main
      data-testid="library"
      className="library-pane flex min-w-0 flex-1 flex-col overflow-hidden bg-sheet text-ui text-ink"
    >
      <header className="flex h-11 shrink-0 items-center justify-between bg-tab-strip px-4">
        <span className="font-semibold">{t("library.title")}</span>
        <button
          type="button"
          className={buttonStyle("ghost", "sm")}
          onClick={() => useAppStore.getState().closeLibrary()}
        >
          {t("library.back")}
        </button>
      </header>
      <div className="min-h-0 flex-1 overflow-y-auto px-6 py-6">
        <div className="mx-auto max-w-[1040px]">
          <div className="mb-2 flex flex-wrap items-start justify-between gap-3">
            <div className="min-w-0 flex-1">
              <h1 className="break-words font-serif text-heading font-semibold">
                {selected?.name ?? t(filter === "unsorted" ? "library.unsorted" : "library.all")}
              </h1>
              <p className="mt-1 text-[13px] text-ink-meta">
                {selected?.description || t("library.intro")}
              </p>
            </div>
            <button
              type="button"
              className={buttonStyle("primary")}
              disabled={busy || pending > 0 || groups.length === 0 || shown.length === 0}
              onClick={organize}
            >
              {t(pending ? "library.working" : "library.classify")}
            </button>
          </div>
          <div className="mb-5 flex flex-wrap items-center gap-1">
            <button
              type="button"
              className={buttonStyle("ghost", "sm")}
              onClick={() => {
                setEditing("new");
                setStarters(false);
              }}
            >
              {t("library.newGroup")}
            </button>
            <button
              type="button"
              className={buttonStyle("ghost", "sm")}
              onClick={() => {
                setStarters(!starters);
                setEditing(null);
              }}
            >
              {t("library.chooseStarters")}
            </button>
            {selected && (
              <>
                <button
                  type="button"
                  className={buttonStyle("ghost", "sm")}
                  onClick={() => setEditing(selected)}
                >
                  {t("library.edit")}
                </button>
                <button
                  type="button"
                  className={buttonStyle("ghost", "sm")}
                  onClick={() => setDeleting(!deleting)}
                >
                  {t("library.delete")}
                </button>
              </>
            )}
            <button
              type="button"
              className={`${buttonStyle("ghost", "sm")} ml-auto`}
              aria-label={t("library.settingsTitle")}
              title={t("library.settingsTitle")}
              onClick={() => useAppStore.getState().openSettings("organization")}
            >
              <SettingsLineIcon className="size-4" />
            </button>
          </div>
          {error && (
            <p role="alert" className={`${errorTextClass} mb-4`}>
              {error}
            </p>
          )}
          {deleting && selected && (
            <div className="mb-5 border-y border-rule py-4">
              <p className="mb-3 text-ink-secondary">{t("library.deleteNotice")}</p>
              <div className="flex gap-2">
                <button
                  type="button"
                  disabled={busy}
                  className={buttonStyle("danger")}
                  onClick={() =>
                    void act(async () => {
                      await core.deleteLibraryGroup(selected.id);
                      useAppStore.getState().openLibrary();
                    })
                  }
                >
                  {t("library.delete")}
                </button>
                <button
                  type="button"
                  className={buttonStyle("ghost")}
                  onClick={() => setDeleting(false)}
                >
                  {t("library.cancel")}
                </button>
              </div>
            </div>
          )}
          {editing && (
            <GroupForm
              key={editing === "new" ? "new" : editing.id}
              group={editing}
              busy={busy}
              act={act}
              close={closeForm}
            />
          )}
          {(starters || groups.length === 0) && !editing && snapshot && (
            <StarterGroups busy={busy} act={act} close={() => setStarters(false)} />
          )}
          <div className="mb-3 flex flex-wrap items-center gap-3">
            <input
              aria-label={t("library.search")}
              placeholder={t("library.search")}
              type="search"
              value={search}
              onChange={(event) => {
                setSearch(event.target.value);
                setLimit(100);
              }}
              className={`${inputClass} mt-0 min-w-32 flex-1`}
            />
            <button
              type="button"
              className={buttonStyle("secondary")}
              onClick={() => void useAppStore.getState().pickDocuments()}
            >
              {t("documents.add")}
            </button>
          </div>
          <ActiveTagFilter />
          <div className="mb-3 flex flex-wrap items-center justify-between gap-2 text-[13px] text-ink-meta">
            <p role="status">
              {!snapshot
                ? t("library.loading")
                : pending
                  ? t("library.progress", { count: pending })
                  : failed
                    ? t("library.failed", { count: failed })
                    : waiting
                      ? t("library.waiting", { count: waiting })
                      : t(shown.length === 1 ? "library.count.one" : "library.count", {
                          count: shown.length,
                        })}
            </p>
            <button
              type="button"
              className="hover:text-ink hover:underline focus-visible:outline-2 focus-visible:outline-accent"
              onClick={() => useAppStore.getState().openTagsDialog()}
            >
              {t("library.manageTags")}
            </button>
          </div>
          {shown.length === 0 ? (
            <p className="border-t border-rule py-10 text-ink-meta">
              {t(hasDocuments ? "library.noMatches" : "library.empty")}
            </p>
          ) : (
            <>
              <div
                aria-hidden="true"
                className="library-columns library-column-labels border-y border-rule py-2 text-label font-semibold text-ink-meta"
              >
                <span>{t("library.document")}</span>
                <span>{t("library.folder")}</span>
                <span>{t("tags.title")}</span>
              </div>
              <ul>
                {shown.slice(0, limit).map((doc) => {
                  const item = assignments.get(doc.id);
                  const labels = doc.tags.flatMap((link) => {
                    const tag = tags.find((tag) => tag.id === link.tagId);
                    return tag ? [{ ...tag, needsReview: link.needsReview }] : [];
                  });
                  const status =
                    item?.status === "classifying" || item?.status === "pending"
                      ? "library.working"
                      : item?.status === "waiting" || item?.status === "failed"
                        ? "library.needsAttention"
                        : item?.source === "user"
                          ? "library.manual"
                          : item?.status === "classified"
                            ? "library.saved"
                            : "library.notClassified";
                  return (
                    <li
                      key={doc.id}
                      data-testid="library-document"
                      data-document-id={doc.id}
                      className="library-columns border-b border-rule py-3"
                    >
                      <div className="library-document-name min-w-0">
                        <button
                          type="button"
                          title={doc.name}
                          className="flex max-w-full items-center gap-2 text-left hover:underline focus-visible:outline-2 focus-visible:outline-accent"
                          onClick={() =>
                            useAppStore.getState().openDocument({ documentId: doc.id })
                          }
                        >
                          <DocumentLineIcon
                            kind={doc.kind}
                            className="size-4 shrink-0 text-ink-meta"
                          />
                          <span className="truncate font-semibold">{doc.name}</span>
                        </button>
                        <details className="mt-1 text-[12px] text-ink-meta">
                          <summary className="w-fit cursor-pointer">{t(status)}</summary>
                          <p className="mt-1">
                            {doc.kind.toUpperCase()} · {t("library.originalsStay")}
                          </p>
                          {item?.model && (
                            <p>
                              {item.model.id} ·{" "}
                              {t(item.model.images ? "library.usedPages" : "library.usedText")}
                              {item.model.reason === "slow"
                                ? ` · ${t("library.route.slow")}`
                                : item.model.reason === "memory"
                                  ? ` · ${t("library.route.memory")}`
                                  : item.model.reason === "unavailable"
                                    ? ` · ${t("library.route.unavailable")}`
                                    : ""}
                            </p>
                          )}
                          {item?.error && (
                            <p className={`${errorTextClass} mt-1`}>{item.error.message}</p>
                          )}
                        </details>
                      </div>
                      <select
                        aria-label={t("library.groupFor", { name: doc.name })}
                        disabled={busy}
                        value={item?.groupId ?? ""}
                        className={`${inputClass} mt-0 min-w-0`}
                        onChange={(event) =>
                          void act(() =>
                            core.assignDocumentGroup(doc.id, event.target.value || null),
                          )
                        }
                      >
                        <option value="">{t("library.unsorted")}</option>
                        {groups.map((group) => (
                          <option key={group.id} value={group.id}>
                            {group.name}
                          </option>
                        ))}
                      </select>
                      <DocumentTagMenu
                        item={doc}
                        buttonClassName="flex min-h-8 min-w-0 flex-wrap items-center gap-1 rounded-sm text-left text-[12px] text-ink-secondary outline-none hover:bg-frame focus-visible:outline-2 focus-visible:outline-accent"
                      >
                        {labels.length ? (
                          labels.map((tag) => (
                            <span
                              key={tag.id}
                              className="max-w-full truncate rounded-sm bg-chip px-2 py-0.5"
                              title={tag.needsReview ? t("jev.chip.needsReview") : tag.description}
                            >
                              {tag.name}
                              {tag.needsReview ? " · ?" : ""}
                            </span>
                          ))
                        ) : (
                          <span className="px-2 text-ink-meta">{t("library.addTags")}</span>
                        )}
                      </DocumentTagMenu>
                    </li>
                  );
                })}
              </ul>
            </>
          )}
          {shown.length > limit && (
            <button
              type="button"
              className={`${buttonStyle("secondary")} mt-4`}
              onClick={() => setLimit(limit + 100)}
            >
              {t("library.showMore")}
            </button>
          )}
        </div>
      </div>
    </main>
  );
}

function GroupForm({
  group,
  busy,
  act,
  close,
}: {
  group: LibraryGroup | "new";
  busy: boolean;
  act: Action;
  close(): void;
}) {
  const t = useT();
  const nameInput = useRef<HTMLInputElement>(null);
  useEffect(() => {
    nameInput.current?.focus();
  }, []);
  const [name, setName] = useState(group === "new" ? "" : group.name);
  const [description, setDescription] = useState(group === "new" ? "" : group.description);
  return (
    <form
      className="mb-5 max-w-xl border-y border-rule py-4"
      onSubmit={(event) => {
        event.preventDefault();
        void act(async () => {
          const input = { name, description };
          if (group === "new") await core.createLibraryGroup(input);
          else await core.updateLibraryGroup(group.id, input);
          close();
        });
      }}
    >
      <label className={fieldLabelClass}>
        {t("library.name")}
        <input
          ref={nameInput}
          required
          maxLength={100}
          value={name}
          onChange={(event) => setName(event.target.value)}
          className={inputClass}
        />
      </label>
      <label className={`${fieldLabelClass} mt-3`}>
        {t("library.description")}
        <textarea
          rows={2}
          maxLength={500}
          value={description}
          onChange={(event) => setDescription(event.target.value)}
          placeholder={t("library.descriptionHint")}
          className={inputClass}
        />
      </label>
      <div className="mt-3 flex gap-2">
        <button type="submit" disabled={busy || !name.trim()} className={buttonStyle("primary")}>
          {t("library.save")}
        </button>
        <button type="button" className={buttonStyle("ghost")} onClick={close}>
          {t("library.cancel")}
        </button>
      </div>
    </form>
  );
}

function StarterGroups({ busy, act, close }: { busy: boolean; act: Action; close(): void }) {
  const t = useT();
  const [chosen, setChosen] = useState<string[]>([...libraryPresetKeys]);
  return (
    <section
      aria-label={t("library.chooseStarters")}
      className="mb-5 max-w-xl border-y border-rule py-4"
    >
      <h3 className="mb-1 font-semibold">{t("library.chooseStarters")}</h3>
      <p className="mb-3 text-[13px] text-ink-meta">{t("library.startersHint")}</p>
      {libraryPresetKeys.map((key) => (
        <label key={key} className="flex cursor-pointer items-start gap-3 py-2">
          <input
            type="checkbox"
            checked={chosen.includes(key)}
            onChange={(event) =>
              setChosen(
                event.target.checked ? [...chosen, key] : chosen.filter((item) => item !== key),
              )
            }
            className="mt-1 size-4 accent-ink"
          />
          <span>
            <span className="font-semibold">{t(`library.preset.${key}`)}</span>
            <span className="block text-[13px] text-ink-meta">
              {t(`library.preset.${key}.description`)}
            </span>
          </span>
        </label>
      ))}
      <button
        type="button"
        disabled={busy || chosen.length === 0}
        className={`${buttonStyle("primary")} mt-3`}
        onClick={() =>
          void act(async () => {
            await core.addLibraryStarterGroups(chosen);
            close();
          })
        }
      >
        {t("library.addSelected")}
      </button>
    </section>
  );
}
