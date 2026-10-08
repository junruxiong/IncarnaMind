import { useEffect, useRef, useState } from "react";
import type {
  ChatProvider,
  LibraryClassifier,
  LibraryGroup,
  LibrarySettings,
  LibrarySnapshot,
} from "../../../core/api";
import { supportsLibraryImages } from "../../../shared/libraryModels";
import { libraryPresetKeys } from "../../../shared/libraryPresets";
import { core } from "../core";
import { errorMessage as messageOf } from "../errors";
import { useT } from "../i18n";
import { useAppStore } from "../store";
import { buttonStyle, errorTextClass, fieldLabelClass, inputClass, navRowClass } from "./ui";

type Action = (work: () => Promise<unknown>) => Promise<boolean>;

/** Flat groups, a reviewable document list, and an optional connected classifier. */
export function LibraryPane() {
  const t = useT();
  const documents = useAppStore((state) => state.documents);
  const openDocument = useAppStore((state) => state.openDocument);
  const close = useAppStore((state) => state.closeLibrary);
  const [snapshot, setSnapshot] = useState<LibrarySnapshot | null>(null);
  const [filter, setFilter] = useState("all");
  const [search, setSearch] = useState("");
  const [limit, setLimit] = useState(100);
  const [editing, setEditing] = useState<LibraryGroup | "new" | null>(null);
  const [starters, setStarters] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const refresh = useRef<() => Promise<void>>(async () => {});

  useEffect(() => {
    let stopped = false;
    let request = 0;
    refresh.current = async () => {
      const ticket = ++request;
      try {
        const value = await core.getLibrary();
        if (!stopped && ticket === request) setSnapshot(value);
      } catch (failure) {
        if (!stopped) setError(messageOf(failure));
      }
    };
    const unsubscribe = core.on("library.changed", () => void refresh.current());
    void refresh.current();
    return () => {
      stopped = true;
      unsubscribe();
    };
  }, []);

  const act: Action = async (work) => {
    setError(null);
    setBusy(true);
    try {
      await work();
      await refresh.current();
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
    return (
      (filter === "all" || (filter === "unsorted" ? groupId === null : groupId === filter)) &&
      doc.name.toLocaleLowerCase().includes(search.toLocaleLowerCase())
    );
  });
  const counts = new Map<string | null, number>();
  for (const doc of documents) {
    const id = assignments.get(doc.id)?.groupId ?? null;
    counts.set(id, (counts.get(id) ?? 0) + 1);
  }
  const count = (id: string | null) => counts.get(id) ?? 0;
  const pending =
    snapshot?.assignments.filter(
      (item) => item.status === "pending" || item.status === "classifying",
    ).length ?? 0;
  const waiting = snapshot?.assignments.filter((item) => item.status === "waiting").length ?? 0;
  const failed = snapshot?.assignments.filter((item) => item.status === "failed").length ?? 0;
  const needsClassification = shown.filter((doc) => {
    const item = assignments.get(doc.id);
    return (
      item?.source !== "user" &&
      (!item || item.groupId === null || item.status === "failed" || item.status === "waiting")
    );
  });

  return (
    <main
      data-testid="library"
      className="library-pane flex min-w-0 flex-1 flex-col overflow-hidden bg-sheet text-ui text-ink"
    >
      <header className="flex h-11 shrink-0 items-center justify-between bg-tab-strip px-4">
        <h1 className="font-semibold">{t("library.title")}</h1>
        <button type="button" className={buttonStyle("ghost", "sm")} onClick={close}>
          {t("library.back")}
        </button>
      </header>
      {error && (
        <p role="alert" className={`${errorTextClass} border-b border-rule px-4 py-3`}>
          {error}
        </p>
      )}
      {!snapshot ? (
        <p role="status" className="p-6 text-ink-meta">
          {t("library.loading")}
        </p>
      ) : (
        <div className="library-layout flex min-h-0 flex-1">
          <nav
            aria-label={t("library.groups")}
            className="library-groups flex w-52 shrink-0 flex-col gap-1 overflow-y-auto border-r border-rule bg-frame p-3"
          >
            <span className="px-2 pb-1 text-label font-semibold text-ink-meta">
              {t("library.groups")}
            </span>
            {[
              { id: "all", name: t("library.all"), count: documents.length },
              ...groups.map((group) => ({ ...group, count: count(group.id) })),
              { id: "unsorted", name: t("library.unsorted"), count: count(null) },
            ].map((item) => (
              <button
                type="button"
                key={item.id}
                aria-current={filter === item.id ? "page" : undefined}
                className={`${navRowClass(filter === item.id)} gap-2`}
                onClick={() => {
                  setFilter(item.id);
                  setLimit(100);
                  setEditing(null);
                  setDeleting(false);
                }}
              >
                <span className="min-w-0 flex-1 truncate" title={item.name}>
                  {item.name}
                </span>
                <span className="text-[12px] font-normal tabular-nums text-ink-meta">
                  {item.count}
                </span>
              </button>
            ))}
            <button
              type="button"
              className={`${buttonStyle("ghost", "sm")} mt-3 justify-start`}
              onClick={() => {
                setEditing("new");
                setStarters(false);
              }}
            >
              {t("library.newGroup")}
            </button>
            <button
              type="button"
              className={`${buttonStyle("ghost", "sm")} justify-start`}
              onClick={() => {
                setStarters(!starters);
                setEditing(null);
              }}
            >
              {t("library.chooseStarters")}
            </button>
          </nav>
          <div className="min-w-0 flex-1 overflow-y-auto px-6 py-5">
            <div className="mb-5 flex flex-wrap items-start justify-between gap-3">
              <div className="min-w-0">
                <h2 className="break-words font-serif text-[22px] leading-[30px] font-semibold">
                  {selected?.name ?? t(filter === "unsorted" ? "library.unsorted" : "library.all")}
                </h2>
                <p className="mt-1 max-w-[65ch] break-words text-[13px] text-ink-meta">
                  {selected?.description || t("library.intro")}
                </p>
              </div>
              {selected && (
                <div className="flex gap-1">
                  <button
                    type="button"
                    className={buttonStyle("ghost", "sm")}
                    onClick={() => {
                      setEditing(selected);
                      setDeleting(false);
                    }}
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
                </div>
              )}
            </div>
            {deleting && selected && (
              <div className="mb-5 border-y border-rule py-3">
                <p className="mb-2 text-ink-secondary">{t("library.deleteNotice")}</p>
                <div className="flex gap-2">
                  <button
                    type="button"
                    disabled={busy}
                    className={buttonStyle("danger")}
                    onClick={() =>
                      void act(async () => {
                        await core.deleteLibraryGroup(selected.id);
                        setFilter("all");
                        setDeleting(false);
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
                close={() => setEditing(null)}
              />
            )}
            {(starters || groups.length === 0) && !editing && (
              <StarterGroups busy={busy} act={act} close={() => setStarters(false)} />
            )}
            <details className="mb-5 border-y border-rule py-3">
              <summary className="cursor-pointer font-semibold text-ink-secondary">
                {t("library.modelSettings")}
              </summary>
              <ClassifierForm
                key={JSON.stringify(snapshot.settings)}
                settings={snapshot.settings}
                act={act}
                busy={busy}
              />
            </details>
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
                disabled={
                  busy ||
                  pending > 0 ||
                  groups.length === 0 ||
                  !snapshot.settings.classifier ||
                  needsClassification.length === 0
                }
                className={buttonStyle("primary")}
                onClick={() =>
                  void act(() => core.classifyDocuments(needsClassification.map((doc) => doc.id)))
                }
              >
                {t("library.classify")}
              </button>
              {selected && (
                <button
                  type="button"
                  disabled={
                    busy || pending > 0 || !snapshot.settings.classifier || shown.length === 0
                  }
                  className={buttonStyle("secondary")}
                  onClick={() => void act(() => core.classifyDocuments(shown.map((doc) => doc.id)))}
                >
                  {t("library.reclassify")}
                </button>
              )}
            </div>
            <p role="status" className="mb-3 text-[13px] text-ink-meta">
              {pending
                ? t("library.progress", { count: pending })
                : waiting
                  ? t("library.waiting", { count: waiting })
                  : failed
                    ? t("library.failed", { count: failed })
                    : t(shown.length === 1 ? "library.count.one" : "library.count", {
                        count: shown.length,
                      })}
            </p>
            {!snapshot.settings.classifier && (
              <p className="mb-4 text-[13px] text-ink-secondary">{t("library.manualHint")}</p>
            )}
            {shown.length === 0 ? (
              <p className="border-t border-rule py-8 text-ink-meta">
                {t(documents.length === 0 ? "library.empty" : "library.noMatches")}
              </p>
            ) : (
              <ul className="border-t border-rule">
                {shown.slice(0, limit).map((doc) => {
                  const item = assignments.get(doc.id);
                  return (
                    <li
                      key={doc.id}
                      data-testid="library-document"
                      data-document-id={doc.id}
                      className="flex flex-wrap items-center gap-x-4 gap-y-2 border-b border-rule py-3"
                    >
                      <div className="min-w-28 flex-1">
                        <button
                          type="button"
                          title={doc.name}
                          className="block max-w-full truncate text-left font-semibold hover:underline focus-visible:outline-2 focus-visible:outline-accent"
                          onClick={() => openDocument({ documentId: doc.id })}
                        >
                          {doc.name}
                        </button>
                        <p className="mt-0.5 text-[12px] text-ink-meta">
                          {doc.kind.toUpperCase()} ·{" "}
                          {t(
                            item?.source === "user"
                              ? "library.manual"
                              : item?.status === "classified"
                                ? "library.saved"
                                : item?.status === "classifying"
                                  ? "library.working"
                                  : "library.notClassified",
                          )}
                        </p>
                        {item?.model && (
                          <p className="mt-0.5 text-[12px] text-ink-meta">
                            {item.model.id} ·{" "}
                            {t(item.model.images ? "library.usedPages" : "library.usedText")}
                            {["slow", "memory", "unavailable"].includes(item.model.reason) && (
                              <>
                                {" "}
                                ·{" "}
                                {t(
                                  `library.route.${item.model.reason}` as
                                    | "library.route.slow"
                                    | "library.route.memory"
                                    | "library.route.unavailable",
                                )}
                              </>
                            )}
                          </p>
                        )}
                        {item?.error && (
                          <p className={`${errorTextClass} mt-1`}>{item.error.message}</p>
                        )}
                      </div>
                      <select
                        aria-label={t("library.groupFor", { name: doc.name })}
                        disabled={busy}
                        value={item?.groupId ?? ""}
                        className={`${inputClass.replace("w-full", "")} mt-0 w-44 max-w-full`}
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
                    </li>
                  );
                })}
              </ul>
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
      )}
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

function ClassifierForm({
  settings,
  act,
  busy,
}: {
  settings: LibrarySettings;
  act: Action;
  busy: boolean;
}) {
  const t = useT();
  const [providers, setProviders] = useState<ChatProvider[]>([]);
  const initial = settings.classifier;
  const [selection, setSelection] = useState(
    initial?.kind === "chat" ? initial.choice.providerId : (initial?.kind ?? ""),
  );
  const [modelId, setModelId] = useState(
    initial?.kind === "chat"
      ? initial.choice.modelId
      : initial?.kind === "ollama"
        ? initial.modelId
        : "",
  );
  const [baseUrl, setBaseUrl] = useState(
    initial?.kind === "ollama" || initial?.kind === "auto"
      ? initial.baseUrl
      : "http://localhost:11434",
  );
  const [automatic, setAutomatic] = useState(settings.automatic);
  const [usePageImages, setUsePageImages] = useState(
    initial?.kind === "ollama" && initial.usePageImages === true,
  );
  const [jevEnabled, setJevEnabled] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);
  useEffect(() => {
    let stopped = false;
    const reload = () => {
      void Promise.all([core.listChatProviders(), core.getJevSettings()]).then(
        ([list, jev]) => {
          if (!stopped) {
            setProviders(list);
            setJevEnabled(jev.enabled);
          }
        },
        (failure) => {
          if (!stopped) setLoadError(messageOf(failure));
        },
      );
    };
    reload();
    const stopChat = core.on("chatReadiness.changed", reload);
    const stopJev = core.on("jev.changed", reload);
    return () => {
      stopped = true;
      stopChat();
      stopJev();
    };
  }, []);
  return (
    <form
      className="mt-4 max-w-xl"
      onSubmit={(event) => {
        event.preventDefault();
        let classifier: LibraryClassifier | null = null;
        if (selection === "jev") classifier = { kind: "jev" };
        else if (selection === "auto") classifier = { kind: "auto", baseUrl };
        else if (selection === "ollama")
          classifier = {
            kind: "ollama",
            baseUrl,
            modelId: modelId.trim(),
            usePageImages: supportsLibraryImages(modelId) && usePageImages,
          };
        else if (selection)
          classifier = { kind: "chat", choice: { providerId: selection, modelId: modelId.trim() } };
        void act(() =>
          core.saveLibrarySettings({ classifier, automatic: !!classifier && automatic }),
        );
      }}
    >
      <p className="mb-3 text-[13px] text-ink-meta">{t("library.modelHint")}</p>
      {loadError && (
        <p role="alert" className={errorTextClass}>
          {loadError}
        </p>
      )}
      <label className={fieldLabelClass}>
        {t("library.connection")}
        <select
          aria-label={t("library.connection")}
          className={inputClass}
          value={selection}
          onChange={(event) => {
            const value = event.target.value;
            setSelection(value);
            const current = useAppStore.getState().settings?.user.chatModel;
            setModelId(
              value === "ollama"
                ? "tev1:0.8b"
                : value === current?.providerId
                  ? current.modelId
                  : "",
            );
          }}
        >
          <option value="">{t("library.manualOnly")}</option>
          <option value="auto">{t("library.auto")}</option>
          {providers.map((provider) => (
            <option key={provider.id} value={provider.id}>
              {provider.service?.name ?? (provider.kind === "ollama" ? "Ollama" : provider.kind)}
              {provider.baseUrl ? ` · ${provider.baseUrl}` : ""}
            </option>
          ))}
          <option value="jev" disabled={!jevEnabled}>
            {t("library.jev")}
          </option>
          <option value="ollama">{t("library.ollama")}</option>
        </select>
      </label>
      {selection && selection !== "jev" && selection !== "auto" && (
        <label className={`${fieldLabelClass} mt-3`}>
          {t("library.model")}
          <input
            list="library-model-suggestions"
            className={inputClass}
            required
            maxLength={200}
            value={modelId}
            onChange={(event) => setModelId(event.target.value)}
          />
        </label>
      )}
      <datalist id="library-model-suggestions">
        {(selection === "ollama" ? ["tev1:0.8b", "clef-flash", "tev1:4b"] : []).map((model) => (
          <option key={model} value={model} />
        ))}
      </datalist>
      {(selection === "ollama" || selection === "auto") && (
        <>
          <label className={`${fieldLabelClass} mt-3`}>
            {t("library.server")}
            <input
              required
              type="url"
              className={inputClass}
              value={baseUrl}
              onChange={(event) => setBaseUrl(event.target.value)}
            />
          </label>
          <p className="mt-2 text-[13px] text-ink-meta">
            {t(selection === "auto" ? "library.autoHint" : "library.localHint")}
          </p>
          {selection === "auto" && (
            <p className="mt-2 text-[13px] text-ink-meta">{t("library.autoModels")}</p>
          )}
          {selection === "ollama" && supportsLibraryImages(modelId) && (
            <label className="mt-4 flex items-start gap-2 text-[13px] text-ink-secondary">
              <input
                type="checkbox"
                checked={usePageImages}
                onChange={(event) => setUsePageImages(event.target.checked)}
                className="mt-0.5 size-4 accent-ink"
              />
              {t("library.pageImages")}
            </label>
          )}
        </>
      )}
      {selection && (
        <label className="mt-4 flex items-start gap-2 text-[13px] text-ink-secondary">
          <input
            type="checkbox"
            checked={automatic}
            onChange={(event) => setAutomatic(event.target.checked)}
            className="mt-0.5 size-4 accent-ink"
          />
          {t("library.automatic")}
        </label>
      )}
      <div className="mt-4 flex flex-wrap gap-2">
        <button type="submit" disabled={busy} className={buttonStyle("primary")}>
          {t("library.saveModel")}
        </button>
        <button
          type="button"
          className={buttonStyle("ghost")}
          onClick={() => useAppStore.getState().openSettings("chat-model")}
        >
          {t("library.connectModel")}
        </button>
      </div>
    </form>
  );
}
