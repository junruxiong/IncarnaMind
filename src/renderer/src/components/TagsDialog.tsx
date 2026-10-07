import { type FormEvent, useEffect, useState } from "react";
import type { Tag } from "../../../core/api";
import { core } from "../core";
import { errorMessage } from "../errors";
import { useT } from "../i18n";
import { useAppStore } from "../store";
import { PencilIcon, TrashIcon } from "./icons";
import { buttonClass, inputClass, primaryButtonClass } from "./providers/shared";
import { useModal } from "./useModal";

/** Runs a change, showing its failure in the dialog. Resolves with whether it worked. */
type Run = (action: () => Promise<unknown>) => Promise<boolean>;

const iconButton = "rounded-[6px] p-[3px] text-gray-500 hover:bg-gray-100 hover:text-gray-700";

/**
 * Managing Tags, in a native modal <dialog>: each Tag's name and description,
 * to edit or delete, a form for a new one, and "Re-tag all Documents". The
 * list follows the core's "tags.changed" event.
 */
export function TagsDialog() {
  const t = useT();
  const open = useAppStore((state) => state.tagsDialogOpen);
  const close = useAppStore((state) => state.closeTagsDialog);
  const tags = useAppStore((state) => state.tags);
  const dialog = useModal(open);
  const [error, setError] = useState<string | null>(null);
  const [retagStarted, setRetagStarted] = useState(false);

  useEffect(() => {
    if (!open) return;
    setError(null);
    setRetagStarted(false);
  }, [open]);

  const run: Run = async (action) => {
    setError(null);
    try {
      await action();
      return true;
    } catch (failure) {
      setError(errorMessage(failure));
      return false;
    }
  };

  return (
    <dialog
      ref={dialog}
      onClose={close}
      data-testid="tags-dialog"
      aria-labelledby="tags-title"
      className="m-auto max-h-[90vh] w-[34rem] max-w-[calc(100vw-2rem)] overflow-y-auto rounded-[9px] bg-white p-5 text-gray-800 shadow-custom-focus backdrop:bg-black/20"
    >
      <h2 id="tags-title" className="text-lg font-semibold">
        {t("tags.title")}
      </h2>
      {/* Rendered only while open, so forms start empty each time. */}
      {open && (
        <>
          <p className="mt-1 text-sm text-gray-600">{t("tags.dialog.body")}</p>
          {tags.length === 0 ? (
            <p className="mt-3 text-sm text-gray-500">{t("tags.dialog.empty")}</p>
          ) : (
            <ul className="mt-3 divide-y divide-gray-100 border-y border-gray-100">
              {tags.map((tag) => (
                <TagRow key={tag.id} tag={tag} run={run} />
              ))}
            </ul>
          )}
          <NewTag run={run} />
          {error && (
            <p role="alert" className="mt-2 text-sm text-red-700">
              {error}
            </p>
          )}
          <section className="mt-5 border-t border-gray-100 pt-4">
            <button
              type="button"
              data-testid="retag-all"
              onClick={async () => {
                if (await run(() => core.retagDocuments())) setRetagStarted(true);
              }}
              className={buttonClass}
            >
              {t("tags.dialog.retag")}
            </button>
            <p className="mt-1 text-xs text-gray-500" role={retagStarted ? "status" : undefined}>
              {t(retagStarted ? "tags.dialog.retag.started" : "tags.dialog.retag.hint")}
            </p>
          </section>
        </>
      )}
      <div className="mt-5 flex justify-end">
        <button type="button" onClick={close} className={buttonClass}>
          {t("tags.dialog.done")}
        </button>
      </div>
    </dialog>
  );
}

function TagRow({ tag, run }: { tag: Tag; run: Run }) {
  const t = useT();
  const [mode, setMode] = useState<"view" | "edit" | "delete">("view");

  if (mode === "edit") {
    return (
      <li data-testid="tag-row" data-tag-id={tag.id} className="py-2">
        <TagForm
          initial={tag}
          submitLabel={t("tags.dialog.save")}
          onSubmit={async (name, description) => {
            if (await run(() => core.updateTag(tag.id, { name, description }))) setMode("view");
          }}
          onCancel={() => setMode("view")}
        />
      </li>
    );
  }

  if (mode === "delete") {
    return (
      <li
        data-testid="tag-row"
        data-tag-id={tag.id}
        className="flex items-center gap-2 py-2 text-sm"
      >
        <p className="min-w-0 flex-1 break-words">
          {t("tags.dialog.delete.body", { name: tag.name })}
        </p>
        <button type="button" onClick={() => setMode("view")} className={buttonClass}>
          {t("tags.dialog.cancel")}
        </button>
        <button
          type="button"
          data-testid="confirm-delete-tag"
          onClick={() => void run(() => core.deleteTag(tag.id))}
          className="rounded-[9px] bg-red-600 px-4 py-2 text-sm text-white hover:bg-red-700"
        >
          {t("tags.dialog.delete.confirm")}
        </button>
      </li>
    );
  }

  return (
    <li data-testid="tag-row" data-tag-id={tag.id} className="flex items-start gap-2 py-2">
      <div className="min-w-0 flex-1">
        <p className="text-sm font-medium break-words">
          {tag.name}
          {tag.preset && (
            <span className="ml-2 rounded-[4px] bg-gray-100 px-1 text-[11px] font-normal text-gray-500">
              {t("tags.dialog.preset")}
            </span>
          )}
        </p>
        {tag.description && <p className="text-xs break-words text-gray-500">{tag.description}</p>}
      </div>
      <button
        type="button"
        data-testid="edit-tag"
        aria-label={t("tags.dialog.edit", { name: tag.name })}
        title={t("tags.dialog.edit", { name: tag.name })}
        onClick={() => setMode("edit")}
        className={iconButton}
      >
        <PencilIcon className="size-[14px]" />
      </button>
      <button
        type="button"
        data-testid="delete-tag"
        aria-label={t("tags.dialog.delete", { name: tag.name })}
        title={t("tags.dialog.delete", { name: tag.name })}
        onClick={() => setMode("delete")}
        className={iconButton}
      >
        <TrashIcon className="size-[14px]" />
      </button>
    </li>
  );
}

/** The form for a new Tag; it empties after each Tag added. */
function NewTag({ run }: { run: Run }) {
  const t = useT();
  const [added, setAdded] = useState(0);
  return (
    <section className="mt-4">
      <h3 className="mb-1 text-sm font-medium">{t("tags.dialog.new")}</h3>
      <TagForm
        key={added}
        initial={{ name: "", description: "" }}
        submitLabel={t("tags.dialog.create")}
        onSubmit={async (name, description) => {
          if (await run(() => core.createTag({ name, description }))) setAdded((n) => n + 1);
        }}
      />
    </section>
  );
}

function TagForm(props: {
  initial: { name: string; description: string };
  submitLabel: string;
  onSubmit(name: string, description: string): Promise<void>;
  onCancel?(): void;
}) {
  const { initial, submitLabel, onSubmit, onCancel } = props;
  const t = useT();
  const [name, setName] = useState(initial.name);
  const [description, setDescription] = useState(initial.description);
  const [busy, setBusy] = useState(false);

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    if (!name.trim() || busy) return;
    setBusy(true);
    try {
      await onSubmit(name, description);
    } finally {
      setBusy(false);
    }
  };

  return (
    <form onSubmit={(event) => void submit(event)} className="flex flex-col gap-2">
      <label className="text-xs text-gray-600">
        {t("tags.dialog.name")}
        <input
          value={name}
          data-testid="tag-name-input"
          maxLength={100}
          onChange={(event) => setName(event.target.value)}
          className={inputClass}
        />
      </label>
      <label className="text-xs text-gray-600">
        {t("tags.dialog.description")}
        <textarea
          value={description}
          data-testid="tag-description-input"
          maxLength={500}
          rows={2}
          placeholder={t("tags.dialog.descriptionPlaceholder")}
          onChange={(event) => setDescription(event.target.value)}
          className={`${inputClass} resize-y`}
        />
      </label>
      <div className="flex justify-end gap-2">
        {onCancel && (
          <button type="button" onClick={onCancel} className={buttonClass}>
            {t("tags.dialog.cancel")}
          </button>
        )}
        <button
          type="submit"
          data-testid="save-tag"
          disabled={!name.trim() || busy}
          className={primaryButtonClass}
        >
          {submitLabel}
        </button>
      </div>
    </form>
  );
}
