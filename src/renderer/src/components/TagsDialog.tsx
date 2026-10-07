import { type FormEvent, useEffect, useState } from "react";
import type { Tag } from "../../../core/api";
import { core } from "../core";
import { errorMessage } from "../errors";
import { useT } from "../i18n";
import { useAppStore } from "../store";
import { PencilLineIcon, TrashLineIcon } from "./lineIcons";
import {
  buttonClass,
  dangerButtonClass,
  dialogActionsClass,
  dialogClass,
  dialogTextClass,
  dialogTitleClass,
  errorTextClass,
  fieldLabelClass,
  ghostButtonClass,
  hintClass,
  iconButtonClass,
  inputClass,
  primaryButtonClass,
  ruledListClass,
  sectionTitleClass,
} from "./ui";
import { useModal } from "./useModal";

/** Runs a change, showing its failure in the dialog. Resolves with whether it worked. */
type Run = (action: () => Promise<unknown>) => Promise<boolean>;

/**
 * Managing Tags, in a native modal <dialog>: each Tag's name and description,
 * in rows split by rules, to edit or delete, a form for a new one, and
 * "Re-tag all Documents". The list follows the core's "tags.changed" event.
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
      className={`${dialogClass} w-[36rem]`}
    >
      <div className="flex flex-col gap-5 px-6 pt-6 pb-5">
        <div className="flex flex-col gap-2">
          <h2 id="tags-title" className={dialogTitleClass}>
            {t("tags.title")}
          </h2>
          {open && <p className={dialogTextClass}>{t("tags.dialog.body")}</p>}
        </div>
        {/* Rendered only while open, so forms start empty each time. */}
        {open && (
          <>
            {tags.length === 0 ? (
              <p className="text-[13px] leading-5 text-ink-meta">{t("tags.dialog.empty")}</p>
            ) : (
              <ul className={ruledListClass}>
                {tags.map((tag) => (
                  <TagRow key={tag.id} tag={tag} run={run} />
                ))}
              </ul>
            )}
            <NewTag run={run} />
            {error && (
              <p role="alert" className={errorTextClass}>
                {error}
              </p>
            )}
            <section className="flex flex-col items-start gap-1 border-t border-rule pt-4">
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
              <p className={hintClass} role={retagStarted ? "status" : undefined}>
                {t(retagStarted ? "tags.dialog.retag.started" : "tags.dialog.retag.hint")}
              </p>
            </section>
          </>
        )}
        <div className={dialogActionsClass}>
          <button type="button" onClick={close} className={primaryButtonClass}>
            {t("tags.dialog.done")}
          </button>
        </div>
      </div>
    </dialog>
  );
}

function TagRow({ tag, run }: { tag: Tag; run: Run }) {
  const t = useT();
  const [mode, setMode] = useState<"view" | "edit" | "delete">("view");

  if (mode === "edit") {
    return (
      <li data-testid="tag-row" data-tag-id={tag.id} className="py-3">
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
      <li data-testid="tag-row" data-tag-id={tag.id} className="flex items-center gap-2 py-2.5">
        <p className="min-w-0 flex-1 text-ui break-words text-ink">
          {t("tags.dialog.delete.body", { name: tag.name })}
        </p>
        <button type="button" onClick={() => setMode("view")} className={buttonClass}>
          {t("tags.dialog.cancel")}
        </button>
        <button
          type="button"
          data-testid="confirm-delete-tag"
          onClick={() => void run(() => core.deleteTag(tag.id))}
          className={dangerButtonClass}
        >
          {t("tags.dialog.delete.confirm")}
        </button>
      </li>
    );
  }

  return (
    <li data-testid="tag-row" data-tag-id={tag.id} className="flex items-start gap-2 py-2.5">
      <div className="min-w-0 flex-1">
        <p className="flex flex-wrap items-baseline gap-x-2 text-ui font-semibold break-words text-ink">
          {tag.name}
          {tag.preset && (
            <span className="text-label font-semibold text-ink-meta">
              {t("tags.dialog.preset")}
            </span>
          )}
        </p>
        {tag.description && (
          <p className="text-[13px] leading-5 break-words text-ink-secondary">{tag.description}</p>
        )}
      </div>
      <button
        type="button"
        data-testid="edit-tag"
        aria-label={t("tags.dialog.edit", { name: tag.name })}
        title={t("tags.dialog.edit", { name: tag.name })}
        onClick={() => setMode("edit")}
        className={iconButtonClass}
      >
        <PencilLineIcon className="size-4" />
      </button>
      <button
        type="button"
        data-testid="delete-tag"
        aria-label={t("tags.dialog.delete", { name: tag.name })}
        title={t("tags.dialog.delete", { name: tag.name })}
        onClick={() => setMode("delete")}
        className={iconButtonClass}
      >
        <TrashLineIcon className="size-4" />
      </button>
    </li>
  );
}

/** The form for a new Tag; it empties after each Tag added. */
function NewTag({ run }: { run: Run }) {
  const t = useT();
  const [added, setAdded] = useState(0);
  return (
    <section className="flex flex-col gap-2">
      <h3 className={sectionTitleClass}>{t("tags.dialog.new")}</h3>
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
    <form onSubmit={(event) => void submit(event)} className="flex flex-col gap-3">
      <label className={fieldLabelClass}>
        {t("tags.dialog.name")}
        <input
          value={name}
          data-testid="tag-name-input"
          maxLength={100}
          onChange={(event) => setName(event.target.value)}
          className={inputClass}
        />
      </label>
      <label className={fieldLabelClass}>
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
          <button type="button" onClick={onCancel} className={ghostButtonClass}>
            {t("tags.dialog.cancel")}
          </button>
        )}
        <button
          type="submit"
          data-testid="save-tag"
          disabled={!name.trim() || busy}
          className={buttonClass}
        >
          {submitLabel}
        </button>
      </div>
    </form>
  );
}
