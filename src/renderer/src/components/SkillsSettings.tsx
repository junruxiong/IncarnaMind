import { useEffect, useId, useRef, useState } from "react";
import {
  SKILL_LIMITS,
  SKILL_SCRIPT_LIMITS,
  type Skill,
  type SkillImportError,
  type SkillImportPreview,
} from "../../../core/api";
import type { SkillPickKind } from "../../../shared/bridge";
import { core } from "../core";
import { errorMessage } from "../errors";
import { useT } from "../i18n";
import { formatSize, pickSkill, skillDescription } from "../skills";
import { useAppStore } from "../store";
import { DuplicateLineIcon, TrashLineIcon } from "./lineIcons";
import {
  buttonClass,
  errorTextClass,
  ghostButtonClass,
  iconButtonClass,
  noticeClass,
  pageIntroClass,
  primaryButtonClass,
  rowStatusClass,
  rowTextClass,
  rowTitleClass,
  ruledListClass,
  ruledRow,
  ruledRowClass,
  sectionTitleClass,
} from "./ui";

/**
 * Settings → Skills: the Skills there are, each turned on or off or removed,
 * and importing one from a folder or a zip, shown first for the User to check.
 * Built-in Skills are labelled, can be duplicated as the User's own, and,
 * once removed, restored.
 */
export function SkillsSettings() {
  const t = useT();
  const skills = useAppStore((state) => state.skills);
  const restoreBuiltInSkills = useAppStore((state) => state.restoreBuiltInSkills);
  const removedBuiltIns = useRemovedBuiltIns();
  const [busy, setBusy] = useState<"reading" | "importing" | null>(null);
  const [preview, setPreview] = useState<SkillImportPreview | null>(null);
  const [error, setError] = useState<SkillImportError | string | null>(null);
  // A preview left open when Settings closes is forgotten.
  const pending = useRef<string | null>(null);
  pending.current = preview?.importId ?? null;
  useEffect(
    () => () => {
      if (pending.current) core.cancelSkillImport(pending.current).catch(() => undefined);
    },
    [],
  );

  const choose = async (kind: SkillPickKind) => {
    setError(null);
    let path: string | null;
    try {
      path = await pickSkill(kind);
    } catch (failure) {
      setError(errorMessage(failure));
      return;
    }
    if (!path) return;
    if (preview) core.cancelSkillImport(preview.importId).catch(() => undefined);
    setPreview(null);
    setBusy("reading");
    try {
      const check = await core.previewSkillImport(path);
      if (check.ok) setPreview(check.preview);
      else setError(check.error);
    } catch (failure) {
      setError(errorMessage(failure));
    } finally {
      setBusy(null);
    }
  };

  const confirm = async (importId: string) => {
    setBusy("importing");
    try {
      await core.importSkill(importId);
      setPreview(null);
    } catch (failure) {
      setPreview(null);
      setError(errorMessage(failure));
    } finally {
      setBusy(null);
    }
  };

  const cancel = (importId: string) => {
    core.cancelSkillImport(importId).catch(() => undefined);
    setPreview(null);
  };

  return (
    <section data-testid="skills-settings" className="flex flex-col gap-4">
      <p className={pageIntroClass}>{t("skills.settings.body")}</p>

      {skills.length === 0 ? (
        <p className={`${rowStatusClass} border-y border-rule py-3.5`}>
          {t("skills.settings.empty")}
        </p>
      ) : (
        <ul className={ruledListClass}>
          {skills.map((skill) => (
            <SkillRow key={skill.id} skill={skill} />
          ))}
        </ul>
      )}

      {preview ? (
        <SkillPreview
          preview={preview}
          importing={busy === "importing"}
          onImport={() => void confirm(preview.importId)}
          onCancel={() => cancel(preview.importId)}
        />
      ) : (
        <div className="flex flex-wrap gap-2">
          <button
            type="button"
            data-testid="skill-import-folder"
            disabled={busy !== null}
            onClick={() => void choose("folder")}
            className={buttonClass}
          >
            {t("skills.import.folder")}
          </button>
          <button
            type="button"
            data-testid="skill-import-zip"
            disabled={busy !== null}
            onClick={() => void choose("zip")}
            className={buttonClass}
          >
            {t("skills.import.zip")}
          </button>
          {removedBuiltIns.length > 0 && (
            <button
              type="button"
              data-testid="skill-restore-built-ins"
              title={t("skills.settings.restoreBuiltIns.hint", {
                names: removedBuiltIns.join(", "),
              })}
              disabled={busy !== null}
              onClick={() => void restoreBuiltInSkills()}
              className={buttonClass}
            >
              {t("skills.settings.restoreBuiltIns")}
            </button>
          )}
        </div>
      )}
      {busy === "reading" && (
        <p className="text-[13px] leading-5 text-ink-meta">{t("skills.import.reading")}</p>
      )}
      {error && <ImportError error={error} />}
    </section>
  );
}

/**
 * Settings → Skill scripts: the switch that lets Skills run their scripts at
 * all (each run still asks, unless the Skill's scripts always run), and how
 * long a script may run before it is stopped. Both belong to this device.
 */
export function SkillScriptsSettings() {
  const t = useT();
  const enabled = useAppStore((state) => state.settings?.device.skillScriptsEnabled);
  const timeout = useAppStore((state) => state.settings?.device.skillScriptTimeoutSeconds);
  const updateSettings = useAppStore((state) => state.updateSettings);
  const [draft, setDraft] = useState<string | null>(null);
  const timeoutId = useId();
  const enabledId = useId();
  if (enabled === undefined || timeout === undefined) return null;

  const { minTimeoutSeconds: min, maxTimeoutSeconds: max } = SKILL_SCRIPT_LIMITS;
  const commit = () => {
    if (draft === null) return;
    const seconds = Number(draft);
    setDraft(null);
    if (Number.isInteger(seconds) && seconds >= min && seconds <= max && seconds !== timeout) {
      void updateSettings({ device: { skillScriptTimeoutSeconds: seconds } });
    }
  };

  return (
    <section data-testid="skill-scripts-settings" className="flex flex-col">
      <h4 className={`mb-2 ${sectionTitleClass}`}>{t("scripts.settings.title")}</h4>
      <div className={ruledListClass}>
        <div className={ruledRowClass}>
          <div className="flex min-w-0 flex-col gap-0.5">
            <label htmlFor={enabledId} className={rowTitleClass}>
              {t("scripts.settings.enabled")}
            </label>
            <p id={`${enabledId}-description`} className={rowTextClass}>
              {t("scripts.settings.body")}
            </p>
          </div>
          <input
            id={enabledId}
            type="checkbox"
            role="switch"
            aria-checked={enabled}
            aria-describedby={`${enabledId}-description`}
            data-testid="skill-scripts-enabled"
            checked={enabled}
            onChange={(event) =>
              void updateSettings({ device: { skillScriptsEnabled: event.target.checked } })
            }
            className="switch"
          />
        </div>
        <div className={ruledRow("center")}>
          <label
            htmlFor={timeoutId}
            className={enabled ? "text-ui text-ink" : "text-ui text-ink-placeholder"}
          >
            {t("scripts.settings.timeout")}
          </label>
          <span className="flex items-center gap-2 text-[13px] leading-5 text-ink-meta">
            <input
              id={timeoutId}
              type="number"
              data-testid="skill-scripts-timeout"
              min={min}
              max={max}
              step={1}
              disabled={!enabled}
              value={draft ?? String(timeout)}
              onChange={(event) => setDraft(event.target.value)}
              onBlur={commit}
              onKeyDown={(event) => {
                if (event.key === "Enter") commit();
              }}
              className="h-8 w-20 rounded-sm border border-rule-strong bg-sheet px-2 text-ui text-ink focus:border-accent focus:outline-1 focus:outline-offset-0 focus:outline-accent disabled:opacity-60"
            />
            {t("scripts.settings.seconds")}
          </span>
        </div>
      </div>
    </section>
  );
}

/**
 * The names of the built-in Skills the User removed, asked again whenever the
 * Skills change (a removal, a restore). Empty until the core answers.
 */
function useRemovedBuiltIns(): string[] {
  const [removed, setRemoved] = useState<string[]>([]);
  useEffect(() => {
    let current = true;
    const refresh = () => {
      core
        .listRemovedBuiltInSkills()
        .then((names) => {
          if (current) setRemoved(names);
        })
        .catch(() => undefined);
    };
    refresh();
    const stop = core.on("skills.changed", refresh);
    return () => {
      current = false;
      stop();
    };
  }, []);
  return removed;
}

/** One Skill: on or off, what it's for, its licence and scripts, and removing it. */
function SkillRow({ skill }: { skill: Skill }) {
  const t = useT();
  const setSkillEnabled = useAppStore((state) => state.setSkillEnabled);
  const removeSkill = useAppStore((state) => state.removeSkill);
  const duplicateSkill = useAppStore((state) => state.duplicateSkill);
  const scripts = skill.files.filter((file) => file.script).length;
  const details = [
    skill.license && t("skills.settings.license", { license: skill.license }),
    scripts === 1
      ? t("skills.settings.scripts.one")
      : scripts > 1
        ? t("skills.settings.scripts", { count: scripts })
        : null,
  ].filter(Boolean);

  return (
    <li
      data-testid="skill-item"
      data-skill-name={skill.name}
      data-enabled={skill.enabled}
      data-built-in={skill.builtIn}
      className={ruledRowClass}
    >
      <div className={`flex min-w-0 flex-col gap-0.5 ${skill.enabled ? "" : "opacity-60"}`}>
        <p className="flex min-w-0 items-baseline gap-2">
          <span className={`truncate ${rowTitleClass}`}>{skill.name}</span>
          {skill.builtIn && (
            <span
              data-testid="skill-built-in"
              title={t("skills.settings.builtIn.hint")}
              className="shrink-0 text-label font-semibold text-ink-meta"
            >
              {t("skills.settings.builtIn")}
            </span>
          )}
        </p>
        <p className={`line-clamp-2 ${rowTextClass}`} title={skillDescription(skill, t)}>
          {skillDescription(skill, t)}
        </p>
        {details.length > 0 && (
          <p className="text-[12px] leading-[18px] text-ink-meta">{details.join(" · ")}</p>
        )}
      </div>
      {/* 28px buttons and the switch, centred on the name's 20px line. */}
      <div className="-my-1 flex items-center gap-1">
        {skill.builtIn && (
          <button
            type="button"
            data-testid="skill-duplicate"
            aria-label={t("skills.settings.duplicateLabel", { name: skill.name })}
            title={t("skills.settings.duplicate")}
            onClick={() => void duplicateSkill(skill.id)}
            className={iconButtonClass}
          >
            <DuplicateLineIcon className="size-4" />
          </button>
        )}
        <button
          type="button"
          data-testid="skill-remove"
          aria-label={t("skills.settings.removeLabel", { name: skill.name })}
          title={t("skills.settings.remove")}
          onClick={() => void removeSkill(skill.id)}
          className={iconButtonClass}
        >
          <TrashLineIcon className="size-4" />
        </button>
        <input
          type="checkbox"
          role="switch"
          aria-checked={skill.enabled}
          data-testid="skill-enabled"
          aria-label={t("skills.settings.enabled", { name: skill.name })}
          checked={skill.enabled}
          onChange={(event) => void setSkillEnabled(skill.id, event.target.checked)}
          className="switch ml-2"
        />
      </div>
    </li>
  );
}

/** What importing would add: the Skill's frontmatter and its files, to import or cancel. */
function SkillPreview({
  preview,
  importing,
  onImport,
  onCancel,
}: {
  preview: SkillImportPreview;
  importing: boolean;
  onImport(): void;
  onCancel(): void;
}) {
  const t = useT();
  const size = formatSize(preview.totalBytes, t);
  return (
    <div
      data-testid="skill-preview"
      className="flex flex-col gap-3 rounded-lg border border-rule p-4 text-ui text-ink"
    >
      <p className="text-label font-semibold text-ink-meta">{t("skills.preview.title")}</p>
      <div className="flex flex-col gap-0.5">
        <p data-testid="skill-preview-name" className={rowTitleClass}>
          {preview.name}
        </p>
        <p data-testid="skill-preview-description" className={rowTextClass}>
          {preview.description}
        </p>
        {preview.license && (
          <p className="text-[12px] leading-[18px] text-ink-meta">
            {t("skills.preview.license", { license: preview.license })}
          </p>
        )}
        {preview.compatibility && (
          <p className="text-[12px] leading-[18px] text-ink-meta">
            {t("skills.preview.compatibility", { compatibility: preview.compatibility })}
          </p>
        )}
      </div>
      {preview.replaces && (
        <p data-testid="skill-preview-replaces" className={noticeClass}>
          {t("skills.preview.replaces", { name: preview.replaces.name })}
        </p>
      )}
      <details>
        <summary className="cursor-pointer text-[13px] leading-5 text-ink-secondary select-none hover:text-ink">
          {preview.files.length === 1
            ? t("skills.preview.files.one", { size })
            : t("skills.preview.files", { count: preview.files.length, size })}
        </summary>
        <ul className="mt-1 max-h-40 overflow-y-auto font-mono text-[12px] leading-5 text-ink-secondary">
          {preview.files.map((file) => (
            <li key={file.path} data-testid="skill-preview-file" className="flex gap-2">
              <span className="truncate">{file.path}</span>
              <span className="shrink-0 text-ink-meta">{formatSize(file.size, t)}</span>
              {file.script && (
                <span className="shrink-0 font-sans font-semibold text-ink">
                  {t("skills.preview.script")}
                </span>
              )}
            </li>
          ))}
        </ul>
      </details>
      <div className="flex gap-2">
        <button
          type="button"
          data-testid="skill-import-confirm"
          disabled={importing}
          onClick={onImport}
          className={primaryButtonClass}
        >
          {importing ? t("skills.preview.importing") : t("skills.preview.import")}
        </button>
        <button type="button" disabled={importing} onClick={onCancel} className={ghostButtonClass}>
          {t("skills.preview.cancel")}
        </button>
      </div>
    </div>
  );
}

/** Why a Skill can't be imported, in words, with the technical detail underneath. */
function ImportError({ error }: { error: SkillImportError | string }) {
  const t = useT();
  if (typeof error === "string") {
    return (
      <p role="alert" data-testid="skill-import-error" className={errorTextClass}>
        {t("error.action", { message: error })}
      </p>
    );
  }
  const path = error.path ?? "";
  const message =
    error.kind === "too-large"
      ? t("skills.error.too-large", {
          size: formatSize(SKILL_LIMITS.maxBytes, t),
          instructions: formatSize(SKILL_LIMITS.maxInstructionsBytes, t),
        })
      : error.kind === "too-many-files"
        ? t("skills.error.too-many-files", { count: SKILL_LIMITS.maxFiles })
        : t(`skills.error.${error.kind}`, { path });
  return (
    <div
      role="alert"
      data-testid="skill-import-error"
      data-kind={error.kind}
      className="flex flex-col gap-1 text-[13px] leading-5 text-danger"
    >
      <p>{message}</p>
      <details className="text-[12px] text-ink-meta">
        <summary className="cursor-pointer">{t("skills.error.details")}</summary>
        <p className="mt-1 break-words">{error.message}</p>
      </details>
    </div>
  );
}
