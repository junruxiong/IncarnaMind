import { NodeViewContent, NodeViewWrapper, type ReactNodeViewProps } from "@tiptap/react";
import { useState } from "react";
import {
  BLOCK_ID_ATTRIBUTE,
  type ChatModelChoice,
  type SearchScope,
  type SkillAvailability,
} from "../../../core/api";
import type { MessageKey } from "../../../shared/i18n";
import {
  hasSearchScope,
  SCOPE_ATTRIBUTES,
  type ScopeKind,
  scopeIds,
  searchScopeOf,
} from "../../../shared/searchScope";
import { useAnswers } from "../answers";
import { SkillIcon } from "../components/icons";
import { ReadinessExplanation, settingsPageFor } from "../components/providers/ChatReadinessNotice";
import { providerLabel } from "../components/providers/shared";
import { useT } from "../i18n";
import { scopeChips } from "../scope";
import { useAppStore } from "../store";
import { AskPlayIcon, ChevronDownSmallIcon, RemoveIcon } from "./icons";
import { useMindId } from "./mindContext";
import { askInEditor } from "./questionCommands";

const text = (value: unknown) => (typeof value === "string" ? value : null);

/**
 * A Question: a `frame` band whose text starts at the Mind's text edge. The
 * band reaches into the left margin, where the Ask button (a small square
 * play button) asks it, as Enter does; the picker on its right chooses its
 * model. A Skill forced from the slash menu shows as a chip beside its text,
 * and its Search scope, chosen by typing "@", as chips under it. When it
 * can't be asked, it says why underneath.
 */
export function QuestionView({ node, editor, updateAttributes }: ReactNodeViewProps) {
  const t = useT();
  const mindId = useMindId();
  const questionId = text(node.attrs[BLOCK_ID_ATTRIBUTE]);
  const blocked = useAnswers((state) => (questionId ? state.blocked[questionId] : undefined));
  const loadModels = useAnswers((state) => state.loadModels);
  const openSettings = useAppStore((state) => state.openSettings);
  const forcedSkill = text(node.attrs.forcedSkill);
  const skill = useAppStore((state) =>
    forcedSkill ? state.skills.find((each) => each.name === forcedSkill) : undefined,
  );
  const empty = node.content.size === 0;
  const canAsk = questionId !== null && node.textContent.trim() !== "";
  const dropSkill = () => {
    updateAttributes({ forcedSkill: null });
    if (questionId) useAnswers.getState().dismiss(questionId);
  };
  const scope = searchScopeOf(node.attrs);
  const removeFromScope = (kind: ScopeKind, id: string) => {
    const left = scopeIds(scope, kind).filter((each) => each !== id);
    updateAttributes({ [SCOPE_ATTRIBUTES[kind]]: left.length > 0 ? left : null });
  };

  return (
    <NodeViewWrapper
      className="question-block"
      data-testid="question"
      data-question-id={questionId ?? undefined}
      onPointerEnter={loadModels}
    >
      <button
        type="button"
        contentEditable={false}
        data-testid="question-ask"
        aria-label={t("question.askHint")}
        title={t("question.askHint")}
        disabled={!canAsk}
        // Keep the cursor in the editor.
        onMouseDown={(event) => event.preventDefault()}
        onClick={() => {
          if (canAsk) void askInEditor(editor, mindId, questionId);
        }}
        className="question-ask"
      >
        <AskPlayIcon className="size-3" />
      </button>
      <div className="question-row">
        {empty && (
          <span contentEditable={false} aria-hidden="true" className="question-placeholder">
            {t("question.placeholder")}
          </span>
        )}
        <NodeViewContent className="question-text" />
        {forcedSkill && (
          <SkillChip
            name={forcedSkill}
            state={!skill ? "removed" : skill.enabled ? "enabled" : "disabled"}
            onRemove={dropSkill}
          />
        )}
        <ModelPicker
          providerId={text(node.attrs.providerId)}
          modelId={text(node.attrs.modelId)}
          onFocus={loadModels}
          onChange={(choice) =>
            updateAttributes({
              providerId: choice?.providerId ?? null,
              modelId: choice?.modelId ?? null,
            })
          }
        />
      </div>
      {hasSearchScope(scope) && <ScopeChips scope={scope} onRemove={removeFromScope} />}
      {blocked?.kind === "not-ready" && (
        <p contentEditable={false} data-testid="question-not-ready" className="question-notice">
          <span className="question-notice-text">
            <ReadinessExplanation readiness={blocked.readiness} />
          </span>
          <button
            type="button"
            onClick={() => openSettings(settingsPageFor(blocked.readiness))}
            className="question-notice-action"
          >
            {t("question.setUp")}
          </button>
        </p>
      )}
      {blocked?.kind === "skill-unavailable" && (
        <p
          contentEditable={false}
          data-testid="question-skill-unavailable"
          className="question-notice"
        >
          <span className="question-notice-text">
            {t(`skills.unavailable.${blocked.state}`, { name: blocked.skill })}
          </span>
          <button type="button" onClick={() => openSettings()} className="question-notice-action">
            {t("skills.unavailable.settings")}
          </button>
          <button
            type="button"
            onClick={() => {
              dropSkill();
              if (canAsk) void askInEditor(editor, mindId, questionId);
            }}
            className="question-notice-action"
          >
            {t("skills.unavailable.drop")}
          </button>
        </p>
      )}
      {blocked?.kind === "error" && (
        <p contentEditable={false} role="alert" className="question-notice question-notice--error">
          <span className="question-notice-text">
            {t("error.action", { message: blocked.message })}
          </span>
        </p>
      )}
    </NodeViewWrapper>
  );
}

/**
 * The Skill forced on the Question, chosen in the slash menu, as a chip that
 * takes it off again. Muted, and saying why, when it is off or gone.
 */
function SkillChip({
  name,
  state,
  onRemove,
}: {
  name: string;
  state: SkillAvailability;
  onRemove(): void;
}) {
  const t = useT();
  const title =
    state === "enabled" ? t("skills.chip.label", { name }) : t(`skills.chip.${state}`, { name });
  return (
    <span
      contentEditable={false}
      data-testid="question-skill"
      data-state={state}
      title={title}
      className={`question-skill ${state === "enabled" ? "" : "question-skill--unavailable"}`}
    >
      <SkillIcon className="question-skill-icon" />
      <span className="truncate">{name}</span>
      <button
        type="button"
        data-testid="question-skill-remove"
        aria-label={t("skills.chip.remove", { name })}
        title={t("skills.chip.remove", { name })}
        // Keep the cursor in the editor.
        onMouseDown={(event) => event.preventDefault()}
        onClick={onRemove}
        className="scope-chip-remove"
      >
        <RemoveIcon className="size-2.5" />
      </button>
    </span>
  );
}

const CHIP_TITLES: Record<ScopeKind, MessageKey> = {
  folder: "scope.chip.folder",
  tag: "scope.chip.tag",
  document: "scope.chip.document",
};

const DELETED_NAMES: Record<ScopeKind, MessageKey> = {
  folder: "scope.chip.deleted.folder",
  tag: "scope.chip.deleted.tag",
  document: "scope.chip.deleted.document",
};

/** Beyond this many chips (the Documents a Library filter showed, say), the rest fold behind "+N more". */
const CHIPS_SHOWN = 8;

/**
 * The Question's Search scope: a chip for each Folder, Tag and Document (its
 * title says which), with a × that takes it out. One deleted since is struck
 * through: the search ignores it. A long scope shows its first chips and "+N more".
 */
function ScopeChips({
  scope,
  onRemove,
}: {
  scope: SearchScope;
  onRemove(kind: ScopeKind, id: string): void;
}) {
  const t = useT();
  const folders = useAppStore((state) => state.folders);
  const groups = useAppStore((state) => state.library?.groups);
  const tags = useAppStore((state) => state.tags);
  const documents = useAppStore((state) => state.documents);
  const chips = scopeChips({ folders, groups, tags, documents }, scope);
  const [expanded, setExpanded] = useState(false);
  const long = chips.length > CHIPS_SHOWN;
  const shown = long && !expanded ? chips.slice(0, CHIPS_SHOWN - 1) : chips;
  const more = chips.length - shown.length;
  return (
    <div contentEditable={false} data-testid="question-scope" className="question-scope">
      <span className="question-scope-label">{t("scope.label")}</span>
      <ul aria-label={t("scope.picker.label")} className="contents">
        {shown.map((chip) => {
          const deleted = chip.name === null;
          const name = chip.name ?? t(DELETED_NAMES[chip.kind]);
          const remove = t("scope.chip.remove", { name });
          return (
            <li
              key={`${chip.kind}:${chip.id}`}
              data-testid="scope-chip"
              data-kind={chip.kind}
              data-id={chip.id}
              data-deleted={deleted ? "true" : undefined}
              title={deleted ? t("scope.chip.deleted") : t(CHIP_TITLES[chip.kind], { name })}
              className={`scope-chip ${deleted ? "scope-chip--deleted" : ""}`}
            >
              {deleted ? (
                <>
                  <s className="truncate">{name}</s>
                  <span className="sr-only">{t("scope.chip.deleted")}</span>
                </>
              ) : (
                <span className="truncate">{name}</span>
              )}
              <button
                type="button"
                data-testid="scope-chip-remove"
                aria-label={remove}
                title={remove}
                // Keep the cursor in the editor.
                onMouseDown={(event) => event.preventDefault()}
                onClick={() => onRemove(chip.kind, chip.id)}
                className="scope-chip-remove"
              >
                <RemoveIcon className="size-2.5" />
              </button>
            </li>
          );
        })}
        {long && (
          <li>
            <button
              type="button"
              data-testid="scope-chips-more"
              aria-expanded={expanded}
              aria-label={expanded ? undefined : t("scope.chip.moreLabel", { count: more })}
              // Keep the cursor in the editor.
              onMouseDown={(event) => event.preventDefault()}
              onClick={() => setExpanded(!expanded)}
              className="scope-chip scope-chip-more"
            >
              {expanded ? t("scope.chip.fewer") : t("scope.chip.more", { count: more })}
            </button>
          </li>
        )}
      </ul>
    </div>
  );
}

const choiceValue = (providerId: string, modelId: string) => JSON.stringify([providerId, modelId]);

function parseChoice(value: string): ChatModelChoice | null {
  if (!value) return null;
  const [providerId, modelId] = JSON.parse(value) as [string, string];
  return { providerId, modelId };
}

/** The Question's model: the default, or one of the saved providers' models. */
function ModelPicker({
  providerId,
  modelId,
  onChange,
  onFocus,
}: {
  providerId: string | null;
  modelId: string | null;
  onChange(choice: ChatModelChoice | null): void;
  onFocus(): void;
}) {
  const t = useT();
  const groups = useAnswers((state) => state.models) ?? [];
  const chatModel = useAppStore((state) => state.settings?.user.chatModel ?? null);
  const picked = providerId && modelId ? choiceValue(providerId, modelId) : "";
  const listed = groups.some(
    (group) => group.provider.id === providerId && group.models.includes(modelId ?? ""),
  );
  // Nothing to choose between without a chat model.
  if (!chatModel && !picked) return null;

  return (
    <span contentEditable={false} className="question-model-picker">
      <select
        data-testid="question-model"
        aria-label={t("question.model.label")}
        title={t("question.model.label")}
        value={picked}
        onFocus={onFocus}
        onChange={(event) => onChange(parseChoice(event.target.value))}
        className="question-model"
      >
        <option value="">
          {chatModel
            ? t("question.model.default", { model: chatModel.modelId })
            : t("question.model.defaultUnset")}
        </option>
        {/* A model picked earlier that the provider no longer lists. */}
        {picked && !listed && <option value={picked}>{modelId}</option>}
        {groups
          .filter((group) => group.models.length > 0)
          .map((group) => (
            <optgroup key={group.provider.id} label={providerLabel(group.provider, t)}>
              {group.models.map((model) => (
                <option key={model} value={choiceValue(group.provider.id, model)}>
                  {model}
                </option>
              ))}
            </optgroup>
          ))}
      </select>
      <ChevronDownSmallIcon className="question-model-chevron" />
    </span>
  );
}
