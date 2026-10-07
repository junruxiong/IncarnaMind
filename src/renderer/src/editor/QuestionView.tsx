import { NodeViewContent, NodeViewWrapper, type ReactNodeViewProps } from "@tiptap/react";
import {
  BLOCK_ID_ATTRIBUTE,
  type ChatModelChoice,
  type SkillAvailability,
} from "../../../core/api";
import { useAnswers } from "../answers";
import { AskIcon, CloseIcon, SkillIcon } from "../components/icons";
import { ReadinessExplanation } from "../components/providers/ChatReadinessNotice";
import { providerLabel } from "../components/providers/shared";
import { useT } from "../i18n";
import { useAppStore } from "../store";
import { useMindId } from "./mindContext";
import { askInEditor } from "./questionCommands";

const text = (value: unknown) => (typeof value === "string" ? value : null);

/**
 * A Question, as the old editor's query block: the button on its left (or
 * Enter) asks it, and the picker on its right chooses its model. When it can't
 * be asked, it says why underneath.
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
        aria-label={t("question.ask")}
        title={t("question.askHint")}
        disabled={!canAsk}
        // Keep the cursor in the editor.
        onMouseDown={(event) => event.preventDefault()}
        onClick={() => {
          if (canAsk) void askInEditor(editor, mindId, questionId);
        }}
        className="question-ask"
      >
        <AskIcon className="size-4" />
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
      {blocked?.kind === "not-ready" && (
        <p contentEditable={false} data-testid="question-not-ready" className="question-notice">
          <span className="flex-1">
            <ReadinessExplanation readiness={blocked.readiness} />
          </span>
          <button type="button" onClick={openSettings} className="question-notice-action">
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
          <span className="flex-1">
            {t(`skills.unavailable.${blocked.state}`, { name: blocked.skill })}
          </span>
          <button type="button" onClick={openSettings} className="question-notice-action">
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
        <p contentEditable={false} role="alert" className="question-notice">
          {t("error.action", { message: blocked.message })}
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
      <SkillIcon className="size-3.5 shrink-0" />
      <span className="truncate">{name}</span>
      <button
        type="button"
        data-testid="question-skill-remove"
        aria-label={t("skills.chip.remove", { name })}
        title={t("skills.chip.remove", { name })}
        // Keep the cursor in the editor.
        onMouseDown={(event) => event.preventDefault()}
        onClick={onRemove}
        className="question-skill-remove"
      >
        <CloseIcon className="size-3" />
      </button>
    </span>
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
    <select
      contentEditable={false}
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
  );
}
