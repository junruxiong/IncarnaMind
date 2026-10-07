import { NodeViewContent, NodeViewWrapper, type ReactNodeViewProps } from "@tiptap/react";
import { BLOCK_ID_ATTRIBUTE, type ChatModelChoice, type SearchScope } from "../../../core/api";
import type { MessageKey } from "../../../shared/i18n";
import {
  hasSearchScope,
  SCOPE_ATTRIBUTES,
  type ScopeKind,
  scopeIds,
  searchScopeOf,
} from "../../../shared/searchScope";
import { useAnswers } from "../answers";
import { AskIcon, CloseIcon, DocumentIcon, FolderIcon, TagIcon } from "../components/icons";
import { ReadinessExplanation, settingsPageFor } from "../components/providers/ChatReadinessNotice";
import { providerLabel } from "../components/providers/shared";
import { useT } from "../i18n";
import { type ScopeChip, scopeChips } from "../scope";
import { useAppStore } from "../store";
import { useMindId } from "./mindContext";
import { askInEditor } from "./questionCommands";

const text = (value: unknown) => (typeof value === "string" ? value : null);

/**
 * A Question, as the old editor's query block: the button on its left (or
 * Enter) asks it, and the picker on its right chooses its model. Its Search
 * scope, chosen by typing "@", shows as chips under its text. When it can't
 * be asked, it says why underneath.
 */
export function QuestionView({ node, editor, updateAttributes }: ReactNodeViewProps) {
  const t = useT();
  const mindId = useMindId();
  const questionId = text(node.attrs[BLOCK_ID_ATTRIBUTE]);
  const blocked = useAnswers((state) => (questionId ? state.blocked[questionId] : undefined));
  const loadModels = useAnswers((state) => state.loadModels);
  const openSettings = useAppStore((state) => state.openSettings);
  const empty = node.content.size === 0;
  const canAsk = questionId !== null && node.textContent.trim() !== "";
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
          <span className="flex-1">
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
      {blocked?.kind === "error" && (
        <p contentEditable={false} role="alert" className="question-notice">
          {t("error.action", { message: blocked.message })}
        </p>
      )}
    </NodeViewWrapper>
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

function ChipIcon({ chip }: { chip: ScopeChip }) {
  if (chip.kind === "folder") return <FolderIcon className="size-3.5" />;
  if (chip.kind === "tag") return <TagIcon className="size-3.5" />;
  return <DocumentIcon kind={chip.documentKind ?? "text"} className="size-3.5" />;
}

/**
 * The Question's Search scope: a chip for each Folder, Tag and Document, with
 * a × that takes it out. One deleted since is struck through: the search ignores it.
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
  const tags = useAppStore((state) => state.tags);
  const documents = useAppStore((state) => state.documents);
  const chips = scopeChips({ folders, tags, documents }, scope);
  return (
    <div contentEditable={false} data-testid="question-scope" className="question-scope">
      <span className="question-scope-label">{t("scope.label")}</span>
      <ul aria-label={t("scope.picker.label")} className="contents">
        {chips.map((chip) => {
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
              <ChipIcon chip={chip} />
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
                <CloseIcon className="size-[10px]" />
              </button>
            </li>
          );
        })}
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
