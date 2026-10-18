import { useEffect, useId, useState } from "react";
import type {
  ChatModelChoice,
  ChatModelGroup,
  ModelRoles,
  UserSettings,
} from "../../../../core/api";
import type { MessageKey } from "../../../../shared/i18n";
import { core } from "../../core";
import { useLanguage, useT } from "../../i18n";
import { useModelNames } from "../../modelNames";
import { useAppStore } from "../../store";
import { fieldClass, ruledListClass, sectionTitleClass } from "../ui";
import { providerName } from "./ProviderPage";

type Role = "chatModel" | "quickTasksModel" | "imageModel";

const ROWS: readonly { role: Role; label: MessageKey; tip: MessageKey }[] = [
  { role: "chatModel", label: "models.defaults.answers", tip: "models.defaults.answers.tip" },
  {
    role: "quickTasksModel",
    label: "models.defaults.quickTasks",
    tip: "models.defaults.quickTasks.tip",
  },
  { role: "imageModel", label: "models.defaults.images", tip: "models.defaults.images.tip" },
];

/** A choice as a select's value; "" is Automatic, the setting left to be worked out. */
const keyOf = (choice: ChatModelChoice | null | undefined): string =>
  choice ? `${choice.providerId}\n${choice.modelId}` : "";

const choiceOf = (value: string): ChatModelChoice | null => {
  const [providerId, modelId] = value.split("\n");
  return providerId && modelId ? { providerId, modelId } : null;
};

/**
 * Settings › Models › Defaults (#179): the model for Answers, the quick-tasks
 * model and the model that reads images, each from the models in the menu of
 * every provider. The two others start as Automatic, which names the model it
 * comes to now; a model chosen for images is one that reads them.
 */
export function ModelDefaults({ groups }: { groups: readonly ChatModelGroup[] }) {
  const t = useT();
  const id = useId();
  const user = useAppStore((state) => state.settings?.user);
  const [roles, setRoles] = useState<ModelRoles | null>(null);

  // The models Automatic comes to follow the settings and the providers' menus.
  // biome-ignore lint/correctness/useExhaustiveDependencies: asked again whenever they change.
  useEffect(() => {
    core.getModelRoles().then(setRoles, (failure: unknown) => {
      console.error("Couldn't read which models the other jobs use:", failure);
    });
  }, [user?.chatModel, user?.quickTasksModel, user?.imageModel, groups]);

  return (
    <section data-testid="model-defaults" className="flex flex-col gap-2">
      <h4 className={sectionTitleClass}>{t("models.defaults.title")}</h4>
      <div className={ruledListClass}>
        {ROWS.map((row) => (
          <DefaultRow
            key={row.role}
            id={`${id}-${row.role}`}
            row={row}
            groups={groups}
            user={user}
            roles={roles}
          />
        ))}
      </div>
    </section>
  );
}

function DefaultRow({
  id,
  row,
  groups,
  user,
  roles,
}: {
  id: string;
  row: (typeof ROWS)[number];
  groups: readonly ChatModelGroup[];
  user: UserSettings | undefined;
  roles: ModelRoles | null;
}) {
  const t = useT();
  const language = useLanguage();
  const nameOf = useModelNames();
  const updateSettings = useAppStore((state) => state.updateSettings);
  const saved = keyOf(user?.[row.role]);
  // Shown at once; the setting follows when the core has it.
  const [value, setValue] = useState(saved);
  useEffect(() => setValue(saved), [saved]);

  const modelOf = (choice: ChatModelChoice) => nameOf(choice.modelId, choice.providerId).name;
  const labelOf = (choice: ChatModelChoice) => {
    const group = groups.find((each) => each.provider.id === choice.providerId);
    return group
      ? `${modelOf(choice)} · ${providerName(group.provider, t, language)}`
      : modelOf(choice);
  };
  const options = groups.flatMap((group) =>
    group.models
      .filter((modelId) => row.role !== "imageModel" || group.about[modelId]?.readsImages)
      .map((modelId) => ({ providerId: group.provider.id, modelId })),
  );
  const current = choiceOf(value);
  // The model in use stays shown even when no menu lists it now.
  if (current && !options.some((each) => keyOf(each) === value)) options.unshift(current);
  const automatic =
    row.role === "chatModel"
      ? null
      : ((row.role === "quickTasksModel" ? roles?.quickTasks : roles?.images) ?? null);
  const empty = options.length === 0 && row.role === "chatModel";

  const choose = (next: string) => {
    const choice = choiceOf(next);
    if (row.role === "chatModel" && !choice) return;
    setValue(next);
    void updateSettings({
      user:
        row.role === "chatModel"
          ? { chatModel: choice }
          : row.role === "quickTasksModel"
            ? { quickTasksModel: choice }
            : { imageModel: choice },
    });
  };

  return (
    <div className="flex min-h-14 flex-wrap items-center justify-between gap-x-6 gap-y-1 py-2">
      <label
        htmlFor={id}
        title={t(row.tip)}
        className="text-ui font-semibold text-ink"
        data-testid={`model-default-label-${row.role}`}
      >
        {t(row.label)}
      </label>
      <span id={`${id}-tip`} className="sr-only">
        {t(row.tip)}
      </span>
      <select
        id={id}
        aria-describedby={`${id}-tip`}
        data-testid={`model-default-${row.role}`}
        value={value}
        disabled={empty}
        title={current ? current.modelId : undefined}
        onChange={(event) => choose(event.target.value)}
        className={`${fieldClass} w-full max-w-[26rem] min-w-0 flex-1 basis-60`}
      >
        {row.role === "chatModel" ? (
          (empty || !current) && <option value="">{t("models.defaults.noModel")}</option>
        ) : (
          <option value="">
            {automatic
              ? t("models.defaults.automatic", { model: modelOf(automatic) })
              : t("models.defaults.automaticNone")}
          </option>
        )}
        {options.map((choice) => (
          <option key={keyOf(choice)} value={keyOf(choice)} title={choice.modelId}>
            {labelOf(choice)}
          </option>
        ))}
      </select>
    </div>
  );
}
