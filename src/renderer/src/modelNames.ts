import { type ModelSource, modelName } from "../../shared/modelName";
import { useAnswers } from "./answers";

/** What shows a model: its name, and its id for the tooltip where the two differ. */
export interface ShownModel {
  name: string;
  /** The id, to show on hover; undefined when the name is the id. */
  id: string | undefined;
}

/**
 * Names models for display (`modelName`), knowing the providers the menu has
 * loaded: with a provider id, that provider's kind and own names; without one,
 * any loaded provider's names, then the whole catalog.
 */
export function useModelNames(): (modelId: string, providerId?: string | null) => ShownModel {
  const groups = useAnswers((state) => state.models);
  return (modelId, providerId) => {
    const group = providerId ? groups?.find((each) => each.provider.id === providerId) : undefined;
    const source: ModelSource = group
      ? { kind: group.provider.kind, endpoint: group.provider.endpoint, names: group.names }
      : { names: groups?.find((each) => each.names[modelId])?.names };
    const name = modelName(modelId, source);
    return { name, id: name === modelId ? undefined : modelId };
  };
}
