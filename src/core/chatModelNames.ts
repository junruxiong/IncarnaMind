import { modelName } from "../shared/modelName";
import type { ChatProvider } from "./api";

export type ModelPlace =
  | { kind: "service"; name: string }
  | { kind: "this-computer" }
  | { kind: "unknown" };

export interface ModelDescription {
  name: string;
  place: ModelPlace;
}

/**
 * The model a record of AI use names, or null when it names none. A provider
 * that is no longer saved leaves the place unknown: the model may have run on
 * a service, and a record never guesses.
 */
export function describeModel(
  providers: readonly ChatProvider[],
  providerId: string | null,
  modelId: string | null,
): ModelDescription | null {
  if (!modelId) return null;
  const provider = providers.find((each) => each.id === providerId);
  if (!provider) return { name: modelName(modelId, {}), place: { kind: "unknown" } };
  return {
    name: modelName(modelId, { kind: provider.kind, endpoint: provider.endpoint }),
    place: provider.service
      ? { kind: "service", name: provider.service.name }
      : { kind: "this-computer" },
  };
}
