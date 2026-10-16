import { modelName } from "../shared/modelName";
import type { ChatProvider } from "./api";

/** A model as a statement or record names it, with the service it runs on: null for one on this computer. */
export interface ModelDescription {
  name: string;
  service: string | null;
}

/** The model a record of AI use names, or null when it names none. */
export function describeModel(
  providers: readonly ChatProvider[],
  providerId: string | null,
  modelId: string | null,
): ModelDescription | null {
  if (!modelId) return null;
  const provider = providers.find((each) => each.id === providerId);
  const name = modelName(
    modelId,
    provider ? { kind: provider.kind, endpoint: provider.endpoint } : {},
  );
  return { name, service: provider?.service?.name ?? null };
}
