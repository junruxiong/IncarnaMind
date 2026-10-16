/** Looking for a local preset's server the way Ollama is looked for (see ./ollama), then for the models it lists. */
import type { LocalServerStatus } from "../api";
import { InvalidInputError } from "../errors";
import { catalogProvider, fieldAccepts, fieldUrl, isPreset } from "./catalog/providers";
import type { Endpoint } from "./catalog/types";
import { listProviderModels } from "./modelLists";
import { anythingAnswers } from "./ollama";

export interface LocalServers {
  /** `address` is a host and port; the preset's default when left out. */
  detect(
    catalogId: string,
    apiKey: string | null,
    address?: string | null,
  ): Promise<LocalServerStatus>;
}

/** The preset's endpoint and the address it is built from, refusing anything that isn't a host and port. */
function addressOf(catalogId: string, given?: string | null) {
  const preset = catalogProvider(catalogId);
  const endpoint: Endpoint | undefined =
    isPreset(preset) && preset.group === "local" ? preset.endpoints[0] : undefined;
  const field = endpoint?.field;
  if (!endpoint || !field) {
    throw new InvalidInputError({
      key: "error.provider.unknown",
      values: { provider: catalogId },
    });
  }
  const address = given?.trim() || field.default || "";
  if (!fieldAccepts(field, address)) {
    throw new InvalidInputError({ key: "error.provider.fieldInvalid", values: { value: address } });
  }
  return { endpoint, address, baseUrl: fieldUrl(endpoint, address) };
}

export function createLocalServers(): LocalServers {
  return {
    async detect(catalogId, apiKey, given) {
      const { endpoint, address, baseUrl } = addressOf(catalogId, given);
      const bearer: Record<string, string> = apiKey ? { authorization: `Bearer ${apiKey}` } : {};
      if (!(await anythingAnswers(`${baseUrl}/models`, bearer))) return { running: false, address };
      const listed = await listProviderModels({
        kind: "openai-compatible",
        baseUrl,
        apiKey,
        endpoint: endpoint.id,
        catalogId,
      });
      return {
        running: true,
        address,
        models: listed.filter((model) => !model.other).map((model) => model.id),
      };
    },
  };
}

/** Stand-ins that ask nothing: those in `running` list the models given, any other isn't running. */
export function scriptedLocalServers(
  running: Readonly<Record<string, string[]>> = {},
): LocalServers {
  return {
    async detect(catalogId, _apiKey, given) {
      const { address } = addressOf(catalogId, given);
      const models = running[catalogId];
      return models ? { running: true, address, models } : { running: false, address };
    },
  };
}
