import type { ChatProvider } from "../../core/api";
import { catalogProviderOf, untestedForCitations } from "../../core/providers/catalog/providers";
import type { MessageKey } from "../../shared/i18n";

/** "Not yet tested for checked Citations", for a model of a preset that hasn't passed the citing evaluation. */
export const untestedHintKey = (
  provider: Pick<ChatProvider, "kind" | "catalogId">,
): MessageKey | null =>
  untestedForCitations(catalogProviderOf(provider.kind, provider.catalogId))
    ? "providers.untested"
    : null;
