import { OLLAMA_DOWNLOAD_URL } from "../../../../core/providers/catalog/billing";
import { useT } from "../../i18n";
import { buttonStyle } from "../ui";

/** Opens Ollama's download page; the main process opens it in the User's browser. */
export function getOllama(): void {
  window.open(OLLAMA_DOWNLOAD_URL, "_blank", "noreferrer");
}

/** "Get Ollama ↗", where Ollama isn't on this computer. */
export function GetOllamaButton({
  large = false,
  variant = "primary",
}: {
  large?: boolean;
  variant?: "primary" | "secondary";
}) {
  const t = useT();
  return (
    <button
      type="button"
      data-testid="ollama-get"
      onClick={getOllama}
      className={buttonStyle(variant, large ? "lg" : "md")}
    >
      {t("providers.ollama.get")} ↗
    </button>
  );
}
