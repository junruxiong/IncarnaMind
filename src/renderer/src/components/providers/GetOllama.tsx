import { OLLAMA_DOWNLOAD_URL } from "../../../../core/providers/catalog/billing";
import { useT } from "../../i18n";
import { buttonStyle } from "../ui";

export function getOllama(): void {
  // The main process opens it in the User's browser.
  window.open(OLLAMA_DOWNLOAD_URL, "_blank", "noreferrer");
}

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
