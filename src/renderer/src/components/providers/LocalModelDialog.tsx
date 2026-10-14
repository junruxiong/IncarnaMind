import { useT } from "../../i18n";
import {
  buttonClass,
  buttonStyle,
  dialogActionsClass,
  dialogBodyClass,
  dialogClass,
  dialogTextClass,
  dialogTitleClass,
} from "../ui";
import { useModal } from "../useModal";
import { OllamaAction } from "./ChatProviderChoices";
import { GetOllamaButton } from "./GetOllama";
import { type Ollama, OllamaProgress, ollamaDescription, useOllama } from "./OllamaCard";

/**
 * Where "Keep everything on this computer" refused a Question and no model is on
 * this computer: the choice of a local model, in place over the Mind. Ollama
 * says whether it's here, with Get Ollama or Open Ollama when it isn't running,
 * and one click downloads the recommended model and uses it.
 */
export function LocalModelDialog({ open, onClose }: { open: boolean; onClose(): void }) {
  const dialog = useModal(open);
  return (
    <dialog
      ref={dialog}
      onClose={onClose}
      data-testid="local-model-dialog"
      aria-labelledby="local-model-title"
      className={`${dialogClass} w-[28rem]`}
    >
      {/* Looks for Ollama each time it opens. */}
      {open && <LocalModelChoice onClose={onClose} />}
    </dialog>
  );
}

function LocalModelChoice({ onClose }: { onClose(): void }) {
  const t = useT();
  const ollama = useOllama(onClose);
  const { status } = ollama;
  return (
    <div className={dialogBodyClass}>
      <h2 id="local-model-title" className={`${dialogTitleClass} break-words`}>
        {t("localModel.title")}
      </h2>
      <div className="flex flex-col gap-2">
        <p data-testid="local-model-status" className={dialogTextClass}>
          {status && !status.running && status.installed
            ? t("localModel.installed")
            : ollamaDescription(ollama, t)}
        </p>
        {status?.recommendedModel && (
          <p data-testid="local-model-recommended" className={dialogTextClass}>
            {t("localModel.recommended", { model: status.recommendedModel })}
          </p>
        )}
        <OllamaProgress ollama={ollama} />
      </div>
      <div className={dialogActionsClass}>
        <button type="button" onClick={onClose} className={buttonClass}>
          {t("localModel.cancel")}
        </button>
        <LocalModelActions ollama={ollama} />
      </div>
    </div>
  );
}

/** Use the recommended model once Ollama runs; until then, open it or get it, and check again. */
function LocalModelActions({ ollama }: { ollama: Ollama }) {
  const t = useT();
  const { status, opening } = ollama;
  if (status?.running) return <OllamaAction ollama={ollama} />;
  return (
    <>
      <button
        type="button"
        data-testid="ollama-check-again"
        disabled={status === null || opening}
        onClick={() => void ollama.detect()}
        className={buttonClass}
      >
        {t("providers.ollama.checkAgain")}
      </button>
      {status?.installed ? (
        <button
          type="button"
          data-testid="ollama-open"
          disabled={opening}
          onClick={() => void ollama.open()}
          className={buttonStyle("primary")}
        >
          {t(opening ? "localModel.opening" : "localModel.open")}
        </button>
      ) : (
        status !== null && <GetOllamaButton />
      )}
    </>
  );
}
