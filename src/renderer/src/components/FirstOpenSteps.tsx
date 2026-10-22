import { type ReactNode, useEffect, useRef } from "react";
import type { CommandId } from "../../../shared/commands";
import type { MessageKey } from "../../../shared/i18n";
import { shortcutLabel } from "../commands/keymap";
import { type CommandContext, commands, offered } from "../commands/registry";
import { askReadyQuestion } from "../conversations";
import { CheckMarkIcon } from "../editor/icons";
import { useT } from "../i18n";
import { openModelSetup } from "../modelSetup";
import { stepsShownOf } from "../nudges";
import { revealAssistant } from "../quietAssistant";
import { useAppStore } from "../store";
import { ExampleBanner } from "./GettingStarted";
import { CloseLineIcon } from "./lineIcons";
import { ProblemLine } from "./ProblemLine";
import { SLOT, Slotted } from "./Slotted";
import { buttonStyle } from "./ui";

type StepKey = "citation" | "question" | "files";

const NEXT: readonly { id: CommandId; label: MessageKey }[] = [
  { id: "comparison.start", label: "firstOpen.next.compare" },
  { id: "meeting.record", label: "firstOpen.next.meeting" },
  { id: "mind.export", label: "firstOpen.next.export" },
];

// Step 2's button and the line after it wrap in the narrowest window (900 px, English), the tallest
// state at any width; the card keeps that height so the page never moves (e2e/welcomeCard.spec.ts).
const CARD_HEIGHT = "min-h-[219px]";

const HERE: CommandContext = { selection: [], focus: null, surface: "mind" };

function Mark({ number, state }: { number: number; state: "open" | "later" | "done" }) {
  if (state === "done") {
    return (
      <span
        aria-hidden="true"
        className="flex size-[18px] shrink-0 items-center justify-center rounded-full bg-success text-white"
      >
        <CheckMarkIcon check="found" className="size-2.5" />
      </span>
    );
  }
  return (
    <span
      aria-hidden="true"
      className={`flex size-[18px] shrink-0 items-center justify-center rounded-full text-[10px] leading-none font-bold ${
        state === "open" ? "bg-ink text-white" : "border-[1.5px] border-tab-divider text-ink-meta"
      }`}
    >
      {number}
    </span>
  );
}

function Step({
  step,
  number,
  state,
  children,
}: {
  step: StepKey;
  number: number;
  state: "open" | "later" | "done";
  children?: ReactNode;
}) {
  const t = useT();
  const tone = state === "open" ? "font-semibold text-ink" : "text-ink-meta";
  return (
    <li
      data-testid="first-open-step"
      data-step={step}
      data-state={state}
      className="flex flex-col gap-1"
    >
      <div className="flex items-center gap-2.5 text-[14px] leading-5">
        <Mark number={number} state={state} />
        <span className={tone}>
          {state === "done" && <span className="sr-only">{t("gettingStarted.done")}</span>}
          {t(`firstOpen.step.${step}.${state === "done" ? "done" : "title"}`)}
        </span>
      </div>
      {state === "open" && <div className="flex flex-col gap-2 pl-7">{children}</div>}
    </li>
  );
}

export function FirstOpenSteps({ mindId }: { mindId: string }) {
  const t = useT();
  const progress = useAppStore((state) => state.settings?.device.gettingStarted);
  const pickDocuments = useAppStore((state) => state.pickDocuments);
  const addLinkedFolder = useAppStore((state) => state.addLinkedFolder);
  const updateGettingStarted = useAppStore((state) => state.updateGettingStarted);
  const ready = useAppStore((state) => state.examples?.ready ?? null);
  const modelReady = useAppStore((state) => state.chatReadiness?.ready === true);
  const placeAssistant = useAppStore((state) => state.placeAssistant);
  const skipped = useAppStore((state) => state.skippedFiles);
  const addFiles = useRef<HTMLButtonElement>(null);
  // The ready Question's button goes once it is asked: the focus moves on to the next step.
  const focusNext = useRef(false);
  const filesOpen =
    progress !== undefined &&
    progress.citationChecked &&
    (progress.askedOwn || progress.askedReady) &&
    !progress.indexed;
  useEffect(() => {
    if (!filesOpen || !focusNext.current) return;
    focusNext.current = false;
    addFiles.current?.focus();
  }, [filesOpen]);
  if (!progress) return null;

  const done: Record<StepKey, boolean> = {
    citation: progress.citationChecked,
    question: progress.askedOwn || progress.askedReady,
    files: progress.indexed,
  };
  const order: StepKey[] = ["citation", "question", "files"];
  const current = order.find((key) => !done[key]);
  const stateOf = (key: StepKey): "open" | "later" | "done" =>
    done[key] ? "done" : key === current ? "open" : "later";
  const allDone = current === undefined;
  // Done means a first checked Answer of one's own can come: the ready Question needs no model (#1027).
  const setUp = allDone && modelReady;
  const chooseModel = () => {
    openModelSetup();
    revealAssistant();
    if (useAppStore.getState().layout?.assistant.folded) void placeAssistant({ folded: false });
  };

  return (
    <section
      aria-labelledby={`first-open-${mindId}`}
      data-testid="first-open-steps"
      className={`mb-6 ${CARD_HEIGHT} rounded-[10px] bg-sheet px-3.5 pt-3.5 pb-2.5 font-sans shadow-[inset_0_0_0_1px_var(--color-rule)]`}
    >
      <div className="flex items-start justify-between gap-2">
        <h2 id={`first-open-${mindId}`} className="text-[16px] leading-6 font-semibold text-ink">
          {t(setUp ? "firstOpen.done.title" : "firstOpen.title")}
        </h2>
        <button
          type="button"
          data-testid="first-open-hide"
          aria-label={t("firstOpen.hide")}
          title={t("firstOpen.hide")}
          onClick={() => updateGettingStarted({ hidden: true })}
          className="inline-flex size-[22px] items-center justify-center rounded-md text-ink-meta hover:bg-chip hover:text-ink focus-visible:outline-offset-0"
        >
          <CloseLineIcon strokeWidth={2.5} className="size-3" />
        </button>
      </div>
      <ol className={`flex flex-col gap-3 ${allDone ? "pt-2" : ""}`}>
        <Step step="citation" number={1} state={stateOf("citation")}>
          <p className="text-[13px] leading-5 text-ink-secondary">
            {t("firstOpen.step.citation.body")}
          </p>
        </Step>
        <Step step="question" number={2} state={stateOf("question")}>
          {ready ? (
            <>
              <p className="text-[13px] leading-5 text-ink-secondary">
                {t("firstOpen.step.question.body")}
              </p>
              <div className="flex flex-wrap items-center gap-x-3 gap-y-1.5">
                <button
                  type="button"
                  data-testid="first-open-ready"
                  onClick={() => {
                    focusNext.current = true;
                    void askReadyQuestion();
                  }}
                  className={buttonStyle("secondary", "sm")}
                >
                  {t("firstOpen.step.question.ready", { question: ready })}
                </button>
                <span className="text-[13px] leading-5 text-ink-meta">
                  <OwnQuestion message="firstOpen.step.question.own" />
                </span>
              </div>
            </>
          ) : (
            <p className="text-[13px] leading-5 text-ink-secondary">
              <OwnQuestion message="firstOpen.step.question.ownOnly" />
            </p>
          )}
        </Step>
        <Step step="files" number={3} state={stateOf("files")}>
          <p className="text-[13px] leading-5 text-ink-secondary">
            {t("firstOpen.step.files.body")}
          </p>
          <div className="flex flex-wrap gap-2">
            <button
              ref={addFiles}
              type="button"
              data-testid="first-open-add-files"
              onClick={() => void pickDocuments()}
              className={buttonStyle("primary", "sm")}
            >
              {t("startGuide.documents.addFiles")}
            </button>
            <button
              type="button"
              data-testid="first-open-link-folder"
              onClick={() => void addLinkedFolder()}
              className={buttonStyle("secondary", "sm")}
            >
              {t("startGuide.documents.linkFolder")}
            </button>
          </div>
          {skipped.length > 0 && (
            <ProblemLine testId="first-open-skipped">
              {t("documents.skipped", { names: skipped.join(", ") })}
            </ProblemLine>
          )}
        </Step>
      </ol>
      {allDone && !modelReady && (
        <div className="flex flex-col items-start gap-2 pt-3">
          <p className="text-[13px] leading-5 text-ink-secondary">{t("firstOpen.model.body")}</p>
          <button
            type="button"
            data-testid="first-open-choose-model"
            onClick={chooseModel}
            className={buttonStyle("primary", "sm")}
          >
            {t("firstOpen.model.choose")}
          </button>
        </div>
      )}
      {setUp && (
        <p className="pt-3 text-[13px] leading-5 text-ink-secondary">
          {t("firstOpen.next.lead")}{" "}
          {NEXT.flatMap(({ id, label }) => {
            const command = commands.get(id);
            return command && offered(command, HERE) ? [{ id, label }] : [];
          }).map(({ id, label }, index) => (
            <span key={id}>
              {index > 0 && " · "}
              <button
                type="button"
                data-testid="first-open-next"
                data-command={id}
                onClick={() => void commands.run(id)}
                className="rounded-sm text-accent-strong underline-offset-2 hover:underline focus-visible:outline-offset-0"
              >
                {t(label)}
              </button>
            </span>
          ))}
        </p>
      )}
    </section>
  );
}

function OwnQuestion({ message }: { message: MessageKey }) {
  const t = useT();
  return (
    <Slotted
      message={t(message, { shortcut: SLOT })}
      slot={
        <kbd className="rounded-sm bg-chip px-[5px] py-px font-sans text-[12px] text-ink-strong">
          {shortcutLabel("composer.focus")}
        </kbd>
      }
    />
  );
}

export function ExampleIntro({ mindId }: { mindId: string }) {
  const stepsShown = useAppStore(
    (state) =>
      state.examples?.mindId === mindId && stepsShownOf(state.settings?.device.gettingStarted),
  );
  return stepsShown ? <FirstOpenSteps mindId={mindId} /> : <ExampleBanner mindId={mindId} />;
}
