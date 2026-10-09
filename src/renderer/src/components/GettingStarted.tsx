import type { ReactNode } from "react";
import type { GettingStarted } from "../../../core/api";
import { CheckMarkIcon } from "../editor/icons";
import { QUESTION_SHORTCUT_LABEL } from "../editor/questionCommands";
import { useT } from "../i18n";
import { useAppStore } from "../store";
import { AskIcon, PlugIcon } from "./icons";
import { ChevronRightLineIcon, CloseLineIcon } from "./lineIcons";
import { buttonStyle } from "./ui";

/**
 * Onboarding (DESIGN.md, First run): the example Mind ("Example" chips, its
 * banner, its Answer written in advance), the "Get started" checklist at the
 * bottom of the sidebar, and the three steps in an empty Mind.
 */

type StepKey = "citation" | "documents" | "question";

interface Step {
  key: StepKey;
  done: boolean;
}

/** Stands in for an element in a translated message, which is split around it. */
const SLOT = "";

/** A translated message with an element where its `{name}` placeholder was. */
function Slotted({ message, slot }: { message: string; slot: ReactNode }) {
  const [before, after] = message.split(SLOT);
  return (
    <>
      {before}
      {slot}
      {after}
    </>
  );
}

/**
 * The checklist's steps while it is shown: from a first run on, until the
 * User hides it or has done all three. Null otherwise.
 */
export function useGettingStarted(): Step[] | null {
  const progress = useAppStore((state) => state.settings?.device.gettingStarted);
  return progress ? stepsOf(progress) : null;
}

function stepsOf(progress: GettingStarted): Step[] | null {
  if (!progress.started || progress.hidden) return null;
  const steps: Step[] = [
    { key: "citation", done: progress.citationChecked },
    { key: "documents", done: progress.indexed },
    { key: "question", done: progress.askedOwn },
  ];
  return steps.every((step) => step.done) ? null : steps;
}

/** Whether a Mind is the example one. */
export const useIsExample = (mindId: string | undefined) =>
  useAppStore((state) => mindId !== undefined && state.examples?.mindId === mindId);

/** "Example", after the example Mind's name and its Linked folder's. */
export function ExampleChip() {
  const t = useT();
  return (
    <span
      data-testid="example-chip"
      className="h-[18px] shrink-0 rounded-sm bg-chip px-1.5 text-[11px] leading-[18px] font-semibold text-ink-secondary"
    >
      {t("examples.chip")}
    </span>
  );
}

/** A step's mark: its number, ringed (blue for the next one), or a green tick once done. */
function StepMark({ number, done, next }: { number: number; done: boolean; next: boolean }) {
  if (done) {
    return (
      <span
        aria-hidden="true"
        className="flex size-4 shrink-0 items-center justify-center rounded-full bg-success text-white"
      >
        <CheckMarkIcon check="found" className="size-2.5" />
      </span>
    );
  }
  return (
    <span
      aria-hidden="true"
      className={`flex size-4 shrink-0 items-center justify-center rounded-full border-[1.5px] text-[10px] leading-none font-bold ${
        next ? "border-accent text-accent" : "border-tab-divider text-ink-meta"
      }`}
    >
      {number}
    </span>
  );
}

/**
 * "Get started · N of 3", at the bottom of the sidebar: each step ticks
 * itself as the User does it, and the next one to do takes them there. It can
 * be hidden, and goes once all three are done.
 */
export function GetStartedCard() {
  const t = useT();
  const steps = useGettingStarted();
  const updateGettingStarted = useAppStore((state) => state.updateGettingStarted);
  const openExamples = useAppStore((state) => state.openExamples);
  const createMind = useAppStore((state) => state.createMind);
  const startQuestion = useAppStore((state) => state.startQuestion);
  if (!steps) return null;

  const done = steps.filter((step) => step.done).length;
  const next = steps.find((step) => !step.done)?.key;
  const go: Record<StepKey, () => void> = {
    citation: () => void openExamples(),
    // A Mind of their own, which shows the ways to add Documents or connect apps.
    documents: () => void createMind(),
    question: startQuestion,
  };

  return (
    <section
      aria-labelledby="get-started-title"
      data-testid="get-started"
      className="mx-2 mb-2 flex shrink-0 flex-col rounded-lg bg-sheet px-2 pt-2.5 pb-1.5 shadow-[inset_0_0_0_1px_var(--color-rule)]"
    >
      <div className="flex items-center justify-between pb-1 pl-2">
        <h2 id="get-started-title" className="text-[13px] leading-5 font-semibold text-ink">
          {t("gettingStarted.title")}{" "}
          <span data-testid="get-started-count" className="font-normal text-ink-meta">
            {t("gettingStarted.count", { done, total: steps.length })}
          </span>
        </h2>
        <button
          type="button"
          data-testid="get-started-hide"
          aria-label={t("gettingStarted.hide")}
          title={t("gettingStarted.hide")}
          onClick={() => updateGettingStarted({ hidden: true })}
          className="inline-flex size-[22px] items-center justify-center rounded-md text-ink-meta hover:bg-chip hover:text-ink focus-visible:outline-offset-0"
        >
          <CloseLineIcon strokeWidth={2.5} className="size-3" />
        </button>
      </div>
      <ol className="flex flex-col">
        {steps.map((step, index) => {
          const label = t(`gettingStarted.step.${step.key}`);
          const mark = <StepMark number={index + 1} done={step.done} next={step.key === next} />;
          // 28px, or taller when a long step wraps (as on the canvas): it isn't cut short.
          const rowClass =
            "flex min-h-7 w-full items-center gap-2 rounded-md px-2 py-1 text-left text-[13px] leading-5";
          return (
            <li key={step.key} data-testid="get-started-step" data-step={step.key}>
              {step.done ? (
                <span data-done="" className={`${rowClass} text-ink-meta line-through`}>
                  {mark}
                  <span className="min-w-0">
                    <span className="sr-only">{t("gettingStarted.done")}</span>
                    {label}
                  </span>
                </span>
              ) : step.key === next ? (
                <button
                  type="button"
                  onClick={go[step.key]}
                  className={`${rowClass} font-semibold text-ink hover:bg-hover focus-visible:outline-offset-0`}
                >
                  {mark}
                  <span className="min-w-0 flex-1">{label}</span>
                  <ChevronRightLineIcon
                    strokeWidth={2}
                    className="size-3.5 shrink-0 text-ink-meta"
                  />
                </button>
              ) : (
                <span className={`${rowClass} text-ink-secondary`}>
                  {mark}
                  <span className="min-w-0">{label}</span>
                </span>
              )}
            </li>
          );
        })}
      </ol>
      <button
        type="button"
        data-testid="onboarding-groups"
        className={`${buttonStyle("ghost", "sm")} mt-1 justify-start`}
        onClick={() => useAppStore.getState().openLibrary()}
      >
        {t("library.chooseStarters")}
      </button>
    </section>
  );
}

/**
 * Above the example Mind's title: what a Mind is, and the ways on: a Mind of
 * one's own, to add Documents or apps to, or removing the examples.
 */
export function ExampleBanner() {
  const t = useT();
  const createMind = useAppStore((state) => state.createMind);
  const removeExamples = useAppStore((state) => state.removeExamples);
  return (
    <aside
      aria-label={t("examples.banner.label")}
      data-testid="example-banner"
      // Reaching 12px into the margins, so its text keeps the Mind's text edge.
      className="-mx-3 mb-6 flex items-start gap-2.5 rounded-lg bg-accent-wash p-3 font-sans text-ui text-ink-strong"
    >
      <svg
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth={2}
        strokeLinecap="round"
        strokeLinejoin="round"
        aria-hidden="true"
        className="mt-0.5 size-4 shrink-0 text-accent-strong"
      >
        <circle cx="12" cy="12" r="9" />
        <path d="M12 11v5" />
        <path d="M12 8v.01" />
      </svg>
      <div className="flex min-w-0 flex-1 flex-col gap-1.5">
        <p>
          <strong className="font-semibold text-ink">{t("examples.banner.title")}</strong>{" "}
          {t("examples.banner.body")}
        </p>
        <div className="flex flex-wrap gap-2">
          <button
            type="button"
            data-testid="example-add-own"
            onClick={() => void createMind()}
            className={buttonStyle("primary", "sm")}
          >
            {t("examples.banner.addOwn")}
          </button>
          <button
            type="button"
            data-testid="example-remove"
            onClick={() => void removeExamples()}
            className={`${buttonStyle("ghost", "sm")} text-ink-strong`}
          >
            {t("examples.banner.remove")}
          </button>
        </div>
      </div>
    </aside>
  );
}

/** A step of the three, numbered: filled while it is the one to do, a green tick once done. */
function GuideStep({
  number,
  state,
  title,
  children,
}: {
  number: number;
  state: "next" | "later" | "done";
  title: string;
  children: ReactNode;
}) {
  return (
    <li data-testid="start-step" className="grid grid-cols-[24px_minmax(0,1fr)] gap-x-3">
      <span
        aria-hidden="true"
        className={`flex size-6 items-center justify-center rounded-full text-[12px] font-bold ${
          state === "next"
            ? "bg-ink text-white"
            : state === "done"
              ? "bg-success text-white"
              : "border-[1.5px] border-tab-divider text-ink-meta"
        }`}
      >
        {state === "done" ? <CheckMarkIcon check="found" className="size-3" /> : number}
      </span>
      <div className="flex min-w-0 flex-col gap-2">
        <h3 className="text-[16px] leading-6 font-semibold text-ink">{title}</h3>
        {children}
      </div>
    </li>
  );
}

/**
 * Under an empty Mind's first line, while the checklist is shown: the three
 * steps, add Documents or connect apps, ask, and check where each claim comes
 * from, and the example Mind to see it first. Typing in the Mind puts it away.
 */
export function StartGuide() {
  const t = useT();
  const progress = useAppStore((state) => state.settings?.device.gettingStarted);
  const examplesAvailable = useAppStore((state) => state.examples?.available === true);
  const addLinkedFolder = useAppStore((state) => state.addLinkedFolder);
  const pickDocuments = useAppStore((state) => state.pickDocuments);
  const openSettings = useAppStore((state) => state.openSettings);
  const openExamples = useAppStore((state) => state.openExamples);
  if (!progress) return null;

  const done = [progress.indexed, progress.askedOwn, progress.citationChecked];
  const next = done.indexOf(false);
  const stateOf = (index: number): "next" | "later" | "done" =>
    done[index] ? "done" : index === next ? "next" : "later";

  return (
    <section
      aria-labelledby="start-guide-title"
      data-testid="start-guide"
      className="mt-10 flex flex-col gap-1 font-sans"
    >
      <h2 id="start-guide-title" className="mb-2 text-[13px] leading-5 font-semibold text-ink-meta">
        {t("startGuide.title")}
      </h2>
      <ol className="flex flex-col gap-5">
        <GuideStep number={1} state={stateOf(0)} title={t("startGuide.documents.title")}>
          <p className="-mt-2 text-ui text-ink-secondary">{t("startGuide.documents.body")}</p>
          <div className="flex flex-wrap gap-2">
            <button
              type="button"
              data-testid="start-link-folder"
              onClick={() => void addLinkedFolder()}
              className={buttonStyle("primary")}
            >
              {t("startGuide.documents.linkFolder")}
            </button>
            <button
              type="button"
              data-testid="start-add-files"
              onClick={() => void pickDocuments()}
              className={buttonStyle("secondary")}
            >
              {t("startGuide.documents.addFiles")}
            </button>
            <button
              type="button"
              data-testid="start-connect-app"
              onClick={() => openSettings("connectors")}
              className={buttonStyle("secondary")}
            >
              <PlugIcon className="size-3.5" />
              {t("startGuide.documents.connectApp")}
            </button>
          </div>
          <p className="text-[13px] leading-5 text-ink-meta">{t("startGuide.documents.apps")}</p>
        </GuideStep>

        <GuideStep number={2} state={stateOf(1)} title={t("startGuide.ask.title")}>
          <p className="-mt-2 text-ui text-ink-secondary">
            <Slotted
              message={t("startGuide.ask.body", { shortcut: SLOT })}
              slot={
                <kbd className="rounded-sm bg-chip px-[5px] py-px font-sans text-[12px] text-ink-strong">
                  {QUESTION_SHORTCUT_LABEL}
                </kbd>
              }
            />
          </p>
          <div
            aria-hidden="true"
            className="relative rounded-lg bg-frame py-2 pr-3 pl-11 text-[15px] leading-6 text-ink-placeholder"
          >
            <span className="absolute top-2 left-2.5 flex size-6 items-center justify-center rounded-md bg-tab-divider text-white">
              <AskIcon className="size-3" />
            </span>
            {t("startGuide.ask.example")}
          </div>
        </GuideStep>

        <GuideStep number={3} state={stateOf(2)} title={t("startGuide.check.title")}>
          <p className="-mt-2 text-ui text-ink-secondary">
            <Slotted
              message={t("startGuide.check.body", { mark: SLOT })}
              slot={
                <span
                  aria-hidden="true"
                  className="inline-flex h-[18px] items-center gap-0.5 rounded-sm bg-success-wash pr-[5px] pl-[3px] align-[1px] text-[11px] leading-[18px] font-bold text-success"
                >
                  <CheckMarkIcon check="found" className="size-[11px]" />1
                </span>
              }
            />
          </p>
        </GuideStep>
      </ol>
      {examplesAvailable && (
        <p className="mt-5 text-ui text-ink-meta">
          {t("startGuide.example.lead")}{" "}
          <button
            type="button"
            data-testid="start-open-example"
            onClick={() => void openExamples()}
            className="rounded-sm text-accent underline-offset-2 hover:underline focus-visible:outline-offset-0"
          >
            {t("startGuide.example.open")}
          </button>
        </p>
      )}
    </section>
  );
}

/** Whether an Answer is the example one, written in advance: its meta line says so. */
export function useIsExampleAnswer(answerId: string | null) {
  return useAppStore((state) => answerId !== null && state.examples?.answerId === answerId);
}
