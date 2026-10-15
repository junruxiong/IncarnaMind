import { computePosition, flip, offset, shift } from "@floating-ui/dom";
import { type KeyboardEvent, type ReactNode, useId, useRef, useState } from "react";
import type { DocumentKind, ExportFormat } from "../../../../core/api";
import { shortcutLabel } from "../../commands/keymap";
import { useExportDialog } from "../../exportDialog";
import { exportMind, FORMAT_LABELS, useExports } from "../../exports";
import { useT } from "../../i18n";
import { enterFrom, exitToward } from "../../motion";
import { useAppStore } from "../../store";
import {
  ChevronRightLineIcon,
  DocumentLineIcon,
  ExportLineIcon,
  MoreLineIcon,
  SettingsLineIcon,
} from "../lineIcons";
import { menuClass, menuItemClass, menuRuleClass } from "../ui";
import { usePopoverMenu } from "../usePopoverMenu";

const MAIN_FORMATS: readonly ExportFormat[] = ["docx", "pdf"];
const MORE_FORMATS: readonly ExportFormat[] = ["markdown", "html", "source-pack"];

/** The formats Export's menu shows itself, the last one used first, and those under More formats. */
function exportChoices(last: ExportFormat): {
  shown: ExportFormat[];
  more: ExportFormat[];
} {
  const shown = [last, ...MAIN_FORMATS.filter((format) => format !== last)];
  return { shown, more: MORE_FORMATS.filter((format) => !shown.includes(format)) };
}

const ICON_KIND: Readonly<Record<ExportFormat, DocumentKind>> = {
  docx: "docx",
  pdf: "pdf",
  "source-pack": "pdf",
  html: "web",
  markdown: "markdown",
};

function FormatIcon({ format }: { format: ExportFormat }) {
  return <DocumentLineIcon kind={ICON_KIND[format]} className="size-4 shrink-0 text-ink-meta" />;
}

function Item({
  testId,
  icon,
  label,
  hint,
  onClick,
}: {
  testId: string;
  icon: ReactNode;
  label: string;
  hint?: string;
  onClick(): void;
}) {
  return (
    <button
      type="button"
      role="menuitem"
      data-testid={testId}
      onClick={onClick}
      className={`${menuItemClass} pl-2`}
    >
      {icon}
      <span className="flex-1">{label}</span>
      {hint && <span className="pl-4 text-[13px] text-ink-meta">{hint}</span>}
    </button>
  );
}

const itemsIn = (menu: HTMLElement | null) =>
  Array.from(menu?.querySelectorAll<HTMLElement>('[role="menuitem"]') ?? []);

export function ExportMenu({ mindId }: { mindId: string }) {
  const t = useT();
  const menu = usePopoverMenu({ focus: "first", align: "end" });
  const last = useAppStore((state) => state.settings?.user.exportOptions.format ?? "docx");
  const saving = useExports(
    (state) => state.result?.mindId === mindId && state.result.state === "saving",
  );
  const moreId = useId();
  const moreButton = useRef<HTMLButtonElement>(null);
  const moreMenu = useRef<HTMLDivElement>(null);
  const [moreOpen, setMoreOpen] = useState(false);
  const { shown, more } = exportChoices(last);

  const choose = (format: ExportFormat) => {
    menu.close();
    void exportMind(mindId, format);
  };
  const showMore = (focus: boolean) => {
    const sub = moreMenu.current;
    if (!sub) return;
    // Its button names it as its source, so it opens nested in Export's menu rather than closing it.
    if (!sub.matches(":popover-open")) sub.showPopover({ source: moreButton.current ?? undefined });
    if (focus) itemsIn(sub)[0]?.focus();
  };
  const hideMore = () => {
    if (moreMenu.current?.matches(":popover-open")) moreMenu.current.hidePopover();
  };
  const placeMore = () => {
    const anchor = moreButton.current;
    const sub = moreMenu.current;
    if (!anchor || !sub) return;
    void computePosition(anchor, sub, {
      placement: "right-start",
      strategy: "fixed",
      middleware: [offset({ mainAxis: 4, crossAxis: -4 }), flip(), shift({ padding: 8 })],
    }).then(({ x, y }) => {
      Object.assign(sub.style, { left: `${x}px`, top: `${y}px` });
      enterFrom(sub, anchor);
    });
  };
  const onMoreKeys = (event: KeyboardEvent<HTMLDivElement>) => {
    const items = itemsIn(moreMenu.current);
    const index = items.indexOf(document.activeElement as HTMLElement);
    if (event.key === "ArrowLeft" || event.key === "Escape") {
      event.preventDefault();
      event.stopPropagation();
      hideMore();
      moreButton.current?.focus();
      return;
    }
    const next =
      event.key === "ArrowDown"
        ? items[(index + 1) % items.length]
        : event.key === "ArrowUp"
          ? items[(index - 1 + items.length) % items.length]
          : undefined;
    if (!next) return;
    event.preventDefault();
    next.focus();
  };

  return (
    <>
      <button
        {...menu.buttonProps}
        type="button"
        data-testid="export-mind"
        data-format={last}
        aria-busy={saving || undefined}
        title={t("export.action")}
        className="tab-export"
      >
        <ExportLineIcon />
        <span>{t("export.action")}</span>
      </button>
      <div
        {...menu.menuProps}
        role="menu"
        aria-label={t("export.action")}
        data-testid="export-menu"
        className={menuClass}
      >
        {menu.open && (
          <>
            {shown.map((format, index) => (
              <div key={format} onPointerEnter={hideMore}>
                <Item
                  testId={`export-as-${format}`}
                  icon={<FormatIcon format={format} />}
                  label={t(FORMAT_LABELS[format])}
                  {...(index === 0 && { hint: shortcutLabel("mind.export") })}
                  onClick={() => choose(format)}
                />
              </div>
            ))}
            <button
              ref={moreButton}
              type="button"
              role="menuitem"
              data-testid="export-more"
              aria-haspopup="menu"
              aria-expanded={moreOpen}
              popoverTarget={moreId}
              // Shown already by the pointer coming onto it, a click mustn't close it again.
              popoverTargetAction="show"
              onPointerEnter={() => showMore(false)}
              onKeyDown={(event) => {
                if (event.key !== "ArrowRight" && event.key !== "Enter" && event.key !== " ")
                  return;
                event.preventDefault();
                showMore(true);
              }}
              className={`${menuItemClass} pl-2 aria-expanded:bg-hover`}
            >
              <MoreLineIcon className="size-4 shrink-0 text-ink-meta" />
              <span className="flex-1">{t("export.more")}</span>
              <ChevronRightLineIcon className="size-3.5 text-ink-meta" />
            </button>
            <div className={menuRuleClass} />
            <div onPointerEnter={hideMore}>
              <Item
                testId="export-options"
                icon={<SettingsLineIcon className="size-4 shrink-0 text-ink-meta" />}
                label={t("export.options")}
                onClick={() => {
                  menu.close();
                  useExportDialog.getState().open(mindId);
                }}
              />
            </div>
          </>
        )}
      </div>
      <div
        ref={moreMenu}
        id={moreId}
        popover="auto"
        role="menu"
        aria-label={t("export.more")}
        data-testid="export-more-menu"
        onBeforeToggle={(event) => {
          setMoreOpen(event.newState === "open");
          if (event.newState === "open") placeMore();
          else if (moreMenu.current) exitToward(moreMenu.current, moreButton.current);
        }}
        onKeyDown={onMoreKeys}
        className={menuClass}
      >
        {more.map((format) => (
          <Item
            key={format}
            testId={`export-as-${format}`}
            icon={<FormatIcon format={format} />}
            label={t(FORMAT_LABELS[format])}
            onClick={() => {
              hideMore();
              choose(format);
            }}
          />
        ))}
      </div>
    </>
  );
}
