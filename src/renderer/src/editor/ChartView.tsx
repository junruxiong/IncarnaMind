import type { Node as ProseMirrorNode } from "@tiptap/pm/model";
import { NodeViewWrapper, type ReactNodeViewProps, useEditorState } from "@tiptap/react";
import { type KeyboardEvent, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { BLOCK_ID_ATTRIBUTE, type CitationCheck } from "../../../core/api";
import { chartLegend } from "../../../shared/chartLayout";
import {
  type ChartAttributes,
  type ChartData,
  type ChartKind,
  type ChartValue,
  chartCells,
  chartPoints,
  chartSources,
  drawnSeries,
  kindFits,
  MAX_CHART_VALUES,
  seriesName as nameOfSeries,
  readChart,
} from "../../../shared/charts";
import {
  badgeMessage,
  type CitationState,
  citationSource,
  citationState,
} from "../../../shared/citations";
import { citationLocation, formatLocation } from "../../../shared/locations";
import { ObjectMenu, useObjectMenu } from "../components/ObjectMenu";
import { useTabPlace } from "../components/tabs/tabItems";
import { useLanguage, useT } from "../i18n";
import { openCited, quoteSaysSo } from "../openCited";
import { useAppStore } from "../store";
import { ChartFoot } from "./ChartFoot";
import { ChartKindMenu } from "./ChartKindMenu";
import { ChartPlot, type PointEvents, type PointRef, samePoint } from "./ChartPlot";
import { ChartPointCard } from "./ChartPointCard";
import { ChartValues } from "./ChartValues";
import { changeChart, chartTable, tableUpdate, toggledSeries, withKind } from "./chartCommands";
import { type ChartHandle, chartHandles } from "./chartHandles";
import { Popover } from "./Popover";

/** As a Citation's card: the pointer rests a moment before it shows, and may leave a moment before it goes. */
const SHOW_MS = 120;
const HIDE_MS = 200;

interface Card {
  point: PointRef;
  anchor: Element;
  pinned: boolean;
}

function usePointCard() {
  const [card, setCard] = useState<Card | null>(null);
  const askers = useRef(new Set<string>());
  const wanted = useRef<{ point: PointRef; anchor: Element } | null>(null);
  const timer = useRef<number | undefined>(undefined);
  const settle = useCallback((delay: number) => {
    window.clearTimeout(timer.current);
    timer.current = window.setTimeout(() => {
      setCard((shown) => {
        const asked = askers.current.size > 0;
        if (!asked) return shown?.pinned ? shown : null;
        const next = wanted.current;
        if (!next) return shown;
        if (shown && samePoint(shown.point, next.point) && shown.anchor === next.anchor)
          return shown;
        return { ...next, pinned: false };
      });
    }, delay);
  }, []);
  useEffect(() => () => window.clearTimeout(timer.current), []);
  return {
    card,
    peek: (point: PointRef, anchor: Element, source: string, on: boolean) => {
      if (on) {
        askers.current.add(source);
        wanted.current = { point, anchor };
        settle(source === "focus" ? 0 : SHOW_MS);
      } else {
        askers.current.delete(source);
        settle(HIDE_MS);
      }
    },
    hold: (source: string, on: boolean) => {
      if (on) {
        askers.current.add(source);
        window.clearTimeout(timer.current);
      } else {
        askers.current.delete(source);
        settle(HIDE_MS);
      }
    },
    pin: (point: PointRef, anchor: Element) => {
      wanted.current = { point, anchor };
      window.clearTimeout(timer.current);
      setCard({ point, anchor, pinned: true });
    },
    dismiss: () => {
      askers.current.clear();
      window.clearTimeout(timer.current);
      setCard(null);
    },
  };
}

/** Worked out again only when the table or the chart changes, not on every keystroke elsewhere in the Mind. */
function useTableUpdate(
  editor: ReactNodeViewProps["editor"],
  node: ProseMirrorNode,
  language: string,
): ChartData | null {
  const last = useRef<{ table: ProseMirrorNode | null; chart: unknown; update: ChartData | null }>(
    null,
  );
  return useEditorState({
    editor,
    selector: ({ editor: current }) => {
      if (!current.isEditable) return null;
      const table = chartTable(current.state.doc, readChart(node.attrs));
      if (last.current && last.current.table === table && last.current.chart === node.attrs.data) {
        return last.current.update;
      }
      const update = table ? tableUpdate(current.state.doc, node.attrs, language) : null;
      last.current = { table, chart: node.attrs.data, update };
      return update;
    },
  });
}

function checksOf(states: readonly (CitationState | null)[]): {
  check: CitationCheck | null;
  notFound: number;
} {
  const checks = states.flatMap((state) => (state ? [state.check] : []));
  const notFound = checks.filter(
    (check) => check === "not-found" || check === "unsupported",
  ).length;
  const order: CitationCheck[] = [
    "contradicted",
    "not-found",
    "unsupported",
    "cant-check",
    "checking",
    "found",
  ];
  return { check: order.find((check) => checks.includes(check)) ?? null, notFound };
}

/** A chart in a Mind (#436), as on the design canvas's Chart boards. */
export function ChartView({ node, editor, getPos, deleteNode }: ReactNodeViewProps) {
  const t = useT();
  const language = useLanguage();
  const chart = useMemo(() => readChart(node.attrs), [node.attrs]);
  const id = node.attrs[BLOCK_ID_ATTRIBUTE];
  const blockId = typeof id === "string" ? id : "";
  const editable = useEditorState({
    editor,
    selector: ({ editor: current }) => current.isEditable,
  });
  const [asTable, setAsTable] = useState(false);
  const [renaming, setRenaming] = useState(false);
  const [frame, setFrame] = useState<HTMLDivElement | null>(null);
  const update = useTableUpdate(editor, node, language);
  const documents = useAppStore((state) =>
    state.status.kind === "ready" ? state.documents : null,
  );
  const kept = useAppStore((state) => state.keptCitationTexts);
  /** The Mind's group: the Document opens in the group to its right. */
  const place = useTabPlace();
  const { card, peek, hold, pin, dismiss } = usePointCard();

  const stateOf = useCallback(
    (value: ChartValue | null | undefined): CitationState | null =>
      value?.citation ? citationState(value.citation, documents, kept) : null,
    [documents, kept],
  );
  const seriesName = useCallback((index: number) => nameOfSeries(chart, index, t), [chart, t]);
  const labelHeading = chart.data.labelHeading;
  const drawn = drawnSeries(chart);
  const points = chartPoints(chart);
  const sources = chartSources(chart);
  const checks = checksOf(
    chartCells(chart).flatMap(({ value }) => (value.value === null ? [] : [stateOf(value)])),
  );

  const change = (
    attributes: Partial<Pick<ChartAttributes, "kind" | "title" | "drawn" | "data">>,
  ) => {
    const pos = getPos();
    if (pos !== undefined) changeChart(editor, pos, attributes);
  };
  const setKind = (kind: ChartKind) => {
    dismiss();
    change(withKind(chart, kind));
  };
  const toggleTable = () => {
    dismiss();
    setAsTable((shown) => !shown);
  };
  const applyUpdate = () => {
    if (!update) return;
    // Columns that came or went: draw them all again.
    const same = JSON.stringify(update.series) === JSON.stringify(chart.data.series);
    change({ data: update, ...(same ? {} : { drawn: null }) });
  };

  const latest = useRef({ chart, asTable, update, editable, setKind, toggleTable, applyUpdate });
  latest.current = { chart, asTable, update, editable, setKind, toggleTable, applyUpdate };
  useEffect(() => {
    if (!blockId) return;
    const handle: ChartHandle = {
      kind: () => latest.current.chart.kind,
      fits: (kind) => latest.current.editable && kindFits(kind, latest.current.chart.data),
      setKind: (kind) => {
        if (latest.current.editable) latest.current.setKind(kind);
      },
      asTable: () => latest.current.asTable,
      toggleTable: () => latest.current.toggleTable(),
      canUpdate: () => latest.current.editable && latest.current.update !== null,
      update: () => latest.current.applyUpdate(),
      rename: () => {
        if (latest.current.editable) setRenaming(true);
      },
      remove: () => {
        if (latest.current.editable) deleteNode();
      },
    };
    chartHandles.set(blockId, handle);
    return () => {
      if (chartHandles.get(blockId) === handle) chartHandles.delete(blockId);
    };
  }, [blockId, deleteNode]);

  const titleOf = (point: PointRef) => {
    const row = chart.data.rows[point.row]?.label ?? "";
    // A pie's slices are one column's: its card says which, as a bar's does among several.
    if (
      chart.kind === "pie" ||
      ((chart.kind === "bar" || chart.kind === "line") && drawn.length > 1)
    ) {
      return t("chart.point.title", { row, series: seriesName(point.series) });
    }
    return row || seriesName(point.series);
  };
  const detailOf = (point: PointRef): string | null => {
    if (chart.kind === "scatter") {
      const across = drawn[0];
      const value = across === undefined ? null : chart.data.rows[point.row]?.values[across];
      return value && across !== undefined
        ? t("chart.point.named", { name: seriesName(across), value: value.shown })
        : null;
    }
    if (chart.kind === "pie") {
      const total = points.reduce((sum, each) => sum + each.value.value, 0);
      const value = chart.data.rows[point.row]?.values[point.series]?.value;
      if (value == null || total <= 0) return null;
      const share = new Intl.NumberFormat(language, {
        style: "percent",
        minimumFractionDigits: 1,
        maximumFractionDigits: 1,
      }).format(value / total);
      const name = seriesName(point.series);
      return t("chart.point.share", {
        share,
        series: language === "zh-CN" ? spacedInChinese(name) : name,
      });
    }
    return null;
  };
  const valueText = (point: PointRef) => {
    const value = chart.data.rows[point.row]?.values[point.series];
    if (!value) return "";
    return chart.kind === "scatter"
      ? t("chart.point.named", { name: seriesName(point.series), value: value.shown })
      : value.shown;
  };
  const sourceText = (value: ChartValue | null | undefined) => {
    const state = stateOf(value);
    if (!value?.citation || !state) return t("chart.point.noCitation");
    const badge = badgeMessage(state, value.citation, t);
    return `${citationSource(value.citation, t).text}. ${t(badge.key, badge.params)}`;
  };

  const openValue = (point: PointRef) => {
    const value = chart.data.rows[point.row]?.values[point.series];
    const state = stateOf(value);
    if (!value?.citation || !state) return;
    const documentId = state.documentId ?? value.citation.documentId;
    if (!documentId) return;
    const location = citationLocation(value.citation);
    openCited(
      { ...value.citation, documentId },
      {
        check: state.check,
        withQuote: quoteSaysSo(state.check) || state.reason === "document-removed",
        label: location ? formatLocation(location, t) : null,
        from: place?.groupId,
      },
    );
  };
  const events: PointEvents = {
    peek: (point, anchor, source, on) => peek(point, anchor, source, on),
    activate: (point, anchor) => {
      pin(point, anchor);
      const state = stateOf(chart.data.rows[point.row]?.values[point.series]);
      // A quote that wasn't found opens from its card's Open, as a Citation's does.
      if (state && state.check !== "not-found") openValue(point);
    },
    label: (point) =>
      t("chart.point", {
        label: titleOf(point),
        value: valueText(point),
        source: sourceText(chart.data.rows[point.row]?.values[point.series]),
      }),
  };

  const objectMenu = useObjectMenu({ kind: "chart", id: blockId });
  const title = chart.title ? t("chart.label", { title: chart.title }) : t("chart.label.untitled");
  const kindName = t(`chart.kindName.${chart.kind}`);
  const plotLabel = t("chart.plot.label", {
    kind: kindName,
    title: chart.title || kindName,
    count: points.length,
  });
  const legend = !asTable ? chartLegend(chart, seriesName) : [];
  const shownCard = card ? chart.data.rows[card.point.row]?.values[card.point.series] : null;
  const shownState = stateOf(shownCard);

  const onKeyDown = (event: KeyboardEvent<HTMLElement>) => {
    if (event.key === "F2" && editable && !renaming) {
      event.preventDefault();
      setRenaming(true);
      return;
    }
    if (editable) objectMenu.rowProps.onKeyDown(event);
  };

  return (
    <NodeViewWrapper
      className="chart-block"
      data-testid="chart"
      data-kind={chart.kind}
      data-chart-id={blockId || undefined}
      contentEditable={false}
      {...(editable && objectMenu.rowProps)}
      onKeyDown={onKeyDown}
    >
      <div className="chart-head" data-chart-ui="">
        {renaming ? (
          <ChartTitleField
            title={chart.title}
            onDone={(next) => {
              setRenaming(false);
              if (next !== null && next !== chart.title) change({ title: next });
            }}
          />
        ) : (
          <button
            type="button"
            tabIndex={-1}
            className="chart-title"
            data-testid="chart-title"
            title={title}
            onDoubleClick={() => {
              if (editable) setRenaming(true);
            }}
          >
            {title}
          </button>
        )}
        <span className="chart-head-space" />
        {editable && (
          <ChartKindMenu
            chart={chart}
            seriesName={seriesName}
            fits={(kind) => kindFits(kind, chart.data)}
            onKind={setKind}
            onSeries={(series) => {
              dismiss();
              change({ drawn: toggledSeries(chart, series) });
            }}
          />
        )}
        <button
          type="button"
          data-testid="chart-show-table"
          aria-pressed={asTable}
          className="chart-head-button"
          onClick={toggleTable}
        >
          {t(asTable ? "chart.showAsChart" : "chart.showAsTable")}
        </button>
        {editable && blockId && (
          <ObjectMenu
            target={{ kind: "chart", id: blockId }}
            label={t("chart.more")}
            here={{ "block.delete": () => deleteNode() }}
            className="chart-menu-anchor"
            tabIndex={-1}
          />
        )}
      </div>
      <div ref={setFrame} className="chart-body" data-testid="chart-body">
        {points.length === 0 && !asTable ? (
          <p className="chart-empty" data-testid="chart-empty">
            {chart.kind === "scatter" && drawn.length < 2
              ? t("chart.scatter.needsTwo")
              : t("chart.empty")}
          </p>
        ) : asTable ? (
          <ChartValues
            chart={chart}
            seriesName={seriesName}
            labelHeading={labelHeading}
            active={card?.point ?? null}
            events={events}
            checkOf={(value) => {
              const state = stateOf(value);
              if (!value.citation || !state)
                return { check: null, label: t("chart.point.noCitation") };
              const badge = badgeMessage(state, value.citation, t);
              return { check: state.check, label: t(badge.key, badge.params) };
            }}
          />
        ) : (
          <ChartPlot
            chart={chart}
            language={language}
            seriesName={seriesName}
            active={card?.point ?? null}
            events={events}
            label={plotLabel}
          />
        )}
      </div>
      <ChartFoot
        legend={legend}
        sources={sources}
        check={checks.check}
        notFound={checks.notFound}
        rows={chart.data.rows.length}
        cutFrom={chart.data.cutFrom}
        onUpdate={editable && update ? applyUpdate : null}
      />
      {card &&
        shownCard &&
        // In the body, not the editor: ProseMirror must not see the card's events.
        createPortal(
          <Popover
            anchor={card.anchor}
            placement="top"
            // Between the plot's top and bottom: never over the chart's head (its "Bar ⌄") or its foot.
            {...(frame && { within: frame })}
            onDismiss={dismiss}
            aria-label={events.label(card.point)}
            data-testid="chart-card"
            onPointerEnter={() => hold("card", true)}
            onPointerLeave={() => hold("card", false)}
            onFocus={() => hold("card-focus", true)}
            onBlur={() => hold("card-focus", false)}
          >
            <ChartPointCard
              title={titleOf(card.point)}
              value={valueText(card.point)}
              detail={detailOf(card.point)}
              notDrawn={shownCard.value === null}
              cut={
                chart.data.cutFrom
                  ? t("chart.cut", {
                      rows: chart.data.rows.length,
                      total: chart.data.cutFrom,
                      max: MAX_CHART_VALUES,
                    })
                  : null
              }
              citation={shownCard.citation}
              state={shownState}
              onOpen={() => {
                openValue(card.point);
                dismiss();
              }}
            />
          </Popover>,
          document.body,
        )}
    </NodeViewWrapper>
  );
}

/** Chinese text keeps a space beside Latin letters and numbers (docs/agents/zh-CN.md, rule 2): "占 2025 的", "占营业收入的". */
function spacedInChinese(name: string): string {
  const latin = /[A-Za-z0-9]/;
  return `${latin.test(name.at(0) ?? "") ? " " : ""}${name}${latin.test(name.at(-1) ?? "") ? " " : ""}`;
}

/** A blank title changes nothing; Enter while an input method composes never cuts it off. */
function ChartTitleField({ title, onDone }: { title: string; onDone(title: string | null): void }) {
  const t = useT();
  const [value, setValue] = useState(title);
  const field = useRef<HTMLInputElement>(null);
  const done = useRef(false);
  const finish = (next: string | null) => {
    if (done.current) return;
    done.current = true;
    onDone(next === null || next.trim() === "" ? null : next.trim());
  };
  useEffect(() => {
    field.current?.focus();
    field.current?.select();
  }, []);
  return (
    <input
      ref={field}
      data-testid="chart-title-field"
      aria-label={t("chart.title.field")}
      className="chart-title-field"
      value={value}
      onChange={(event) => setValue(event.target.value)}
      onKeyDown={(event) => {
        if (event.nativeEvent.isComposing || event.keyCode === 229) return;
        if (event.key === "Enter") {
          event.preventDefault();
          finish(value);
        } else if (event.key === "Escape") {
          event.preventDefault();
          event.stopPropagation();
          finish(null);
        }
      }}
      onBlur={() => finish(value)}
    />
  );
}
