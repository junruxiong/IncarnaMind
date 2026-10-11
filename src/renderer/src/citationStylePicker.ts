/**
 * The one citation style picker (#289), whichever way it was opened: the References block's
 * style button, the export dialog, a Mind's ⋯ menu, or ⌘K's "Citation style…". The picker
 * itself is `CitationStylePickerHost`, mounted once in the window.
 */
import { create } from "zustand";
import type { MindCitationStyle } from "../../core/api";
import { CITATION_STYLES, type CitationStyleId } from "../../shared/citationStyles";
import { type MessageKey, type MessageParams, translate } from "../../shared/i18n";
import { core } from "./core";
import { useAppStore } from "./store";
import { useUndo } from "./undo";

interface PickerState {
  /** The Mind whose style is being chosen, and what the picker opens beside; null: closed. */
  open: { mindId: string; anchor: HTMLElement | null; current: MindCitationStyle | null } | null;
  /** Opens the picker for a Mind beside `anchor` (none: at the top of the window). */
  show(mindId: string, anchor: HTMLElement | null): void;
  close(): void;
}

export const useCitationStylePicker = create<PickerState>()((set, get) => ({
  open: null,
  show(mindId, anchor) {
    set({ open: { mindId, anchor, current: null } });
    core.getMindCitationStyle(mindId).then(
      (current) => {
        const open = get().open;
        if (open?.mindId === mindId) set({ open: { ...open, current } });
      },
      (error: unknown) => useAppStore.getState().reportError(error),
    );
  },
  close: () => set({ open: null }),
}));

type Translate = (key: MessageKey, params?: MessageParams) => string;

/** A style's name in the interface language. */
export const styleLabel = (style: CitationStyleId, t: Translate) => {
  const found = CITATION_STYLES.find((each) => each.id === style);
  return found ? t(found.label) : style;
};

/**
 * Chooses a Mind's citation style, at once, which new Minds in its Folder then start in, and
 * offers Undo, which puts back both as they were (the Mind's "no choice", Numbered, if it had
 * none). Returns false if the core refused.
 */
export async function chooseCitationStyle(
  mindId: string,
  style: CitationStyleId,
  before: MindCitationStyle | null,
): Promise<boolean> {
  const { reportError, settings } = useAppStore.getState();
  const language = settings?.language ?? "en";
  const t: Translate = (key, params) => translate(language, key, params);
  try {
    const previous = before ?? (await core.getMindCitationStyle(mindId));
    if (previous.chosen && previous.style === style) return true;
    await core.setMindCitationStyle(mindId, style);
    useUndo.getState().offer({
      message: t("citationStyle.changed", { style: styleLabel(style, t) }),
      undo: async () => {
        await core.setMindCitationStyle(mindId, previous.chosen ? previous.style : null, {
          folderDefault: previous.folderDefault,
        });
      },
    });
    return true;
  } catch (error) {
    reportError(error);
    return false;
  }
}
