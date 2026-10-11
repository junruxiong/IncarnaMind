import { getSchema, type JSONContent } from "@tiptap/core";
import { EditorState } from "@tiptap/pm/state";
import { type Decoration, DecorationSet } from "@tiptap/pm/view";
import { describe, expect, test } from "vitest";
import { CITATION_NODE } from "../../src/core/api";
import {
  citationClusters,
  citationClustersPlugin,
  citedWorkOf,
  clusterCitations,
  clusterPlaceOf,
} from "../../src/renderer/src/editor/citationClusters";
import { noteExtensions } from "../../src/renderer/src/editor/noteSchema";

const schema = getSchema(noteExtensions());

const citation = (documentId: string, check = "found"): JSONContent => ({
  type: CITATION_NODE,
  attrs: { documentId, documentName: documentId, check, pageFrom: 2, pageTo: 3 },
});

const paragraph = (id: string, ...content: JSONContent[]): JSONContent => ({
  type: "paragraph",
  attrs: { id },
  content,
});

const text = (value: string): JSONContent => ({ type: "text", text: value });

const stateWith = (blocks: JSONContent[]) =>
  EditorState.create({
    doc: schema.nodeFromJSON({ type: "doc", content: blocks }),
    plugins: [citationClustersPlugin()],
  });

/** Each cluster's Documents, joined. */
const shown = (state: EditorState) =>
  citationClusters(state).map((cluster) =>
    cluster.members.map((member) => member.attributes.documentId).join("+"),
  );

/** The place each Citation's decorations give it: "cluster:member/size". */
function places(state: EditorState): string[] {
  const found: string[] = [];
  const decorations = clustersDecorations(state);
  state.doc.descendants((node, pos) => {
    if (node.type.name !== CITATION_NODE) return true;
    const place = clusterPlaceOf(decorations.filter((each) => each.from === pos));
    found.push(place ? `${place.cluster}:${place.member}/${place.size}` : "none");
    return false;
  });
  return found;
}

function clustersDecorations(state: EditorState): Decoration[] {
  const plugin = state.plugins[0];
  const set = plugin?.props.decorations?.call(plugin, state);
  return set instanceof DecorationSet ? set.find() : [];
}

describe("Citation clusters", () => {
  test("Citations with only spaces between them are one cluster; words or a new Block part them", () => {
    const state = stateWith([
      paragraph("a", text("Tides "), citation("tides"), text(" "), citation("moon"), text(".")),
      paragraph("b", text("Harbours "), citation("harbour"), text(" and "), citation("tables")),
      paragraph("c", citation("almanac")),
    ]);
    expect(shown(state)).toEqual(["tides+moon", "harbour", "tables", "almanac"]);
    expect(places(state)).toEqual(["0:0/2", "0:1/2", "1:0/1", "2:0/1", "3:0/1"]);
    expect(citationClusters(state)).toEqual(
      expect.arrayContaining([expect.objectContaining({ signature: expect.any(String) })]),
    );
    expect(clusterCitations(state.doc).map((each) => each.signature)).toEqual(
      citationClusters(state).map((each) => each.signature),
    );
  });

  test("follow the text: a word typed between two Citations parts them, and deleting it joins them again", () => {
    let state = stateWith([
      paragraph("a", text("Tides "), citation("tides"), text(" "), citation("moon"), text(".")),
    ]);
    // The space between them is at 9: "Tides " (1–7), the Citation (7), the space (8).
    state = state.apply(state.tr.insertText("and", 9));
    expect(shown(state)).toEqual(["tides", "moon"]);
    expect(places(state)).toEqual(["0:0/1", "1:0/1"]);
    state = state.apply(state.tr.delete(9, 12));
    expect(shown(state)).toEqual(["tides+moon"]);
  });

  test("typing elsewhere moves the places along without working them out again", () => {
    let state = stateWith([
      paragraph("a", text("Tides "), citation("tides"), text(" "), citation("moon")),
      paragraph("b", text("Notes")),
    ]);
    const before = citationClusters(state);
    state = state.apply(state.tr.insertText("More ", state.doc.content.size - 6));
    expect(citationClusters(state)).toBe(before);
    expect(places(state)).toEqual(["0:0/2", "0:1/2"]);
  });

  test("a check that changes redraws the cluster, which shows each work's check", () => {
    let state = stateWith([
      paragraph("a", citation("tides", "checking"), text(" "), citation("moon", "checking")),
    ]);
    const before = clustersDecorations(state);
    state = state.apply(
      state.tr.setNodeMarkup(1, undefined, {
        ...state.doc.nodeAt(1)?.attrs,
        check: "not-found",
      }),
    );
    expect(clustersDecorations(state)).not.toEqual(before);
    expect(shown(state)).toEqual(["tides+moon"]);
  });

  test("a Citation's cited work: its Document, and its Location from its pages", () => {
    const [cluster] = clusterCitations(stateWith([paragraph("a", citation("tides"))]).doc);
    const member = cluster?.members[0];
    if (!member) throw new Error("No Citation.");
    expect(citedWorkOf(member.attributes)).toEqual({
      documentId: "tides",
      documentName: "tides",
      location: { kind: "page", from: 2, to: 3 },
    });
  });
});
