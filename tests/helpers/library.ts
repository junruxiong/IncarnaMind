import { MockLanguageModelV4 } from "ai/test";

/**
 * A chat model for Organize: it files each Document in the first Folder it is offered, or
 * the one chosen, with the Tags chosen. `hold` keeps its answers back until `release`.
 */
export function classifierModel() {
  const calls: string[] = [];
  let chosen: string | undefined;
  let selectedTags: string[] = [];
  let gate: Promise<void> | null = null;
  let release = () => {};
  const model = new MockLanguageModelV4({
    doGenerate: async (options) => {
      const schema =
        options.responseFormat?.type === "json" ? options.responseFormat.schema : undefined;
      const group = (schema as { properties?: { groupId?: { enum: string[] } } } | undefined)
        ?.properties?.groupId;
      let text = JSON.stringify({ tags: [] });
      if (group) {
        calls.push(JSON.stringify(options.prompt));
        const choice = chosen ?? group.enum[0];
        const tagIds = selectedTags;
        if (gate) await gate;
        text = JSON.stringify({ groupId: choice, tags: tagIds });
      }
      return {
        content: [{ type: "text", text }],
        finishReason: { unified: "stop", raw: undefined },
        warnings: [],
        usage: {
          inputTokens: { total: 5, noCache: 5, cacheRead: undefined, cacheWrite: undefined },
          outputTokens: { total: 1, text: 1, reasoning: undefined },
        },
      };
    },
  });
  return {
    model,
    calls,
    tag: (ids: string[]) => {
      selectedTags = ids;
    },
    choose: (id: string) => {
      chosen = id;
    },
    hold() {
      gate = new Promise((resolve) => {
        release = resolve;
      });
    },
    release() {
      gate = null;
      release();
    },
  };
}
