/**
 * Tagging on a ChatGPT plan: the plan's limits are ChatGPT's own, so
 * tagging takes one Document at a time, its requests spaced, and when the
 * plan's limit is reached it waits for it to reset, sending nothing meanwhile.
 * Through the core, against the fake OpenAI the plan's own tests use.
 */
import { describe, expect, test, vi } from "vitest";
import { type Core, createAiSdkChatModel } from "../../src/core";
import { PLAN_PACE } from "../../src/core/library/planPace";
import type { PaceTimers } from "../../src/core/sources/pace";
import { createFakeBrowser, type FakeOpenAI, startFakeOpenAI } from "../helpers/chatgpt";
import {
  createMemoryKeychain,
  createTempDataFolder,
  manualClock,
  startCore,
} from "../helpers/core";
import { addAndProcess, writeSourceFile } from "../helpers/documents";
import { waitForTagging } from "../helpers/tags";

const LIMIT = {
  error: { code: "subscription_sharing_usage_limit_exceeded", message: "Limit reached." },
};

/**
 * A clock for the core's pacing that moves by itself for a short wait (the
 * gap between two requests), and holds a long one (the wait for the plan's
 * limit to reset) until the test moves it on.
 */
function pacingClock() {
  let time = Date.parse("2026-10-11T09:00:00.000Z");
  const short: number[] = [];
  const held: { until: number; resolve(): void }[] = [];
  const timers: PaceTimers = {
    now: () => time,
    async sleep(ms, signal) {
      signal.throwIfAborted();
      if (ms <= 0) return;
      if (ms <= PLAN_PACE.gapMs) {
        short.push(ms);
        time += ms;
        return;
      }
      await new Promise<void>((resolve, reject) => {
        held.push({ until: time + ms, resolve });
        signal.addEventListener("abort", () => reject(signal.reason), { once: true });
      });
    },
  };
  return {
    timers,
    /** The gaps waited between plan requests, in milliseconds. */
    short,
    /** How many long waits are held. */
    held: () => held.length,
    /** Moves the clock on, ending the long waits that are over. */
    advance(ms: number) {
      time += ms;
      for (const wait of held.filter((each) => each.until <= time)) {
        held.splice(held.indexOf(wait), 1);
        wait.resolve();
      }
    },
  };
}

/** Signed in to the plan, which answers Questions, with every consent accepted. */
async function onPlan() {
  // The sign-in's tokens are checked against the core's clock: one clock for both.
  const clock = manualClock();
  const fake = await startFakeOpenAI({ now: () => clock.now().getTime() });
  const pacing = pacingClock();
  const core = startCore(await createTempDataFolder(), {
    keychain: createMemoryKeychain(),
    browser: createFakeBrowser(fake),
    now: clock.now,
    createChatModel: createAiSdkChatModel,
    chatGptPlan: fake.endpoints,
    paceTimers: pacing.timers,
  });
  expect((await core.signInToChatGpt()).ok).toBe(true);
  core.on("consent.requested", (request) => void core.respondToConsent(request.requestId, true));
  await core.saveChatProvider({ kind: "chatgpt", modelId: "gpt-6.1-sol" });
  fake.reply = { kind: "answer", text: JSON.stringify({ tags: [] }) };
  return { core, fake, pacing };
}

async function add(core: Core, names: readonly string[]) {
  const sources = await createTempDataFolder();
  return addAndProcess(
    core,
    await Promise.all(names.map((name) => writeSourceFile(sources, `${name}.md`, `${name}.`))),
  );
}

/** The requests tagging sent to the plan. */
const tagging = (fake: FakeOpenAI) =>
  fake.responses().filter((request) => JSON.stringify(request.body).includes("document_tags"));

describe("Tagging on a ChatGPT plan", () => {
  test("Automatic tags with GPT-6 Luna, and the section says it uses the plan", async () => {
    const { core } = await onPlan();
    expect((await core.getTaggingModels()).automatic).toMatchObject({
      name: "GPT-6 Luna",
      reason: "plan",
      group: "plan",
      cost: { kind: "plan" },
    });
  });

  test("one Document at a time, its requests spaced", async () => {
    const { core, fake, pacing } = await onPlan();
    const documents = await add(core, ["One", "Two", "Three"]);
    await waitForTagging(
      core,
      documents.map((each) => each.id),
    );
    const sent = tagging(fake);
    expect(sent).toHaveLength(3);
    expect(sent.every((request) => request.body.model === "gpt-6-luna")).toBe(true);
    // Each after the one before had ended, a gap apart.
    expect(pacing.short).toEqual([PLAN_PACE.gapMs, PLAN_PACE.gapMs]);
  });

  test("on the plan's limit, Documents wait, nothing more is sent, and tagging goes on after the wait", async () => {
    const { core, fake, pacing } = await onPlan();
    fake.reply = { kind: "error", status: 429, body: LIMIT };
    const documents = await add(core, ["One", "Two", "Three"]);
    const ids = documents.map((each) => each.id);
    const waiting = await waitForTagging(
      core,
      ids,
      (each) => each.tagging === "waiting-for-provider",
    );
    expect(waiting.map((each) => each.taggingError?.kind)).toContain("plan-limit");
    // One request met the limit; the others never went.
    expect(tagging(fake)).toHaveLength(1);
    expect((await core.getTaggingModels()).planPausedUntil).toEqual(expect.any(String));

    // A re-tag while it waits sends nothing either: it never retries in a loop.
    await core.retagDocuments(ids);
    await waitForTagging(core, ids, (each) => each.tagging === "waiting-for-provider");
    expect(tagging(fake)).toHaveLength(1);
    expect(pacing.held()).toBe(1);

    // The limit resets: tagging goes on by itself.
    fake.reply = { kind: "answer", text: JSON.stringify({ tags: [] }) };
    pacing.advance(PLAN_PACE.firstWaitMs);
    await waitForTagging(core, ids);
    expect(tagging(fake)).toHaveLength(4);
    expect((await core.getTaggingModels()).planPausedUntil).toBeNull();
  });

  test("when the limit holds after the wait, the next wait is longer, and only one Document tried", async () => {
    const { core, fake, pacing } = await onPlan();
    fake.reply = { kind: "error", status: 429, body: LIMIT };
    const documents = await add(core, ["One", "Two"]);
    const ids = documents.map((each) => each.id);
    await waitForTagging(core, ids, (each) => each.tagging === "waiting-for-provider");
    expect(tagging(fake)).toHaveLength(1);

    pacing.advance(PLAN_PACE.firstWaitMs);
    await vi.waitFor(() => expect(tagging(fake)).toHaveLength(2));
    await vi.waitFor(() => expect(pacing.held()).toBe(1));
    await waitForTagging(core, ids, (each) => each.tagging === "waiting-for-provider");
    expect(tagging(fake)).toHaveLength(2);
    // Not over yet after another first wait: it doubled.
    pacing.advance(PLAN_PACE.firstWaitMs);
    expect(pacing.held()).toBe(1);
  });
});
