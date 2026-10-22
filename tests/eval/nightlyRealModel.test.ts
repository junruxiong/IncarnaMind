import { describe, expect, test } from "vitest";
import {
  type Attempt,
  missingMessage,
  missingSecrets,
  OUT_OF_CREDIT,
  RATES,
  type Result,
  runSteps,
  SECRETS,
  STEPS,
  spendOf,
  summary,
} from "../../scripts/nightlyRealModel";
import { usageOf } from "../../scripts/usageTally";
import { catalogFacts } from "../../src/core/providers/catalog";

const [live, everyday, kimiLive] = STEPS;
if (!live || !everyday || !kimiLive) throw new Error("The steps changed.");

const KEY_1 = "sk-ant-first-key-0000";
const KEY_2 = "sk-ant-second-key-1111";

describe("the secrets", () => {
  // protects=a night with a secret missing, or set to nothing, stops before it spends anything and names the secret; fails_when=a missing secret goes unnoticed until a model refuses the empty key, or the message names no secret; no_claim=that the secrets hold working keys.
  test("a missing or empty secret is named in one plain line", () => {
    const all = {
      NIGHTLY_ANTHROPIC_KEY_1: KEY_1,
      NIGHTLY_ANTHROPIC_KEY_2: KEY_2,
      NIGHTLY_KIMI_KEY: "k",
    };
    expect(missingSecrets(all, STEPS)).toEqual([]);
    expect(missingSecrets({ ...all, NIGHTLY_KIMI_KEY: "  " }, STEPS)).toEqual(["NIGHTLY_KIMI_KEY"]);
    const message = missingMessage(missingSecrets({ NIGHTLY_ANTHROPIC_KEY_1: KEY_1 }, STEPS));
    expect(message).toContain("NIGHTLY_ANTHROPIC_KEY_2");
    expect(message).toContain("NIGHTLY_KIMI_KEY");
    expect(message).not.toContain(KEY_1);
    expect(message).not.toContain("\n");
  });

  // protects=a run of only some steps asks only for the secrets those steps use; fails_when=running the live tests by hand needs a Kimi key; no_claim=which steps the night runs.
  test("only the steps asked for need their secrets", () => {
    expect(missingSecrets({}, [kimiLive])).toEqual(["NIGHTLY_KIMI_KEY"]);
    expect(
      missingSecrets({ NIGHTLY_ANTHROPIC_KEY_1: KEY_1, NIGHTLY_ANTHROPIC_KEY_2: KEY_2 }, [
        live,
        everyday,
      ]),
    ).toEqual([]);
  });
});

describe("the second key", () => {
  const run = (exitCode: number, log = "") => ({ exitCode, log, seconds: 1 });
  const credit = run(
    1,
    "AI_APICallError: Your credit balance is too low to access the Anthropic API.",
  );

  // protects=a step whose first key is out of credit runs again on the second, and the steps after it go straight to the second; fails_when=the night fails because one key ran dry, or tries the empty key again for every step; no_claim=how the provider words the error.
  test("an out-of-credit failure moves the night to the next key for good", async () => {
    const asked: string[] = [];
    const attempts = await runSteps([live, everyday, kimiLive], async (step, secret) => {
      asked.push(`${step.id} ${secret}`);
      return step === live && secret === SECRETS.anthropic[0] ? credit : run(0);
    });
    expect(asked).toEqual([
      "live NIGHTLY_ANTHROPIC_KEY_1",
      "live NIGHTLY_ANTHROPIC_KEY_2",
      "everyday NIGHTLY_ANTHROPIC_KEY_2",
      "kimi-live NIGHTLY_KIMI_KEY",
    ]);
    expect(attempts.map((each) => each.outOfCredit)).toEqual([true, false, false, false]);
  });

  // protects=any other failure is a failure, not a reason to spend again on the second key; fails_when=a failing case runs twice; no_claim=none.
  test("a failure that isn't about credit runs once", async () => {
    const asked: string[] = [];
    await runSteps([live], async (_step, secret) => {
      asked.push(secret);
      return run(1, "1 failed, a rate limit and a billing page were mentioned");
    });
    expect(asked).toEqual(["NIGHTLY_ANTHROPIC_KEY_1"]);
  });

  // protects=when both keys are out of credit the night says so and does not ask a third time; fails_when=a step runs with a key already known to be empty; no_claim=none.
  test("a step with no key left is reported as not run", async () => {
    const asked: string[] = [];
    const attempts = await runSteps([live, everyday], async (step, secret) => {
      asked.push(`${step.id} ${secret}`);
      return credit;
    });
    expect(asked).toEqual(["live NIGHTLY_ANTHROPIC_KEY_1", "live NIGHTLY_ANTHROPIC_KEY_2"]);
    expect(attempts.at(-1)).toMatchObject({ step: everyday, skipped: true, exitCode: 1 });
  });

  // protects=the words Anthropic and Kimi use for a used-up key are recognised; fails_when=a message from the provider's documentation no longer matches; no_claim=that the providers keep these words.
  test("recognises how the providers say a key is out of credit", () => {
    for (const message of [
      "Your credit balance is too low to access the Anthropic API. Please go to Plans & Billing to upgrade or purchase credits.",
      "You have reached your specified workspace API usage limits. You will regain access on 2026-11-01.",
      '{"error":{"type":"exceeded_current_quota_error","message":"insufficient balance"}}',
    ]) {
      expect(OUT_OF_CREDIT.test(message)).toBe(true);
    }
    expect(OUT_OF_CREDIT.test("rate limit exceeded, retry in 20 s")).toBe(false);
  });
});

describe("the spend", () => {
  // protects=the prices the night counts with are the catalog's, which the app's own cost meter uses; fails_when=the catalog changes a price or a model the night uses and its table doesn't; no_claim=that the prices are what the providers charge today.
  test("the night's price table is the catalog's", () => {
    const provider = (model: string) => (model.startsWith("claude") ? "anthropic" : "kimi");
    for (const [model, price] of Object.entries(RATES)) {
      expect(catalogFacts(provider(model), model)?.price).toEqual(price);
    }
  });

  // protects=a request is counted at its model's price, with the cache read at its cheaper rate and the cache write above the input price; fails_when=the cache is billed as ordinary input, or a Chinese price is added to dollars as yuan; no_claim=that the provider's invoice matches to the cent.
  test("counts tokens at the catalog's prices, in dollars", () => {
    const haiku = {
      model: "claude-haiku-5-5",
      inputTokens: 1_000_000,
      cachedInputTokens: 0,
      cacheWriteTokens: 0,
      outputTokens: 1_000_000,
    };
    expect(spendOf([haiku], "anthropic").dollars).toBeCloseTo(0.6);
    const cached = {
      ...haiku,
      inputTokens: 1_000_000,
      cachedInputTokens: 1_000_000,
      outputTokens: 0,
    };
    expect(spendOf([cached], "anthropic").dollars).toBeCloseTo(0.01);
    const kimi = {
      model: "kimi-k2.6",
      inputTokens: 0,
      cachedInputTokens: 0,
      cacheWriteTokens: 0,
      outputTokens: 1_000_000,
    };
    expect(spendOf([kimi], "kimi").dollars).toBeCloseTo(27 / 7.1);
    expect(spendOf([{ ...haiku, model: "claude-unknown-9" }], "anthropic")).toEqual({
      dollars: 0,
      unpricedTokens: 2_000_000,
    });
  });

  // protects=the tokens read from Anthropic's and Kimi's replies, one JSON body or a stream, are what the bill counts; fails_when=a stream's first count of output tokens is kept instead of its last, or Anthropic's cache tokens are left out of the input; no_claim=none.
  test("reads usage from a body and from a stream of events", () => {
    const stream = [
      'event: message_start\ndata: {"type":"message_start","message":{"model":"claude-haiku-5-5","usage":{"input_tokens":10,"cache_creation_input_tokens":200,"cache_read_input_tokens":3000,"output_tokens":1}}}',
      'event: message_delta\ndata: {"type":"message_delta","usage":{"output_tokens":42}}',
    ].join("\n\n");
    expect(usageOf(stream)).toEqual({
      inputTokens: 3210,
      cachedInputTokens: 3000,
      cacheWriteTokens: 200,
      outputTokens: 42,
    });
    const kimi =
      '{"id":"x","usage":{"prompt_tokens":500,"completion_tokens":60,"total_tokens":560,"cached_tokens":400}}';
    expect(usageOf(kimi)).toEqual({
      inputTokens: 500,
      cachedInputTokens: 400,
      cacheWriteTokens: 0,
      outputTokens: 60,
    });
  });
});

describe("the summary", () => {
  const result = (step: Attempt["step"], over: Partial<Result>): Result => ({
    step,
    secret: "NIGHTLY_ANTHROPIC_KEY_1",
    exitCode: 0,
    log: "",
    seconds: 90,
    outOfCredit: false,
    cases: [],
    usage: [],
    ...over,
  });

  // protects=the night's report shows each case's result, which key it used, a key found empty, and the spend; fails_when=a failed case is not named, or a key running dry goes unmentioned; no_claim=how the report looks.
  test("names the failed cases, a key out of credit and the spend", () => {
    const night = summary(
      [
        result(live, {
          outOfCredit: true,
          exitCode: 1,
          secret: "NIGHTLY_ANTHROPIC_KEY_1",
        }),
        result(live, {
          secret: "NIGHTLY_ANTHROPIC_KEY_2",
          cases: [{ name: "an Answer | cites", passed: true }],
          usage: [
            {
              model: "claude-haiku-5-5",
              inputTokens: 1_000_000,
              cachedInputTokens: 0,
              cacheWriteTokens: 0,
              outputTokens: 0,
            },
          ],
        }),
        result(everyday, {
          secret: "NIGHTLY_ANTHROPIC_KEY_2",
          exitCode: 1,
          cases: [
            { name: "owner-1 (en)", passed: false, detail: "1 of 3 runs passed, floor 67%" },
            { name: "owner-2 (en)", passed: true },
          ],
        }),
      ],
      "2026-10-23",
    );
    expect(night.passed).toBe(false);
    expect(night.text).toContain("# Nightly real-model checks, 2026-10-23");
    expect(night.text).toContain("- NIGHTLY_ANTHROPIC_KEY_1 is out of credit.");
    expect(night.text).toContain("Spend: $0.10");
    expect(night.text).toContain("| owner-1 (en) | **failed**: 1 of 3 runs passed, floor 67% |");
    expect(night.text).toContain("| an Answer \\| cites | passed |");
    expect(night.text).toContain("| 1 of 2 |");
  });
});
