/**
 * The fake OpenAI (`./fakeOpenAI`) for the unit tests: stopped when the test
 * finishes, with a fake browser that plays the User signing in.
 */
import { onTestFinished } from "vitest";
import type { Browser } from "../../src/core";
import { type Approval, createFakeOpenAI, type FakeOpenAIServer } from "./fakeOpenAI";

export * from "./fakeOpenAI";

/** Starts the fake on a free port; stopped when the test finishes. */
export async function startFakeOpenAI(
  options: { now?: () => number } = {},
): Promise<FakeOpenAIServer> {
  const fake = await createFakeOpenAI(options);
  onTestFinished(() => fake.close());
  return fake;
}

export interface BrowserVisit {
  status: number;
  html: string;
}

export interface FakeBrowser extends Browser {
  /** Every URL the app opened. */
  opened: string[];
  /** The page the loopback server showed for each redirect, once it arrives. */
  visits: Promise<BrowserVisit>[];
  /**
   * What the User does on OpenAI's page: "approve" signs in, "decline"
   * refuses, "ignore" walks away.
   */
  behaviour: "approve" | "decline" | "ignore";
  /** How the User approves: which account, and whether they allow plan use. */
  approval: Approval;
}

/** A browser in which the User signs in to the fake OpenAI, redirecting back like the real page. */
export function createFakeBrowser(fake: FakeOpenAIServer): FakeBrowser {
  const browser: FakeBrowser = {
    opened: [],
    visits: [],
    behaviour: "approve",
    approval: {},
    async open(url) {
      browser.opened.push(url);
      fake.authorizations.push(Object.fromEntries(new URL(url).searchParams));
      if (browser.behaviour === "ignore") return;
      const redirect = fake.redirectFor(
        url,
        browser.behaviour === "approve" ? { approve: browser.approval } : "decline",
      );
      // Like a real browser, the redirect happens after `open` returns.
      browser.visits.push(
        new Promise((resolve) => setTimeout(resolve, 5)).then(async () => {
          const response = await fetch(redirect);
          return { status: response.status, html: await response.text() };
        }),
      );
    },
  };
  return browser;
}
