# Tests and CI

Read this before writing code or tests, whether you are a person or a coding agent.

**The bar:** a change is done when its pull request's checks are green on GitHub, not when the tests pass on your computer. CI runs on clean machines that are slower than yours and have a different screen. A test that passes only on your computer is a broken test.

CI (`.github/workflows/ci.yml`) runs three checks on every push and pull request:

| Check | Where | What it runs |
|---|---|---|
| Typecheck, lint and unit tests | Ubuntu | `npm ci`, `npm run typecheck`, `npm run lint`, `npm test` |
| Electron smoke test (macOS) | macOS | `npm ci`, `npm run test:smoke`: builds the app and runs every spec in `e2e/` |
| Every commit is signed off (DCO) | Ubuntu | a `Signed-off-by:` line on each commit (`git commit -s`) |

`main` must stay green. A pull request is merged only when all three pass and it is up to date with `main`. If `main` goes red, fixing it comes before new work.

## Before you open a pull request

1. Merge the latest `main` into your branch.
2. Run what CI runs: `npm run typecheck`, `npm run lint`, `npm test`.
3. Run `npm run test:smoke`: the whole suite, or at least every spec that touches what you changed. To find them, search `e2e/` for the test ids, labels and strings you changed or removed.
4. When you change or remove UI, update or remove the tests that used it in the same pull request. A test still looking for something your change removed is broken by your change.

## After you open it

- Watch the checks until they finish: `gh pr checks <number> --watch`. Don't report the work as done, or ask for a merge, while a check is red or still running.
- When a check fails, read why: `gh run view <run id> --log-failed`, and for the smoke test, the Playwright traces uploaded as the `smoke-test-results` artifact. Fix it in this pull request, whether or not your change caused it, and say which in the pull request.
- "Flaky" is a claim that needs evidence: it failed, then passed on a rerun with nothing changed, and you can say what timing it depends on. Then fix the test. Don't hide it with more retries, longer timeouts or a skip.

## Tests that pass on any machine

- **Launch through `launchApp` in `e2e/app.ts`.** It gives each test a temporary data folder and home folder, test hooks, the fake embedding model and, with `fakeChat`, the scripted chat model. No test uses the network, an API key or the User's own files.
- **Set the window size the test needs,** with `setWindowSize` in `e2e/app.ts`. Never depend on the screen's size, the display's scale, the window being in front, or the computer's fonts and language. CI's macOS screen is not yours.
- **Assert relations, not exact pixels.** For example, "the controls sit under the text", "it wraps by word", or "it fits inside the card". Exact sizes are for what `DESIGN.md` fixes, such as a 44px band.
- **Wait for a state, not a time.** Use `expect(locator).toBeVisible()`, a test id or a value from the bridge, never `waitForTimeout` in new tests.
- **Drive the app like a person.** Move the mouse to an element before clicking or dragging it (as `slideToHandle`, `dragBy` and `dragBlock` in `e2e/app.ts` do), and use the keyboard where a person would. A helper needed in a second spec moves into `e2e/app.ts` rather than being copied.
- **Keep wall-clock limits out of unit tests.** CI machines run two to five times slower than a laptop. A speed budget belongs in a dedicated speed test with headroom, not in a unit test with a tight millisecond limit.
- **Make each test stand alone.** It must pass on its own and in any order, and it cleans up what it made.
- **Unit tests run in plain Node.** They must not need the Electron binary (CI doesn't download it) or a model download.
- **Write paths and text that work on every platform.** Use `path.join`, never assume `/` or a case-insensitive file system, and give the Chinese strings the same tests as the English ones.

## Trying AI features by hand

The scripted chat model (`fakeChat`) is for automated tests only: CI has no keys, and its runs must give the same result every time. Before you call an AI feature done (Answers, Citations, Organize, tagging, Tasks, anything a model does), use it by hand in the built app with a real model:

- **A cloud model on your own key.** A small, fast model, such as Anthropic's Haiku or Sonnet, is enough.
- **A small local model through Ollama.** For example `qwen3.5:4b`, with the context at 8,192 tokens or less, one model loaded at a time, and unloaded afterwards.
- **An Ollama cloud model, if your computer is busy.** Its name ends in `-cloud`. The app treats it as leaving the computer, which is fine for test Documents.

Use a temporary data folder and test Documents, never your own files. Say in the pull request which model you used and what you checked: that the Answer cites, the quotes are found, the feature reads well in Chinese, and it copes with a slow or failing model.

## In a pull request

The template's "How it was tested" section asks for the commands you ran and a green CI run. Paste the run's link.
