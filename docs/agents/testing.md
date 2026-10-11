# Tests and CI

Read this before writing code or tests, whether you are a person or a coding agent.

**The bar:** a change is done when its pull request's checks are green on GitHub, not when the tests pass on your computer. CI runs on clean machines that are slower than yours and have a different screen. A test that passes only on your computer is a broken test.

CI (`.github/workflows/ci.yml`) runs three checks on every pull request (a push to `main` runs the first only: the smoke test passed on the pull request, which was up to date with `main`):

| Check | Where | What it runs |
|---|---|---|
| Typecheck, lint and unit tests | Ubuntu | `npm ci`, `npm run typecheck`, `npm run lint`, `npm test` |
| Electron smoke test (Linux) | Ubuntu | `npm ci`, then `npm run test:smoke` on a virtual display (`xvfb-run`): builds the app and runs every spec in `e2e/`, split into four shards that run at once (`Smoke shard N/4`); this check passes when all four do |
| Every commit is signed off (DCO) | Ubuntu | a `Signed-off-by:` line on each commit (`git commit -s`) |

`main` must stay green. A pull request is merged only when all three pass and it is up to date with `main`. If `main` goes red, fixing it comes before new work.

Every night, and by hand from the Actions tab, the Electron smoke test (macOS) in `.github/workflows/nightly.yml` runs the same suite on macOS, with the specs only macOS has, such as the traffic lights in the title bar. A failing night is fixed before new work, like a red `main`.

## Before you open a pull request

1. Merge the latest `main` into your branch.
2. Run what CI runs: `npm run typecheck`, `npm run lint`, `npm test`.
3. Run `npm run test:smoke`: the whole suite, or at least every spec that touches what you changed. To find them, search `e2e/` for the test ids, labels and strings you changed or removed.
4. When you change or remove UI, update or remove the tests that used it in the same pull request. A test still looking for something your change removed is broken by your change.
5. Try the edge cases below that apply to your change, by test or by hand, and say in the pull request which you tried and which don't apply.

## After you open it

- Watch the checks until they finish: `gh pr checks <number> --watch`. Don't report the work as done, or ask for a merge, while a check is red or still running.
- When a check fails, read why: `gh run view <run id> --log-failed`, and for the smoke test, the failing shard's log and the Playwright traces uploaded as its `smoke-test-results-linux-N` artifact (`smoke-test-results-macos` from the nightly run). Fix it in this pull request, whether or not your change caused it, and say which in the pull request.
- "Flaky" is a claim that needs evidence: it failed, then passed on a rerun with nothing changed, and you can say what timing it depends on. Then fix the test. Don't hide it with more retries, longer timeouts or a skip.

## Tests that pass on any machine

- **Launch through `launchApp` in `e2e/app.ts`.** It gives each test a temporary data folder and home folder, test hooks, the fake embedding model and, with `fakeChat`, the scripted chat model. No test uses the network, an API key or the User's own files.
- **Set the window size the test needs,** with `setWindowSize` in `e2e/app.ts`. Never depend on the screen's size, the display's scale, the window being in front, or the computer's fonts and language. CI's screens, a virtual one on Linux and a small one on macOS, are not yours.
- **Assert relations, not exact pixels.** For example, "the controls sit under the text", "it wraps by word", or "it fits inside the card". Exact sizes are for what `DESIGN.md` fixes, such as a 44px band.
- **Wait for a state, not a time.** Use `expect(locator).toBeVisible()`, a test id or a value from the bridge, never `waitForTimeout` in new tests.
- **Drive the app like a person.** Move the mouse to an element before clicking or dragging it (as `slideToHandle`, `dragBy` and `dragBlock` in `e2e/app.ts` do), and use the keyboard where a person would. A helper needed in a second spec moves into `e2e/app.ts` rather than being copied.
- **Keep wall-clock limits out of unit tests.** CI machines run two to five times slower than a laptop. A speed budget belongs in a dedicated speed test with headroom, not in a unit test with a tight millisecond limit.
- **Make each test stand alone.** It must pass on its own and in any order, and it cleans up what it made.
- **Unit tests run in plain Node.** They must not need the Electron binary (CI doesn't download it) or a model download.
- **Write paths and text that work on every platform.** Use `path.join`, never assume `/` or a case-insensitive file system, and give the Chinese strings the same tests as the English ones.
- **Write each spec for Linux and macOS.** Pull requests run the smoke test on Linux; macOS runs it each night. Press shortcuts with `ControlOrMeta+…`, never `Meta+…`, and expect a shortcut's label in either form (`⌘J` or `Ctrl+J`). A spec for something only macOS has, such as the traffic lights or full screen, skips elsewhere with `test.skip(process.platform !== "darwin", "<why>")`, and the nightly run covers it.

## Trying AI features by hand

The scripted chat model (`fakeChat`) is for automated tests only: CI has no keys, and its runs must give the same result every time. Before you call an AI feature done (Answers, Citations, Organize, tagging, Tasks, anything a model does), use it by hand in the built app with real models from **several providers**, because they differ in how they follow instructions, cite and format:

- **Anthropic:** Haiku first. Sonnet only when Haiku fails at the task or the feature needs the stronger model; say why in the pull request.
- **OpenAI:** GPT-6 Luna, the newest small model, priced like Haiku. Not GPT-6 Sol or Astra unless the feature needs them. Use the ChatGPT plan's allowance, through the ChatGPT test profile below, instead of an API key.
- **An Ollama cloud model** (its name ends in `-cloud`, for example a `gpt-oss` or `qwen3` cloud model). The app treats it as leaving the computer, which is fine for test Documents.
- **A small local model through Ollama,** for example `qwen3.5:4b`, with the context at 8,192 tokens or less, one model loaded at a time, and unloaded afterwards. Required when the feature runs in Local mode.

Use at least three of these, including one from Anthropic and one from OpenAI, and say where a model did worse than the others.

**Keep the bill small.** These keys are paid by the person, not a company. Use the cheapest model that answers the question, run each check once per model rather than many times, use short test Documents, and stop as soon as the result is clear. Large or "pro" models need a reason in the pull request.

Use a temporary data folder and test Documents, never your own files. Say in the pull request which models you used and what you checked with each: that the Answer cites, the quotes are found, the feature reads well in Chinese, and it copes with a slow or failing model.

### The ChatGPT test profile

The owner signs in with ChatGPT once, in a test profile; agents then use that sign-in for checks by hand on the owner's ChatGPT plan, without signing in themselves and without ever seeing a token.

- **What it is.** A folder outside every checkout and data folder, `~/.incarnamind-chatgpt-profile`. A development build (`npm run dev`) or a test build started with `INCARNAMIND_CHATGPT_PROFILE` set to it keeps the ChatGPT sign-in there instead of in its data folder: encrypted by the system's keychain (`safeStorage`) like every secret, in a file only the owner's account can read. Everything else (Minds, Documents, settings) stays in the run's own temporary data folder. Release builds ignore the variable, and the smoke tests clear it.
- **Several agents at once.** The tokens rotate when they are refreshed; a lock file in the folder makes processes take turns, so one refreshes and the others use its result.

**The owner's one-time sign-in** (and again if it ever ends, e.g. after 30 days without use):

```sh
export INCARNAMIND_CHATGPT_PROFILE="$HOME/.incarnamind-chatgpt-profile"
export INCARNAMIND_DATA_DIR="$(mktemp -d)"
npm run dev
```

No temporary HOME here: on macOS the app finds the login keychain, which encrypts the sign-in, through HOME (`~/Library/Keychains`). With an empty one there is no keychain, and the app says it can't store the sign-in before it opens OpenAI's page. `INCARNAMIND_DATA_DIR` already keeps this run out of the owner's data folder.

In the app, choose **ChatGPT plan** in the first window (or Settings → Models), then **Continue with ChatGPT**. Sign in with the ChatGPT Plus account in the browser, approve IncarnaMind and allow it to use the plan, then quit. In ChatGPT → Settings → Usage, the owner can cap IncarnaMind's weekly share of the plan.

**Using it as an agent** (on macOS):

```sh
export INCARNAMIND_CHATGPT_PROFILE="$HOME/.incarnamind-chatgpt-profile"
export INCARNAMIND_DATA_DIR="$(mktemp -d)"
TEMP_HOME="$(mktemp -d)"
mkdir -p "$TEMP_HOME/Library" && ln -s "$HOME/Library/Keychains" "$TEMP_HOME/Library/Keychains"
HOME="$TEMP_HOME" npm run dev
```

- **Why the link.** HOME stays temporary, so nothing the app writes there (Documents, dotfiles, other apps' settings) is the owner's. Only the keychains folder is linked, so the app reaches the login keychain, where macOS still decides what it may read. Don't link anything else.
- **No keychain prompt per worktree.** macOS knows the development app by its code signature, and every worktree on the same Electron version has the same Electron binary. After an Electron upgrade, macOS asks once to allow the new one; the owner answers that, not you.
- It starts signed in. GPT-6 Luna is the plan model to try first: choose it in the composer's model chip, or under Settings → Models (**Use for Questions** in the ChatGPT plan section, then the model in Chat model).
- Never read, copy, print or log anything in the profile folder, and never read another app's credentials (`~/.codex/auth.json` and the like). The app reads the profile; you don't.
- Never sign out in the profile: signing out ends the owner's session at OpenAI. Sign-out is covered by the unit tests' fake OpenAI.
- Keep checks small: they use the owner's plan, whose limits are shared with the owner's own ChatGPT.
- If the app says you're signed out of ChatGPT or can't store the sign-in, or macOS asks to allow access to the keychain, stop and ask the owner. Don't sign in or answer the prompt yourself.

## Edge cases

Most bugs a User meets sit at the edges, not on the path the feature was built for. Try each case that applies to your change, with an automated test where it can be one. The audit's fixtures (`e2e/audit/fixtures.ts`) already write many of these files.

- **Empty, one and many.** Nothing at all (an empty Mind, Folder, Library or field), a single item, and a lot: 2,000 Documents, a 300-page PDF, a 50,000-cell sheet, a Mind of about 300 Blocks.
- **Words.** Chinese, Chinese mixed with English, a very long name, emoji and unusual characters, and typing through a Chinese input method while it is still composing.
- **Broken input.** A damaged, empty, password-protected or oversized file; a scanned PDF with no text layer; a format IncarnaMind doesn't read; a file that disappears while it's being read.
- **No network and no consent.** Offline, the consent prompt declined, and Local mode.
- **Models that misbehave.** A slow model, one that fails or can't be reached, and one that returns no Citations or nonsense. Cancel halfway, and act while an Answer is still streaming.
- **Quitting halfway.** Force-quit in the middle of the action, then start again: nothing is lost, doubled or left half done.
- **Twice and back.** Do the action twice quickly (a double click, a held key), then Undo and Redo it.
- **Every way in.** Mouse and keyboard only, a narrow and a wide window, light and dark, English and Chinese.

## In a pull request

The template's "How it was tested" section asks for the commands you ran and a green CI run. Paste the run's link.
