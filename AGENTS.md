# IncarnaMind: how to work on it

These rules are for everyone who changes this repository, people and coding agents alike (Claude Code, Codex, Cursor or any other). They are short; the detail lives in the files they point to. When you brief another agent, pass this file on.

## Words and decisions

- Use the words in [`CONTEXT.md`](CONTEXT.md) (a Passage, not a chunk; a Connector, not a plugin), in code, strings, tests and issues.
- Decisions that are hard to reverse are ADRs in [`docs/adr/`](docs/adr/). Read the ones your change touches, and say so in the pull request when a change goes against one.

## Agent skills

### Issue tracker

Issues live in GitHub Issues on `junruxiong/IncarnaMind`, managed with the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

The default five labels: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: `CONTEXT.md` and `docs/adr/` at the repo root. See `docs/agents/domain.md`.

## Writing code

1. **Read before you write.** Read the issue, the ADRs it touches and the code around the change. Use an existing module, helper or shared building block before adding one; when something is needed a second time, move it somewhere shared rather than copying it.
2. **Make the smallest change that fully does the job.** Don't add abstractions, options or flags that nothing needs yet. Don't take a shortcut that leaves the User a worse experience either.
3. **Respect the layers.**
   - `src/core` is the product and never imports Electron; Biome enforces this. Pass a capability in as an adapter instead.
   - `src/main` is the Electron shell.
   - The renderer reaches the core only through the bridge (`src/shared/bridge.ts`).
4. **Keep files focused.** Put a new feature in its own component, hook or core module rather than growing `App.tsx` or another long file.
5. **Fix types; don't silence them.** TypeScript is strict. No `any`, no `@ts-ignore`, and no `!` assertion just to quiet the compiler.
6. **Never swallow an error.** A failure the User can act on becomes a problem in plain words, where it happened (`DESIGN.md`, Problems). Log the rest with enough context to find it.
7. **Data changes are migrations** (`src/core/storage/migrations.ts`).
   - Take the next free number after the highest on `main`, and check it again after merging `main`.
   - Never edit a migration that has shipped.
   - Keep tables ready to sync: random UUID ids, ISO 8601 UTC timestamps, and soft deletes with `deleted_at`.
8. **Add few dependencies.** A new one must be open source under a permissive licence, maintained and small, and the pull request says why it's needed. No LangChain (ADR-0005). The lockfile changes only through `npm install`.
9. **Keep secrets and the User's data safe.**
   - No keys or tokens in code, logs, tests, issues or pull requests. Store secrets only through `src/core/secrets.ts`.
   - Nothing leaves the computer without the User's consent (ADR-0014, `src/core/consent.ts`).
10. **Every interface string comes in English and Simplified Chinese**, in `src/shared/i18n/`.
11. **Comments say why, not what,** at the density of the code around them.
12. **Work in your own branch and worktree.**
    - Don't change another checkout or the User's data folder, and don't drive the User's apps.
    - Use temporary data folders and a temporary home folder for anything you run.
    - Run `npm ci` in your own worktree before anything that rebuilds native modules, rather than rebuilding a shared `node_modules`.
13. **Read your own diff before you open the pull request,** as a reviewer would, against the issue's acceptance criteria and these rules. Answer review comments, the automated reviewer's included, by fixing them or by saying why not.

## Tests and CI

Read [`docs/agents/testing.md`](docs/agents/testing.md). Work is done when the pull request's CI checks are green on GitHub, not when the tests pass on your computer.

- Run what CI runs: `npm run typecheck`, `npm run lint`, `npm test`, and `npm run test:smoke` for the specs your change touches.
- Update or remove the tests for UI you changed.
- Watch `gh pr checks <number> --watch` and fix any red check before reporting.
- Tests must pass on any machine:
  - launch through `e2e/app.ts` and set the window size;
  - assert relations rather than exact pixels;
  - wait for states rather than times;
  - keep wall-clock limits out of unit tests.
- Test through the core's public interface, as the UI uses it, and assert what a User would see. Tests need no API keys and no network.

## UI

- Read `DESIGN.md` before visual work: the fonts, colours, spacing and direction. Ask before departing from it, and flag code that doesn't match it.
- Read [`docs/agents/interaction.md`](docs/agents/interaction.md) before building or changing any UI. The bar: easy to use and easy to pick up, not fewer features. In short:
  - borrow familiar conventions;
  - edit in place;
  - every drag has a non-drag alternative;
  - never replace the Mind the User is in;
  - Undo instead of confirmations;
  - one right-click menu order and one selection model;
  - the keyboard and ⌘K reach everything;
  - feedback within 100 ms;
  - layouts that hold at every width, in English and Chinese.
- A UI pull request ticks the template's interaction checklist, shows before and after screenshots, and sets the built screen beside the design board it implements, explaining every difference.

## Commits and pull requests

- Sign off every commit (`git commit -s`); see [`CONTRIBUTING.md`](CONTRIBUTING.md).
- One issue per pull request, kept small. Say what the User gets and why, and link the issue ("Closes #123").
- Merge the latest `main` before pushing. A pull request is merged only when its checks are green and it is up to date with `main`.
