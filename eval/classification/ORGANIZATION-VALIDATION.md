# Folder and tag organization — validation, 2026-10-08

The document UI follows the quiet sidebar, typography and surfaces of the [approved reference](https://claude.ai/artifact/KqhVAYczx9jz1EEHN1jFPS#page-dcb81aee8d83). Organization is now one action that assigns an in-app Folder and relevant Tags together. Original files stay at their source paths. Existing Group records become the Folders users see; no data migration or search re-index is needed.

The sidebar separates Folders from Source locations. The document sheet shows names, editable folders and visible tags, with model/error details behind each row's disclosure. Model options live in Settings → Organization. Folder choices are also available in a Question's `@` search scope.

## Results

| Check | Result |
| --- | --- |
| TypeScript and Biome | Passed |
| Full unit suite | 1,057 passed; 2 existing opt-in real search/model tests skipped |
| Full Electron smoke suite | 102 passed; the optional Settings screenshot test was initially skipped |
| Follow-up app checks | Settings screenshots and the affected sidebar, privacy, tags and Jev tests passed; final folder/tag and search-scope run: 7/7 passed |
| Live local organization | 7/7 cases returned the expected folder and tag with installed Ollama models |
| Production installer | Passed from both the unpacked app and the mounted read-only DMG; see the shutdown caveat below |

The app tests use disposable data folders. Cloud providers, connector services and chat answers use deterministic test adapters/local servers; this does not verify real cloud account credentials or every external service. No production document database was used.

## Workflows exercised

- Select starter folders; create, rename and delete a custom folder; browse and collapse folders; move a document manually, including to Unsorted.
- View, add and remove tags; create a custom tag; find documents by name or tag; preserve manual folder and tag corrections across another Organize action and a restart.
- Assign folder and tags in one inference request; reject invalid output without saving a partial result; discard stale results when definitions change; handle deleting the last folder during inference.
- Set a separate connected model, Auto local routing or an explicit Clef override; persist settings and page-image choices; show failures and retry; retain cloud consent and local-only guards.
- Organize new/changed linked files automatically; retain a manual folder choice while updating automatic tags. Verify original file paths/content survive organization and folder deletion.
- Switch between organized folders and source locations. Open documents in the viewer; check narrow layouts, keyboard dismissal and menu bounds. Inspect English and Chinese layouts and every Settings page.
- Use an in-app folder to limit a Question's retrieval and citations. Deleting a scoped folder yields an empty scope, never a search over all documents. Existing source-folder scopes continue to work.
- Regression suite: Minds, notes, tabs, answers, citations, PDF/Office/Markdown/text viewing, imports, linked-folder watching/pause/resume, search, exports, tags, onboarding, settings, privacy, connectors, skills and approvals.

## Live model check

`organization.validation.ts` calls the same combined folder/tag adapters as the app, without embeddings. Tev1 4B and Tev1 0.8B each handled a membrane experiment, a Chinese invoice and meeting actions. Clef-Flash handled an invoice whose information existed only in an attached image. All seven returned the expected folder and applicable tag; uncertain tag scores retain the existing review flag.

This is a small protocol and behavior check, not an accuracy estimate for arbitrary libraries. The earlier folder-only benchmark remains in `RESULTS.md` and `AUTO-VALIDATION.md`; it was not rerun as a folder-plus-tags benchmark. Only this 32 GB M2 Max was available. Slow/low-memory fallback is covered with simulated conditions, not a physical 8–16 GB computer.

## Reproduce

```sh
npm run typecheck
npm run lint
npm test
npm run test:smoke
INCARNAMIND_SCREENSHOTS=/tmp/incarnamind-ui-check npx playwright test e2e/shell.spec.ts --grep 'first run and every Settings page'
npx vitest run --config eval/classification/organization.config.ts
```

The live check requires Ollama with `tev1:4b`, `tev1:0.8b` and `clef-flash` installed. It unloads the models it uses afterwards. Raw results and UI screenshots are local, ignored artifacts under `eval/results/organization/`.

## Packaging

The local Apple-silicon test build is `dist/IncarnaMind-0.1.0-mac-arm64.dmg`. It is ad-hoc signed, not notarized or published. The production smoke test uses an isolated profile and mock keychain; it checks real PDF extraction and native page rendering against a deterministic local inference endpoint.

The complete smoke test passed from the unpacked production app and from the read-only mounted DMG, including normal application shutdown. An earlier run passed all document assertions but timed out during shutdown; a minimal fresh-profile launch also stalled once. A subsequent lifecycle probe reached `before-quit`, `will-quit` and `quit`, exiting in 370 ms, and both complete production reruns passed. The intermittent shutdown cause remains unconfirmed; no speculative lifecycle workaround was added.

Installer SHA-256: `ade7f6aaa5783b2117aa33fbe655860a209a2e3ad536eacff379d117c15cf995`.
