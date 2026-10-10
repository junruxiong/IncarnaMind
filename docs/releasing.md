# Releasing

For maintainers: packaging, cutting a release, signing, and crash reports.

[electron-builder](https://www.electron.build) packages the app (config: `electron-builder.yml`), and electron-updater updates installed apps from GitHub Releases (`src/main/updater.ts`). To package locally, into `dist/`:

```shell
npm run dist:dir     # the unpacked app, for a quick check (e.g. dist/mac-arm64/IncarnaMind.app)
npm run dist         # the installers for the current system; never published
```

## Cutting a release

1. Set the version, commit it and tag it: `npm version 1.2.3`. This updates `package.json` and `package-lock.json`, commits, and creates the tag `v1.2.3`.
2. Push the commit and the tag: `git push && git push origin v1.2.3`.
3. The tag starts the **Release** workflow (`.github/workflows/release.yml`). It checks that the tag matches `package.json`, creates a draft GitHub Release, and builds and uploads:
   - for macOS, a dmg and a zip, each for Apple silicon and Intel;
   - for Windows, an NSIS installer;
   - for Linux, an AppImage;
   - the `latest*.yml` files that auto-update reads.
4. Review the draft, edit its notes, try the installers, and click **Publish release**. Installed apps only see an update once its release is published.

After the first release is published, follow the comments in `README.md` and `README.zh-CN.md`: remove the "not released yet" notes, and add the release badge.

## macOS signing and notarization

The workflow signs and notarizes the macOS app when these repository secrets exist. Without `CSC_LINK`, it builds an unsigned (ad-hoc signed) app and still succeeds.

| Secret | Value |
| --- | --- |
| `CSC_LINK` | The "Developer ID Application" certificate and its private key, exported from Keychain Access as a `.p12` file and base64-encoded (`base64 -i certificate.p12 \| pbcopy`) |
| `CSC_KEY_PASSWORD` | The password of that `.p12` file |
| `APPLE_ID`, `APPLE_APP_SPECIFIC_PASSWORD`, `APPLE_TEAM_ID` | To notarize with an Apple ID: the Apple ID, an [app-specific password](https://support.apple.com/102654) and the team ID |
| `APPLE_API_KEY`, `APPLE_API_KEY_ID`, `APPLE_API_ISSUER` | Or, to notarize with an App Store Connect API key: the contents of the `.p8` file, its key ID and the issuer ID |

The Windows installer stays unsigned for now; signing through the SignPath Foundation is a follow-up.

An unsigned macOS build can't update itself: IncarnaMind tells the user a new version is out and offers its download page.

## Crash reports

Users can opt in to crash reports in **Settings → Privacy**; they go to Sentry, scrubbed of file paths, Document text, Mind content, Questions and Answers (`src/main/crashScrubber.ts`). Only a build made with a Sentry DSN offers them: set `MAIN_VITE_SENTRY_DSN` when building (electron-vite reads it from the environment or a `.env.local` file). The release workflow passes the `SENTRY_DSN` repository secret; without it, releases don't offer crash reports. Never commit a DSN.
