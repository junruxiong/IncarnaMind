# Spike report: Python (Pyodide with numpy and pandas) in a sandboxed hidden window

Issue #444, for the decision in #445. Branch `spike/444-pyodide`; the code is in `spike/python/`
and is not for merging. Measured on 2026-10-18 with Pyodide 314.0.7 (Python 3.14.2), numpy
2.4.6 and pandas 3.0.2, on Electron 44.5.1 (Chromium 152). The machines were a local Apple M2
Max (32 GB) and GitHub's `macos-latest` (arm64), `windows-latest` (x64) and `ubuntu-latest`
(x64) runners. The raw results are in `spike/python/measured/`.

## Recommendation: ship with changes

Nothing got out. No probe reached the network, a file on disk, Node, or the app's bridge on any
of the three systems. Endless loops and running out of memory both came back cleanly. The size
is about 13 MB compressed.

Cold start passes on a recent Mac, but warm start doesn't: starting a new Worker costs as much as
the first one. A Worker that runs out of memory also keeps about 4 GB until it ends. Both need
small changes, listed under "What to change".

## How it is built

- **A hidden window shaped like the meeting capture window** (`src/main/meetings/capture.ts`):
  - `sandbox: true`, `contextIsolation: true`, no Node in the page, its Workers or its frames, no `<webview>`;
  - its preload only passes messages (three functions: ready, reply, onRequest);
  - its own non-persistent session (`session.fromPartition("pyspike", { cache: false })`), where `webRequest` cancels every request but the bundled files' scheme;
  - every permission is denied, and navigation, new windows and downloads are refused.
- **The bundled files** are served by a privileged scheme (`pyspike:`) from the app's own files. The page and its Worker get this Content-Security-Policy:
  `default-src 'none'; script-src pyspike: 'wasm-unsafe-eval'; worker-src pyspike:; connect-src pyspike:; form-action 'none'; base-uri 'none'`.
  Pyodide runs without `'unsafe-eval'`.
- **Python runs in a module Worker**: Pyodide from the bundled files, then `loadPackage(["numpy", "pandas"])` from the same place. There's no package index and no CDN, and micropip isn't bundled.
- **The files ship unmodified.** `fetch.mjs` takes Pyodide's core from npm and the wheels from Pyodide's release, checking each wheel against the sha256 in `pyodide-lock.json`.

## Numbers

| | M2 Max (local) | macOS runner | Windows runner | Linux runner |
|---|---|---|---|---|
| Cold start, Worker to numpy + pandas imported | 3.0 s | 5.1 s | 6.1 s | 5.0 s |
| … of which Pyodide core / packages / imports | 1.0 / 0.4 / 1.6 s | 2.2 / 0.6 / 2.1 s | 1.9 / 0.7 / 2.9 s | 1.7 / 0.7 / 2.6 s |
| Window opened to ready | 3.8 s | 6.2 s | 6.7 s | 5.6 s |
| "Warm" start: a new Worker in the same window | 2.9 s | 3.9 s | 5.9 s | 5.0 s |
| A Python call in a Worker already running (`1 + 1`) | ~1 ms | ~1 ms | ~1 ms | ~1 ms |
| pandas, 20,000-row CSV: parse + group-by | 19 + 18 ms | 26 + 28 ms | 48 + 47 ms | 39 + 43 ms |
| pandas, 50,000-row CSV: parse + group-by | 14 + 12 ms | 19 + 18 ms | 21 + 24 ms | 19 + 21 ms |
| Renderer memory before Pyodide / at rest after it | 90 / 460 MB | 89 / 448 MB | 69 / 269 MB | 95 / 288 MB |
| Peak during the 50,000-row group-by | 486 MB | 476 MB | 293 MB | 319 MB |
| Wasm heap at rest | 75 MB | 75 MB | 75 MB | 75 MB |

The group-by test parses a sheet of region, quarter, product and revenue, sums revenue by region
and quarter (96 groups), and finds the three regions whose quarters grew fastest.

The 50,000-row run is faster than the 20,000-row one because it runs second, once the code is warm.

macOS reports a larger working set than the other systems for the same 75 MB heap.

**Size added:** 21.4 MB raw. That's 14.1 MB gzipped (level 9) and 13.0 MB with Brotli (level 11), the files being the same on every system. The parts:

| File | Raw |
|---|---|
| `pyodide.asm.wasm` | 9.6 MB |
| pandas wheel | 4.2 MB |
| numpy wheel | 3.0 MB |
| `python_stdlib.zip` | 2.5 MB |
| `pyodide.asm.mjs` | 1.25 MB |
| pytz | 0.5 MB |
| dateutil | 0.23 MB |
| the lock file, `pyodide.mjs` and six | together 0.15 MB |

## Probes

Every probe ran on all three systems with the same result. "Blocked" means nothing left the window. A control from the main process fetched `https://example.com/` with status 200 everywhere, so the network was there.

The page's Content-Security-Policy stopped each network probe before it became a request. To test the session's own cancelling as a second layer, the same address was fetched through the sandbox's session from the main process, and it was cancelled (`ERR_BLOCKED_BY_CLIENT`).

| Probe | Result |
|---|---|
| **Escape** | |
| `process`, `require`, `Buffer`, `electron`, the bridge, from Python (`js`, `js.globalThis`) and in the Worker | All absent. The Worker has no `spikeBridge` either; only the page has its three message functions. |
| `js.eval`, `js.Function.new`, `eval`, `new Function` | Refused by the CSP (no `'unsafe-eval'`) |
| `import("node:fs")`, `import("https://…")` | Fail |
| `importScripts("https://…")` | Throws: a module Worker has no `importScripts` |
| `postMessage` to the page | The one channel, by design. The page passes it on only through the preload's three functions. |
| Renderer process | `process.sandboxed` and `contextIsolated` are true on all three |
| **Network** | |
| `fetch`, a made-up host name (DNS), `fetch("file:///etc/hosts")` | Blocked |
| XMLHttpRequest, WebSocket, EventSource, WebTransport | Blocked |
| An image and a stylesheet from a web address | Blocked |
| A form posted to a web address, navigation, `window.open` | Refused by `will-navigate` and the window-open handler |
| Python's `urllib`, `socket`, `pyodide.http.pyfetch` | Fail: no TLS, "socket not connected", "failed to fetch" |
| `import micropip`, `loadPackage("micropip")` | Fail: not bundled, and there's no package index |
| **Files** | |
| Python reading `/etc/hosts`, `~/.ssh/id_rsa`, `C:/Windows/win.ini` | Not found. Python sees only Emscripten's in-memory file system: `/dev`, `/home`, `/lib`, `/proc`, `/tmp`. |
| Python writing `/tmp/…` | Written in memory; nothing appears on the disk |
| A download from a link | Cancelled in `will-download` |
| localStorage, sessionStorage, IndexedDB and the in-memory files across two runs | All empty in the second run |
| **Limits** | |
| `while True: pass` with a 3 s limit | Times out. The page ends the Worker and starts another, which answers in 3.6–5.4 s on the runners and 2.7 s on the M2. |
| Allocating 128 MB blocks until memory runs out | A Python `MemoryError` after 3,968 MB (wasm32's 4 GB ceiling). The Worker keeps working and the window lives. The Worker keeps its 4 GB heap, as Wasm memory never shrinks: the renderer peaked at 4.26 GB on Windows and Linux. |

## What to change before shipping

1. **Keep the Worker running** after its first use, rather than starting one per call. A call to a running Worker takes milliseconds; a new one takes 3–6 s, warm or not.
   - Start it on the first analysis, not at app start: it costs 180–370 MB of renderer memory.
   - If warm start must be under 1 s even after a restart, try Pyodide's memory snapshot in a follow-up spike.
2. **Bound memory.** After a call that raises `MemoryError`, or whose heap grows past a limit (say 1 GB), end the Worker and start a new one, so the 4 GB is given back. Show the User a plain problem: "the data was too large to analyse here".
3. **Time limit:** end and restart the Worker, as the spike does. Pyodide's interrupt buffer would need cross-origin isolation (COOP/COEP) for a `SharedArrayBuffer`, which the spike didn't try.
4. **Linux:** Chromium's sandbox needs the setuid helper (`chrome-sandbox` owned by root, mode 4755, as a .deb installs it) or unprivileged user namespaces.
   - Without either, Electron stops at start-up.
   - The app must never run this window, or the app, with `--no-sandbox`.
   - On Ubuntu 23.10 and later, AppArmor restricts user namespaces, so an AppImage needs a check. The Linux runner passed with the setuid helper.
5. **Keep the CSP as the spike has it.** It stopped every network probe before a request was made. The session's cancelling is the second layer.

## Windows

Nothing differed on Windows except speed. The runner was the slowest: cold start 6.1 s, group-by 21 + 24 ms on 50,000 rows. Memory and every probe were the same as elsewhere, and there was nothing Windows-specific to configure.

## Licences of what would ship

| Part | Licence |
|---|---|
| Pyodide (`pyodide.mjs`, `pyodide.asm.mjs`, `pyodide.asm.wasm`, the lock file) | MPL-2.0: the exception #445 records |
| CPython and its standard library (in `pyodide.asm.wasm`, `python_stdlib.zip`) | PSF-2.0 |
| Emscripten's runtime (in `pyodide.asm.*`) | MIT / University of Illinois NCSA |
| numpy 2.4.6 | BSD-3-Clause, with the notices for what it bundles in its wheel's licence file |
| pandas 3.0.2 | BSD-3-Clause |
| python-dateutil 2.9.0.post0 | Apache-2.0 or BSD-3-Clause, dual |
| pytz 2026.1.post1 | MIT |
| six 1.17.0 | MIT |

The npm package carries no licence file. The app's third-party notices should copy Pyodide's
LICENSE from its source at the shipped version, and each wheel's licence files from the wheel.

## Running it again

```sh
node spike/python/fetch.mjs   # Pyodide and the wheels into spike/python/dist (21 MB)
node spike/python/run.mjs     # two runs with a temporary home folder; results in spike/python/results
```

On Linux, run it under `xvfb-run` with the setuid helper, as `.github/workflows/pyspike.yml` did.
That workflow was temporary and is removed from the branch.
