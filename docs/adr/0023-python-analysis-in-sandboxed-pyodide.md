---
status: accepted (2026-10-18)
---

# Python analysis runs in Pyodide, in a sandboxed hidden window

Analysts ask questions about spreadsheets and CSV files, such as "revenue by region and quarter: which grew fastest?". Models answer those badly in their heads and well with code. So the assistant will write code and run it on data cited from the User's Documents. Running code a model wrote is only safe in a sandbox tested against escape and network access, on every system we ship.

The spike in #444 measured one design on macOS, Windows and Linux. Its report and its rerunnable probes are on the branch `spike/444-pyodide`, in `docs/adr/0023-python-sandbox-spike-report.md` and `spike/python/`.

We run **Python first, offline, in Pyodide** (CPython compiled to WebAssembly), with numpy and pandas:

- **In a hidden, sandboxed renderer**, shaped like the meeting capture window (ADR-0015):
  - `sandbox: true` and `contextIsolation: true`, with no Node in the page, its Workers or its frames;
  - a preload that only passes messages;
  - a non-persistent session of its own;
  - every permission denied, and navigation, new windows and downloads refused.
- **Two walls against the network.** The page and its Worker get a Content-Security-Policy:
  `default-src 'none'; script-src <scheme>: 'wasm-unsafe-eval'; worker-src <scheme>:; connect-src <scheme>:; form-action 'none'; base-uri 'none'`.
  It allows no network and no `'unsafe-eval'`, and Pyodide runs under it. The session's own `webRequest` cancels every request but the bundled files' scheme, as a second wall.
- **What the spike tried, which failed on all three systems:**
  - Node, `require`, `process`, `Buffer`, the window's bridge, through Pyodide's `js` bridge and `pyodide.ffi`;
  - `eval` and `new Function`;
  - `import()` of `node:` and web modules, and `importScripts`;
  - `fetch` (to a web host, to an unknown host name, to `file:`), XMLHttpRequest, WebSocket, EventSource, WebTransport;
  - an image, a stylesheet, a form post, navigation, `window.open`, a download;
  - Python's `urllib`, `socket`, `pyodide.http.pyfetch` and micropip;
  - reading host files, which Python can't see: it has only Emscripten's in-memory file system;
  - writing files, which stay in memory;
  - storage carried from one run to the next (localStorage, sessionStorage, IndexedDB, the in-memory files).
  - In every case nothing left the window, and a control fetch from the main process reached the network.
- **One Worker.** Python runs in a module Worker in that window, started on the first analysis, never at launch.
  - It costs 180–370 MB of memory, and starting it takes 3–6 s (below).
  - It stays warm between calls: a call to a running Worker takes about a millisecond.
  - It is ended and started again after a time-out, a `MemoryError`, or a heap past a cap. Wasm memory never shrinks, so a Worker that hit the 4 GB ceiling keeps it until it ends.
- **Bundled, never fetched.** Pyodide's files and the wheels for numpy, pandas, python-dateutil, pytz and six ship in the app's resources and load from there. There is no CDN and no package index. They add about 14 MB to the installer compressed (21.4 MB unpacked), and the files are the same on every system. scipy, statsmodels and matplotlib are not bundled; they are added only if Users ask.
- **Linux:** Chromium's sandbox needs its setuid helper (`chrome-sandbox` owned by root, mode 4755, as a .deb installs it) or unprivileged user namespaces. The app is never run with `--no-sandbox`. Whether the AppImage gets a working sandbox on Ubuntu 24.04, where AppArmor restricts user namespaces, is a packaging question for the whole app, checked in #1116.

## Licences: an exception for Pyodide

`AGENTS.md` asks for permissive licences. Pyodide is MPL-2.0, a file-level copyleft licence. Accepting it is an exception, **approved by the project owner on 2026-10-17**, because nothing permissive runs Python in a sandbox. ADR-0016 already weighed an MPL-2.0 component and found it livable.

- Pyodide's files ship **unmodified, as their own files**: `pyodide.mjs`, `pyodide.asm.mjs`, `pyodide.asm.wasm`, `python_stdlib.zip` and `pyodide-lock.json`.
- They are **listed in the third-party licences**, with Pyodide's LICENSE at the shipped version and a link to its source. IncarnaMind stays Apache-2.0.
- If we ever change a Pyodide file, the changed file is shared under MPL-2.0.

The rest of what ships for data work:

| Part | Licence |
|---|---|
| CPython and its standard library, inside Pyodide | PSF-2.0 |
| Emscripten's runtime, inside Pyodide | MIT / University of Illinois NCSA |
| numpy | BSD-3-Clause, with the notices in its wheel's licence files |
| pandas | BSD-3-Clause |
| python-dateutil | Apache-2.0 or BSD-3-Clause |
| pytz | MIT |
| six | MIT |

## Measured

The spike used Pyodide 314.0.7 (Python 3.14.2), numpy 2.4.6 and pandas 3.0.2 on Electron 44.5.1. It ran on a local Apple M2 Max and on GitHub's macOS (arm64), Windows (x64) and Linux (x64) runners.

| | M2 Max (local) | macOS runner | Windows runner | Linux runner |
|---|---|---|---|---|
| Cold start, Worker to numpy + pandas imported | 3.0 s | 5.1 s | 6.1 s | 5.0 s |
| A new Worker in the same window | 2.9 s | 3.9 s | 5.9 s | 5.0 s |
| A call to a running Worker | ~1 ms | ~1 ms | ~1 ms | ~1 ms |
| pandas, 50,000-row CSV: parse + group-by | 14 + 12 ms | 19 + 18 ms | 21 + 24 ms | 19 + 21 ms |
| Renderer memory before Pyodide / after it | 90 / 460 MB | 89 / 448 MB | 69 / 269 MB | 95 / 288 MB |
| An endless loop: ended and a new Worker ready | 2.7 s | 3.6 s | 5.4 s | 4.1 s |
| Out of memory | `MemoryError` at 3,968 MB, then the Worker answers; the window lives | the same | the same | the same |

## The runtime interface

Python is the first language behind an interface in the core, written for more than one language: code in, values and tables out, with a time limit and a memory cap. JavaScript on QuickJS (MIT) comes later as a second language behind the same interface. The core never imports Electron: the desktop app supplies the sandboxed window as an adapter.

## Considered options

- **JavaScript on QuickJS first.** It is smaller and faster to start, and permissive. Rejected as the first language because models write data-analysis code more reliably in Python with pandas. It stays the second language.
- **A worker thread or utility process.** Rejected: Electron's Node-side processes are not sandboxed, so model-written code there could reach the file system and the network.
- **A server that runs the code.** Rejected: nothing leaves the computer without the User's consent (ADR-0022), and analysis should work offline.
- **Loading Pyodide and packages from its CDN.** Rejected: it would need the network, and the code that runs could change under us.

## Consequences

- The installer grows by about 14 MB on every system.
- The third-party licence notices gain Pyodide (MPL-2.0) and the packages above.
- `AGENTS.md`'s dependency rule names this exception and this record.
- ADR-0013 is amended: code run in this sandbox is a new Effect scope, "sandboxed code", that runs without asking. It reads only inputs cited from the Documents, which already run without asking, and it can neither write a file nor reach the network.
- The runtime is not offered in the hosted version until that version has a sandbox of its own.
- The window and its Worker are started on first use and kept alive after it, so the first analysis in a session waits 3–6 s; later ones don't.
