/* Spike for #1272. The pane's own page: it pairs with the app it was served from,
   reads the open document, and writes back what the app names. */
(() => {
  const TOKEN_KEY = "incarnamind.office.token";
  const $ = (id) => document.getElementById(id);
  const report = { origin: location.origin, secureContext: window.isSecureContext };
  let host = null;
  let token = null;
  let channel = null;

  const store = {
    get() {
      try {
        return localStorage.getItem(TOKEN_KEY);
      } catch {
        return null;
      }
    },
    set(value) {
      try {
        localStorage.setItem(TOKEN_KEY, value);
      } catch {
        report.storage = "blocked";
      }
    },
    clear() {
      try {
        localStorage.removeItem(TOKEN_KEY);
      } catch {
        report.storage = "blocked";
      }
    },
  };

  const say = (text) => {
    $("status").textContent = text;
  };
  const show = (id, visible) => {
    $(id).hidden = !visible;
  };
  const describe = (error) => (error && error.message ? error.message : String(error));
  const showReport = () => {
    $("report").value = JSON.stringify(report, null, 2);
  };

  async function api(path, { method = "GET", body } = {}) {
    const response = await fetch(path, {
      method,
      cache: "no-store",
      headers: {
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
        ...(body ? { "Content-Type": "application/json" } : {}),
      },
      ...(body ? { body: JSON.stringify(body) } : {}),
    });
    const text = await response.text();
    return { status: response.status, json: text ? JSON.parse(text) : null };
  }

  function revoked() {
    token = null;
    store.clear();
    closeChannel();
    show("paired", false);
    show("pairing", true);
    say("This pairing ended. Pair again with a new code.");
  }

  function closeChannel() {
    if (channel) channel.close();
    channel = null;
  }

  function openWebSocket() {
    return new Promise((resolve) => {
      let socket;
      try {
        socket = new WebSocket(`${location.protocol === "http:" ? "ws" : "wss"}://${location.host}/ws`);
      } catch (error) {
        report.websocket = `failed: ${describe(error)}`;
        return resolve(false);
      }
      const timer = setTimeout(() => {
        report.websocket = "failed: no answer in 5 s";
        socket.close();
        resolve(false);
      }, 5000);
      socket.onopen = () => socket.send(JSON.stringify({ type: "hello", token }));
      socket.onmessage = (event) => {
        const message = JSON.parse(event.data);
        if (message.type === "ready") {
          clearTimeout(timer);
          report.websocket = "ok";
          channel = { close: () => socket.close() };
          resolve(true);
        } else if (message.type === "revoked") {
          revoked();
        }
      };
      socket.onerror = () => {
        clearTimeout(timer);
        report.websocket = "failed: the socket reported an error";
        resolve(false);
      };
      socket.onclose = (event) => {
        clearTimeout(timer);
        if (!report.websocket) report.websocket = `closed before ready: code ${event.code}`;
        resolve(false);
      };
    });
  }

  async function openStream() {
    try {
      const controller = new AbortController();
      const response = await fetch("/api/events", {
        headers: { Authorization: `Bearer ${token}` },
        signal: controller.signal,
      });
      if (!response.ok || !response.body) throw new Error(`status ${response.status}`);
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let pending = "";
      channel = { close: () => controller.abort() };
      report.stream = "ok";
      (async () => {
        for (;;) {
          const { done, value } = await reader.read();
          if (done) return;
          pending += decoder.decode(value, { stream: true });
          for (let end = pending.indexOf("\n\n"); end >= 0; end = pending.indexOf("\n\n")) {
            const block = pending.slice(0, end);
            pending = pending.slice(end + 2);
            if (block.startsWith("event: revoked")) revoked();
          }
        }
      })().catch(() => undefined);
      return true;
    } catch (error) {
      report.stream = `failed: ${describe(error)}`;
      return false;
    }
  }

  async function listen() {
    closeChannel();
    if (!(await openWebSocket())) await openStream();
    showReport();
  }

  async function connect() {
    let ping;
    try {
      ping = await api("/api/ping");
      report.ping = `ok (${ping.status})`;
    } catch (error) {
      report.ping = `failed: ${describe(error)}`;
      show("pairing", false);
      show("paired", false);
      say("Open IncarnaMind to ask and check.");
      return showReport();
    }
    token = store.get();
    if (token) {
      const session = await api("/api/session");
      if (session.status === 200) return paired(session.json);
      token = null;
      store.clear();
    }
    show("paired", false);
    show("pairing", true);
    say("Type the code IncarnaMind shows to pair this pane.");
    showReport();
  }

  async function paired(session) {
    show("pairing", false);
    show("paired", true);
    $("folder").textContent = `Connected to IncarnaMind on this computer. Searches “${session.folder}”.`;
    $("run").textContent =
      host === "word"
        ? "Comment on the sentence IncarnaMind names"
        : "Add a note to the cell IncarnaMind names";
    say("Paired.");
    await listen();
  }

  async function pair() {
    const code = $("code").value.trim();
    const result = await api("/api/pair", { method: "POST", body: { code, host } });
    if (result.status !== 200) {
      const reasons = {
        wrong: "That code is not right.",
        locked: "Too many wrong codes. Ask IncarnaMind for a new one.",
        expired: "That code has expired. Ask IncarnaMind for a new one.",
        "no-code": "IncarnaMind is not showing a code. Ask it for one.",
      };
      say(reasons[result.json && result.json.error] || "IncarnaMind refused the code.");
      return;
    }
    token = result.json.token;
    store.set(token);
    $("code").value = "";
    await paired((await api("/api/session")).json);
  }

  async function unpair() {
    await api("/api/pair", { method: "DELETE" });
    revoked();
    say("Disconnected.");
  }

  const sentenceAnchor = (text) => text.replaceAll("^", "^^");

  async function commentOnWord() {
    const paragraphs = await Word.run(async (context) => {
      const items = context.document.body.paragraphs;
      items.load("items/text");
      await context.sync();
      return items.items.map((paragraph) => paragraph.text);
    });
    const { json } = await api("/api/word/review", { method: "POST", body: { paragraphs } });
    const finding = json.finding;
    if (!finding) return "IncarnaMind found nothing to comment on.";
    await Word.run(async (context) => {
      const body = context.document.body;
      const limit = 250;
      const text = sentenceAnchor(finding.sentence);
      const first = body.search(text.slice(0, limit), { matchCase: true });
      first.load("items");
      const last = text.length > limit ? body.search(text.slice(-limit), { matchCase: true }) : null;
      if (last) last.load("items");
      await context.sync();
      if (first.items.length === 0) throw new Error("The sentence is not in the document any more.");
      const range = last && last.items.length > 0
        ? first.items[0].expandTo(last.items[last.items.length - 1])
        : first.items[0];
      range.insertComment(finding.comment);
      await context.sync();
    });
    return `Commented on “${finding.sentence}”`;
  }

  async function noteInExcel() {
    if (!Office.context.requirements.isSetSupported("ExcelApi", "1.18")) {
      return "This Excel can't add notes from an add-in (it needs ExcelApi 1.18).";
    }
    const read = await Excel.run(async (context) => {
      const range = context.workbook.getSelectedRange();
      range.load(["values", "formulas", "address"]);
      await context.sync();
      return { values: range.values, formulas: range.formulas, address: range.address };
    });
    const { json } = await api("/api/excel/review", {
      method: "POST",
      body: { values: read.values, formulas: read.formulas },
    });
    const finding = json.finding;
    if (!finding) return "IncarnaMind found no cell to add a note to.";
    await Excel.run(async (context) => {
      const cell = context.workbook.getSelectedRange().getCell(finding.row, finding.column);
      context.workbook.worksheets.getActiveWorksheet().notes.add(cell, finding.note);
      await context.sync();
    });
    return `Added a note at row ${finding.row + 1}, column ${finding.column + 1} of ${read.address}`;
  }

  async function run() {
    $("result").textContent = "Working…";
    try {
      $("result").textContent = host === "word" ? await commentOnWord() : await noteInExcel();
    } catch (error) {
      $("result").textContent = `Could not finish: ${describe(error)}`;
    }
  }

  async function testConnection() {
    report.testedAt = new Date().toISOString();
    for (const key of ["ping", "websocket", "stream"]) delete report[key];
    try {
      report.ping = `ok (${(await api("/api/ping")).status})`;
    } catch (error) {
      report.ping = `failed: ${describe(error)}`;
    }
    if (token) await listen();
    showReport();
  }

  function start(info) {
    host = info.host === Office.HostType.Word ? "word" : "excel";
    report.officeHost = String(info.host);
    report.platform = String(info.platform);
    report.userAgent = navigator.userAgent;
    const diagnostics = Office.context.diagnostics;
    if (diagnostics) report.officeVersion = String(diagnostics.version);
    $("pair").addEventListener("click", () => void pair());
    $("unpair").addEventListener("click", () => void unpair());
    $("run").addEventListener("click", () => void run());
    $("test").addEventListener("click", () => void testConnection());
    if (info.platform === Office.PlatformType.OfficeOnline) {
      say(
        "Office on the web can't reach IncarnaMind on this computer. Open this file in Word or Excel on the computer where IncarnaMind is installed.",
      );
      show("diagnostics", false);
      return;
    }
    void connect();
  }

  Office.onReady(start);
})();
