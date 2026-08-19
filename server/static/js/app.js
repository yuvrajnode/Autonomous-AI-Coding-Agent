/* Dashboard for the agent.
 *
 * No framework on purpose: the whole UI is one websocket stream plus four
 * REST endpoints, and a build step would cost more than it buys here.
 */

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

const api = {
  async get(path) {
    const res = await fetch(path);
    if (!res.ok) throw new Error(await describe(res));
    return res.json();
  },
  async post(path, body) {
    const res = await fetch(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!res.ok) throw new Error(await describe(res));
    return res.json();
  },
  async del(path) {
    const res = await fetch(path, { method: "DELETE" });
    if (!res.ok) throw new Error(await describe(res));
    return res.json();
  },
};

async function describe(res) {
  try {
    const body = await res.json();
    return body.detail ? JSON.stringify(body.detail) : `${res.status} ${res.statusText}`;
  } catch {
    return `${res.status} ${res.statusText}`;
  }
}

/* ------------------------------------------------------------------ util */

const esc = (value) =>
  String(value ?? "").replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]),
  );

const clip = (text, n = 220) => {
  const s = String(text ?? "");
  return s.length > n ? `${s.slice(0, n)}…` : s;
};

const money = (usd) => (usd >= 0.01 ? `$${usd.toFixed(3)}` : `$${(usd ?? 0).toFixed(4)}`);
const pct = (v) => `${Math.round((v ?? 0) * 100)}%`;

function ago(epochSeconds) {
  const secs = Math.max(0, Date.now() / 1000 - epochSeconds);
  if (secs < 60) return `${Math.round(secs)}s ago`;
  if (secs < 3600) return `${Math.round(secs / 60)}m ago`;
  if (secs < 86400) return `${Math.round(secs / 3600)}h ago`;
  return `${Math.round(secs / 86400)}d ago`;
}

let toastTimer;
function toast(message, tone = "info") {
  const el = $("#toast");
  el.textContent = message;
  el.dataset.tone = tone;
  el.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => (el.hidden = true), 4200);
}

function emptyState(title, hint) {
  return `<li class="empty"><strong>${esc(title)}</strong>${esc(hint)}</li>`;
}

/* ----------------------------------------------------------------- theme */

function initTheme() {
  const saved = localStorage.getItem("aca-theme");
  const system = window.matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark";
  applyTheme(saved || system);
  $("#theme-toggle").addEventListener("click", () => {
    const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
    applyTheme(next);
    localStorage.setItem("aca-theme", next);
  });
}

function applyTheme(theme) {
  document.documentElement.dataset.theme = theme;
  const button = $("#theme-toggle");
  button.setAttribute("aria-pressed", String(theme === "dark"));
  button.textContent = theme === "dark" ? "Light theme" : "Dark theme";
}

/* ---------------------------------------------------------------- router */

const loaders = {};

function initNav() {
  $$(".nav-item").forEach((button) => {
    button.addEventListener("click", () => show(button.dataset.view));
  });
  const initial = location.hash.replace("#", "") || "runs";
  show(loaders[initial] ? initial : "runs");
}

function show(view) {
  $$(".nav-item").forEach((b) => b.classList.toggle("is-active", b.dataset.view === view));
  $$(".view").forEach((s) => s.classList.toggle("is-active", s.dataset.view === view));
  location.hash = view;
  loaders[view]?.();
}

/* ------------------------------------------------------------------ runs */

const live = {
  runId: null,
  socket: null,
  plan: [],
  startedAt: null,
  stats: { tools: 0, failures: 0, llm: 0, input: 0, output: 0, usd: 0, iterations: 0 },
};

function initComposer() {
  const form = $("#composer");
  const button = $("#run-btn");
  const error = $("#composer-error");

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const task = $("#task").value.trim();
    error.hidden = true;
    if (task.length < 8) {
      error.textContent = "Describe the task in a sentence or two - eight characters is not enough to plan from.";
      error.hidden = false;
      $("#task").focus();
      return;
    }

    button.disabled = true;
    button.classList.add("is-busy");
    try {
      const workspace = $("#workspace").value.trim() || null;
      const { id } = await api.post("/api/runs", { task, workspace });
      $("#task").value = "";
      await loadRuns();
      attach(id, task);
      toast("Run started", "ok");
    } catch (err) {
      error.textContent = `Could not start the run: ${err.message}`;
      error.hidden = false;
      toast(err.message, "error");
    } finally {
      button.disabled = false;
      button.classList.remove("is-busy");
    }
  });
}

async function loadRuns() {
  const list = $("#run-list");
  try {
    const { runs } = await api.get("/api/runs?limit=30");
    if (!runs.length) {
      list.innerHTML = emptyState("No runs yet", "Describe a task above and the timeline will fill in as the agent works.");
      return;
    }
    list.innerHTML = runs
      .map(
        (run) => `
        <li>
          <button class="run-item ${run.id === live.runId ? "is-selected" : ""}" data-id="${esc(run.id)}" type="button">
            <span class="task">${esc(clip(run.task, 84))}</span>
            <span class="meta">${esc(run.status)} · ${esc(ago(run.created_at))}${
              run.duration_s ? ` · ${run.duration_s}s` : ""
            }</span>
          </button>
        </li>`,
      )
      .join("");
    $$(".run-item", list).forEach((button) =>
      button.addEventListener("click", () => attach(button.dataset.id)),
    );
  } catch (err) {
    list.innerHTML = emptyState("Could not reach the API", err.message);
  }
}

async function loadRollup() {
  try {
    const { rollup } = await api.get("/api/metrics");
    $("#rollup").innerHTML = [
      ["runs", rollup.runs],
      ["succeeded", pct(rollup.success_rate)],
      ["avg steps", rollup.avg_iterations],
      ["tool errors", pct(rollup.tool_failure_rate)],
      ["spend", money(rollup.total_usd)],
    ]
      .map(([label, value]) => `<div class="stat"><b>${esc(value)}</b><span>${esc(label)}</span></div>`)
      .join("");
  } catch {
    $("#rollup").innerHTML = "";
  }
}

function resetLive(task) {
  live.plan = [];
  live.startedAt = Date.now();
  live.stats = { tools: 0, failures: 0, llm: 0, input: 0, output: 0, usd: 0, iterations: 0 };
  $("#timeline").innerHTML = "";
  $("#plan").innerHTML = emptyState("Waiting for a plan", "The planner runs first; steps appear here.");
  $("#plan-progress").textContent = "";
  $("#run-summary").textContent = task ? `Working on: ${task}` : "Nothing yet.";
  $("#run-summary").classList.remove("is-final");
  renderRunMetrics();
}

function attach(runId, task) {
  if (live.socket) live.socket.close();
  live.runId = runId;
  resetLive(task);
  setStatus("queued");
  $$(".run-item").forEach((b) => b.classList.toggle("is-selected", b.dataset.id === runId));

  const proto = location.protocol === "https:" ? "wss" : "ws";
  const socket = new WebSocket(`${proto}://${location.host}/ws/runs/${runId}`);
  live.socket = socket;

  socket.addEventListener("message", (message) => {
    const payload = JSON.parse(message.data);
    if (payload.type === "run.closed") {
      setStatus(payload.data.status);
      loadRuns();
      loadRollup();
      return;
    }
    if (payload.type === "error") {
      toast(payload.detail, "error");
      return;
    }
    handleEvent(payload);
  });

  socket.addEventListener("error", () => toast("Lost the connection to the run stream", "error"));
}

function setStatus(status) {
  const badge = $("#run-status");
  badge.dataset.status = status;
  badge.textContent = status;
}

/* -------------------------------------------------------- event handling */

const RENDERERS = {
  "run.started": (d) => ({
    kind: "plan",
    body: `Started in <code>${esc(d.workspace)}</code> with ${d.tools?.length ?? 0} tools on <code>${esc(d.model)}</code>.`,
  }),
  "memory.retrieval": (d) => ({
    kind: "info",
    body: `Grounding from ${esc(d.kind)}: ${d.hits} hit${d.hits === 1 ? "" : "s"}${
      d.sources?.length ? ` — ${esc(d.sources.slice(0, 3).join(", "))}` : ""
    }`,
  }),
  "plan.created": (d) => ({ kind: "plan", body: `Planned ${d.steps.length} steps.`, detail: d.steps.map((s, i) => `${i + 1}. ${s}`).join("\n") }),
  "plan.revised": (d) => ({ kind: "warn", body: `Revised the plan (revision ${d.revision}).`, detail: d.steps.join("\n") }),
  "step.started": (d) => ({ kind: "plan", body: `<strong>Step ${d.iteration}</strong> ${esc(d.step)}` }),
  "agent.thought": (d) => ({ kind: "info", body: esc(clip(d.text, 320)) }),
  "tool.called": (d) => ({
    kind: "tool",
    body: `Calling <code>${esc(d.tool)}</code>`,
    detail: JSON.stringify(d.arguments, null, 2),
  }),
  "tool.result": (d) => ({
    kind: d.ok ? "ok" : "error",
    body: `<code>${esc(d.tool)}</code> ${d.ok ? "returned" : "failed"} in ${d.duration_ms}ms`,
    detail: d.preview,
  }),
  "agent.observation": (d) => ({
    kind: d.step_satisfied ? "ok" : "warn",
    body: esc(d.summary),
    detail: d.concerns?.length ? `concerns:\n- ${d.concerns.join("\n- ")}` : "",
  }),
  "memory.write": (d) => ({ kind: "info", body: `Remembered: ${esc(clip(d.text, 160))}` }),
  "run.warning": (d) => ({ kind: "warn", body: esc(d.message) }),
  "run.finished": (d) => ({ kind: d.status === "succeeded" ? "ok" : "error", body: `Run ${esc(d.status)}.` }),
};

function handleEvent(event) {
  updateStats(event);
  updatePlan(event);

  if (event.type === "run.finished") {
    setStatus(event.data.status);
    const summary = $("#run-summary");
    summary.textContent = event.data.summary || "(no summary)";
    summary.classList.add("is-final");
    loadRuns();
    loadRollup();
  } else if (event.type === "plan.created" || event.type === "step.started") {
    setStatus("acting");
  }

  const render = RENDERERS[event.type];
  if (!render) return;
  const { kind, body, detail } = render(event.data);
  appendEvent(event, kind, body, detail);
}

function appendEvent(event, kind, body, detail) {
  const timeline = $("#timeline");
  const item = document.createElement("li");
  item.className = "event";
  item.dataset.kind = kind;
  const time = new Date(event.ts * 1000).toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
  item.innerHTML = `
    <div class="event-head">
      <span class="event-type">${esc(event.type)}</span>
      <span class="event-time">${esc(time)}</span>
    </div>
    <div class="event-body">${body}</div>
    ${detail ? `<details class="event-detail"><summary>details</summary>${esc(detail)}</details>` : ""}
  `;
  const pinned = timeline.scrollHeight - timeline.scrollTop - timeline.clientHeight < 80;
  timeline.append(item);
  if (pinned) timeline.scrollTop = timeline.scrollHeight;
}

function updateStats(event) {
  const d = event.data;
  if (event.type === "tool.called") live.stats.tools += 1;
  if (event.type === "tool.result" && !d.ok) live.stats.failures += 1;
  if (event.type === "step.started") live.stats.iterations = d.iteration ?? live.stats.iterations;
  if (event.type === "llm.call") {
    live.stats.llm += 1;
    live.stats.input += d.input_tokens ?? 0;
    live.stats.output += d.output_tokens ?? 0;
    live.stats.usd += d.usd ?? 0;
  }
  renderRunMetrics();
}

function renderRunMetrics() {
  const s = live.stats;
  const elapsed = live.startedAt ? ((Date.now() - live.startedAt) / 1000).toFixed(1) : "0.0";
  $("#run-metrics").innerHTML = [
    ["iterations", s.iterations],
    ["tool calls", `${s.tools}${s.failures ? ` (${s.failures} failed)` : ""}`],
    ["model calls", s.llm],
    ["tokens in / out", `${s.input} / ${s.output}`],
    ["cost", money(s.usd)],
    ["elapsed", `${elapsed}s`],
  ]
    .map(([k, v]) => `<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`)
    .join("");
}

function updatePlan(event) {
  if (event.type === "plan.created" || event.type === "plan.revised") {
    live.plan = event.data.steps.map((description) => ({ description, status: "pending" }));
  } else if (event.type === "step.started") {
    const step = live.plan.find((s) => s.description === event.data.step);
    if (step) step.status = "active";
  } else if (event.type === "agent.observation" && event.data.step_satisfied) {
    const active = live.plan.find((s) => s.status === "active");
    if (active) active.status = "done";
  } else {
    return;
  }
  renderPlan();
}

const MARKS = { done: "✓", active: "▸", failed: "✕", pending: "" };

function renderPlan() {
  const list = $("#plan");
  if (!live.plan.length) {
    list.innerHTML = emptyState("Waiting for a plan", "The planner runs first; steps appear here.");
    return;
  }
  list.innerHTML = live.plan
    .map(
      (step, i) => `
      <li class="plan-step" data-status="${esc(step.status)}">
        <span class="step-mark">${MARKS[step.status] || i + 1}</span>
        <span>${esc(step.description)}</span>
      </li>`,
    )
    .join("");
  const done = live.plan.filter((s) => s.status === "done").length;
  $("#plan-progress").textContent = `${done}/${live.plan.length}`;
}

/* ---------------------------------------------------------------- memory */

async function loadMemory(query) {
  const list = $("#memory-list");
  list.innerHTML = `<li>${'<div class="skeleton"></div>'.repeat(4)}</li>`;
  try {
    const url = query ? `/api/memory?q=${encodeURIComponent(query)}` : "/api/memory";
    const { memories, stats } = await api.get(url);
    $("#memory-stats").innerHTML = [
      ["stored", stats.count],
      ["backend", stats.backend],
    ]
      .map(([label, value]) => `<div class="stat"><b>${esc(value)}</b><span>${esc(label)}</span></div>`)
      .join("");

    if (!memories.length) {
      list.innerHTML = emptyState(
        query ? "Nothing matched" : "Memory is empty",
        query
          ? "Try a broader phrase, or add the fact by hand below."
          : "The agent writes here when it learns something worth reusing. You can also seed one yourself.",
      );
      return;
    }
    list.innerHTML = memories
      .map(
        (m) => `
        <li class="memory-item">
          <span class="kind">${esc(m.kind)}</span>
          <span class="body">${esc(m.text)}</span>
          ${m.score ? `<span class="score">${m.score.toFixed(2)}</span>` : ""}
          <button class="icon-btn" data-id="${esc(m.id)}" type="button" aria-label="Forget this memory">forget</button>
        </li>`,
      )
      .join("");
    $$(".icon-btn", list).forEach((button) =>
      button.addEventListener("click", async () => {
        await api.del(`/api/memory/${button.dataset.id}`);
        toast("Forgotten", "ok");
        loadMemory(query);
      }),
    );
  } catch (err) {
    list.innerHTML = emptyState("Could not load memory", err.message);
  }
}

function initMemoryForms() {
  $("#memory-search").addEventListener("submit", (e) => {
    e.preventDefault();
    loadMemory($("#memory-q").value.trim());
  });
  $("#memory-add").addEventListener("submit", async (e) => {
    e.preventDefault();
    const text = $("#memory-text").value.trim();
    if (text.length < 8) return toast("Write a bit more - short memories are not useful later", "error");
    try {
      await api.post("/api/memory", { text, kind: $("#memory-kind").value });
      $("#memory-text").value = "";
      toast("Saved", "ok");
      loadMemory();
    } catch (err) {
      toast(err.message, "error");
    }
  });
}

/* ------------------------------------------------------------- knowledge */

async function loadKnowledge() {
  const docs = $("#doc-list");
  docs.innerHTML = `<li>${'<div class="skeleton"></div>'.repeat(5)}</li>`;
  try {
    const { documents, stats } = await api.get("/api/knowledge");
    $("#knowledge-stats").innerHTML = [
      ["documents", stats.documents],
      ["chunks", stats.chunks],
      ["backend", stats.backend],
    ]
      .map(([label, value]) => `<div class="stat"><b>${esc(value)}</b><span>${esc(label)}</span></div>`)
      .join("");
    docs.innerHTML = documents.length
      ? documents
          .map(
            (doc) => `
        <li class="doc-item">
          <span class="src">${esc(doc.source)}</span>
          <span class="count">${doc.chunks}</span>
        </li>`,
          )
          .join("")
      : emptyState("Index is empty", "Point it at a repository above and the agent can cite it while it works.");
  } catch (err) {
    docs.innerHTML = emptyState("Could not load the index", err.message);
  }
}

function initKnowledgeForms() {
  $("#index-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const button = e.submitter;
    button.classList.add("is-busy");
    button.disabled = true;
    try {
      const report = await api.post("/api/knowledge/index", { path: $("#index-path").value.trim() });
      toast(`Indexed ${report.indexed} document(s) into ${report.chunks} chunks`, "ok");
      loadKnowledge();
    } catch (err) {
      toast(err.message, "error");
    } finally {
      button.classList.remove("is-busy");
      button.disabled = false;
    }
  });

  $("#knowledge-search").addEventListener("submit", async (e) => {
    e.preventDefault();
    const query = $("#knowledge-q").value.trim();
    const target = $("#chunk-list");
    if (!query) return;
    target.innerHTML = '<div class="skeleton"></div><div class="skeleton"></div>';
    try {
      const { chunks } = await api.get(`/api/knowledge?q=${encodeURIComponent(query)}`);
      target.innerHTML = chunks.length
        ? chunks
            .map(
              (chunk) => `
          <article class="chunk">
            <header>
              <span class="cite">${esc(chunk.source)}${chunk.metadata?.lines ? `:${esc(chunk.metadata.lines)}` : ""}</span>
              <span>${chunk.score.toFixed(3)}</span>
            </header>
            <div class="score-bar"><span style="width:${Math.min(100, chunk.score * 100)}%"></span></div>
            <pre>${esc(chunk.text)}</pre>
          </article>`,
            )
            .join("")
        : `<div class="empty"><strong>Nothing passed the relevance floor</strong>That is deliberate - weak chunks are what make an agent invent APIs. Try different wording, or index more.</div>`;
    } catch (err) {
      target.innerHTML = `<div class="empty"><strong>Search failed</strong>${esc(err.message)}</div>`;
    }
  });
}

/* ----------------------------------------------------------------- tools */

async function loadTools() {
  const grid = $("#tool-grid");
  try {
    const { tools } = await api.get("/api/tools");
    grid.innerHTML = tools
      .map(
        (tool) => `
      <article class="tool-card">
        <h3>${esc(tool.name)}</h3>
        <p>${esc(tool.description)}</p>
        <div class="params">${tool.parameters.map((p) => `<code>${esc(p)}</code>`).join("")}</div>
      </article>`,
      )
      .join("");
  } catch (err) {
    grid.innerHTML = `<div class="empty"><strong>Could not load tools</strong>${esc(err.message)}</div>`;
  }
}

/* ---------------------------------------------------------------- health */

async function loadHealth() {
  try {
    const health = await api.get("/api/health");
    $("#health").innerHTML = [
      ["model", health.model],
      ["memory", health.memory_backend],
      ["index", health.index_backend],
      ["api key", health.llm_configured ? "set" : "missing"],
    ]
      .map(([k, v]) => `<div><dt>${esc(k)}</dt><dd>${esc(v)}</dd></div>`)
      .join("");
    if (!health.llm_configured) {
      toast("No ANTHROPIC_API_KEY set — runs will use the scripted demo client", "error");
    }
  } catch {
    $("#health").innerHTML = "<div><dt>api</dt><dd>offline</dd></div>";
  }
}

/* ------------------------------------------------------------------ boot */

loaders.runs = () => {
  loadRuns();
  loadRollup();
};
loaders.memory = () => loadMemory();
loaders.knowledge = () => loadKnowledge();
loaders.tools = () => loadTools();

initTheme();
initNav();
initComposer();
initMemoryForms();
initKnowledgeForms();
loadHealth();
resetLive();
setInterval(() => {
  if (live.socket && live.socket.readyState === WebSocket.OPEN) renderRunMetrics();
}, 1000);
