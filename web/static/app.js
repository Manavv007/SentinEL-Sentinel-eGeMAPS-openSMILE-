/* SentinEL web UI */

const $ = (sel) => document.querySelector(sel);

let lastCalJobId = null;
let lastAnJobId = null;
let lastResult = null;
let timelineChart = null;
let answerChart = null;
let pollTimer = null;

async function checkHealth() {
  const b = $("#healthBadge");
  try {
    const r = await fetch("/api/health");
    const d = await r.json();
    if (d.python_deps_ok === false) {
      b.textContent = "Missing whisperx — use run_web.ps1";
      b.style.background = "#450a0a";
      b.style.color = "#f87171";
      alert(d.message || "Python dependencies missing. See README.");
      return;
    }
    b.textContent = d.status === "ok" ? "API online" : "degraded";
    b.classList.add("ok");
  } catch {
    b.textContent = "API offline";
  }
}

function setTab(name) {
  document.querySelectorAll(".tab").forEach((t) => {
    t.classList.toggle("active", t.dataset.tab === name);
  });
  document.querySelectorAll(".tab-panel").forEach((p) => {
    p.classList.toggle("active", p.id === `panel-${name}`);
  });
}

document.querySelectorAll(".tab").forEach((btn) => {
  btn.addEventListener("click", () => {
    if (btn.disabled) return;
    setTab(btn.dataset.tab);
  });
});

function bindFileInput(input, labelEl, onSelect) {
  input.addEventListener("change", () => {
    const f = input.files[0];
    labelEl.textContent = f ? f.name : labelEl.dataset.default;
    onSelect(f);
  });
}

bindFileInput($("#calVideo"), $("#calVideoLabel"), (f) => {
  $("#calVideoLabel").dataset.default = "Drop calibration video or click to browse";
  $("#btnCalibrate").disabled = !f;
});

bindFileInput($("#intVideo"), $("#intVideoLabel"), (f) => {
  updateAnalyzeBtn();
});

bindFileInput($("#calJson"), $("#calJsonLabel"), () => updateAnalyzeBtn());

document.querySelectorAll('input[name="calSource"]').forEach((r) => {
  r.addEventListener("change", () => {
    const fileMode = r.value === "file";
    $("#calFileWrap").classList.toggle("hidden", !fileMode);
    $("#calJobSelect").classList.toggle("hidden", fileMode);
    updateAnalyzeBtn();
  });
});

$("#calJobSelect").addEventListener("change", updateAnalyzeBtn);

function updateAnalyzeBtn() {
  const hasVideo = $("#intVideo").files.length > 0;
  const src = document.querySelector('input[name="calSource"]:checked').value;
  let hasCal = false;
  if (src === "job") {
    hasCal = !!$("#calJobSelect").value;
  } else {
    hasCal = $("#calJson").files.length > 0;
  }
  $("#btnAnalyze").disabled = !(hasVideo && hasCal);
}

function pollJob(jobId, { box, bar, msg, onDone, fetchResult = false }) {
  if (pollTimer) clearInterval(pollTimer);

  let failStreak = 0;

  pollTimer = setInterval(async () => {
    try {
      const r = await fetch(`/api/jobs/${jobId}`);
      if (!r.ok) {
        const err = await r.text();
        throw new Error(err || `HTTP ${r.status}`);
      }
      const job = await r.json();
      failStreak = 0;

      bar.style.width = `${job.progress}%`;
      msg.textContent = job.message;
      renderLogs(job.logs || []);

      if (job.status === "done") {
        clearInterval(pollTimer);
        pollTimer = null;
        if (fetchResult) {
          msg.textContent = "Loading results…";
          const rr = await fetch(`/api/jobs/${jobId}/result`);
          if (!rr.ok) throw new Error(`Could not load results (HTTP ${rr.status})`);
          const full = await rr.json();
          onDone({ ...job, result: full });
        } else {
          onDone(job);
        }
      } else if (job.status === "error") {
        clearInterval(pollTimer);
        pollTimer = null;
        msg.textContent = job.error || "Job failed";
        bar.style.background = "var(--danger)";
      }
    } catch (e) {
      failStreak += 1;
      const hint =
        failStreak >= 3
          ? " — server may have stopped (run .\\restart_web.ps1)"
          : "";
      msg.textContent = `Poll error: ${e.message}${hint}`;
      if (failStreak >= 8) {
        clearInterval(pollTimer);
        pollTimer = null;
      }
    }
  }, 1200);
}

$("#btnCalibrate").addEventListener("click", async () => {
  const file = $("#calVideo").files[0];
  if (!file) return;

  const fd = new FormData();
  fd.append("video", file);

  $("#btnCalibrate").disabled = true;
  $("#calJobBox").classList.remove("hidden");
  $("#calProgress").style.width = "0%";
  $("#calMsg").textContent = "Uploading…";

  try {
    const r = await fetch("/api/calibrate", { method: "POST", body: fd });
    const { job_id } = await r.json();
    if (!r.ok) throw new Error((await r.json()).detail || r.statusText);

    lastCalJobId = job_id;
    $("#calJobId").textContent = job_id;

    const opt = document.createElement("option");
    opt.value = job_id;
    opt.textContent = `Calibration ${job_id.slice(0, 8)}…`;
    $("#calJobSelect").appendChild(opt);
    $("#calJobSelect").value = job_id;
    $("#calJobSelect").disabled = false;
    updateAnalyzeBtn();

    pollJob(job_id, {
      box: $("#calJobBox"),
      bar: $("#calProgress"),
      msg: $("#calMsg"),
      onDone: () => {
        $("#btnCalibrate").disabled = false;
        $("#calMsg").textContent = "Calibration complete — proceed to Analyze";
      },
    });
  } catch (e) {
    $("#calMsg").textContent = e.message;
    $("#btnCalibrate").disabled = false;
  }
});

$("#btnAnalyze").addEventListener("click", async () => {
  const file = $("#intVideo").files[0];
  if (!file) return;

  const fd = new FormData();
  fd.append("interview", file);

  const src = document.querySelector('input[name="calSource"]:checked').value;
  if (src === "job") {
    fd.append("calibration_job_id", $("#calJobSelect").value);
  } else {
    fd.append("calibration_file", $("#calJson").files[0]);
  }

  $("#btnAnalyze").disabled = true;
  $("#anJobBox").classList.remove("hidden");
  $("#anProgress").style.width = "0%";
  $("#anMsg").textContent = "Uploading…";

  try {
    const r = await fetch("/api/analyze", { method: "POST", body: fd });
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || (typeof data === "string" ? data : r.statusText));

    lastAnJobId = data.job_id;
    $("#anJobId").textContent = data.job_id;

    pollJob(data.job_id, {
      box: $("#anJobBox"),
      bar: $("#anProgress"),
      msg: $("#anMsg"),
      fetchResult: true,
      onDone: (job) => {
        $("#btnAnalyze").disabled = false;
        lastResult = job.result;
        renderResults(job.result);
        $("#resultsTab").disabled = false;
        setTab("results");
      },
    });
  } catch (e) {
    $("#anMsg").textContent = e.message;
    $("#btnAnalyze").disabled = false;
  }
});

$("#btnExport").addEventListener("click", () => {
  if (!lastAnJobId) return;
  window.open(`/api/jobs/${lastAnJobId}/export`, "_blank");
});

function renderLogs(logs) {
  const filter = ($("#logFilter").value || "").toLowerCase();
  const phase = $("#logPhase").value;
  const stream = $("#logStream");
  stream.innerHTML = "";

  logs
    .filter((e) => {
      if (phase && e.phase !== phase) return false;
      if (!filter) return true;
      const blob = JSON.stringify(e).toLowerCase();
      return blob.includes(filter);
    })
    .forEach((e) => {
      const div = document.createElement("div");
      div.className = "log-entry";
      if (e.decision === "SUSPICIOUS" || e.decision === "PROBABLE_SCRIPT_READING") {
        div.classList.add("decision-suspicious");
      }
      if (e.level === "warning") div.classList.add("level-warning");

      const metrics = e.metrics && Object.keys(e.metrics).length
        ? ` | ${JSON.stringify(e.metrics)}`
        : "";
      const decision = e.decision ? ` → ${e.decision}` : "";

      div.innerHTML = `<span class="ts">${(e.ts || "").slice(11, 19)}</span> `
        + `<span class="step">[${e.phase}/${e.step}]</span> `
        + `${escapeHtml(e.message)}${escapeHtml(decision)}${escapeHtml(metrics)}`;
      stream.appendChild(div);
    });
  stream.scrollTop = stream.scrollHeight;
}

$("#logFilter").addEventListener("input", () => {
  if (lastResult?.decision_log) renderLogs(lastResult.decision_log);
});
$("#logPhase").addEventListener("change", () => {
  if (lastResult?.decision_log) renderLogs(lastResult.decision_log);
});

function escapeHtml(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

function renderResults(data) {
  if (!data) return;
  const answers = data.answers || [];
  const alerts = answers.filter((a) => a.status === "PROBABLE_SCRIPT_READING").length;

  $("#summaryRow").innerHTML = `
    <div class="stat-card"><div class="val">${answers.length}</div><div class="lbl">Answers</div></div>
    <div class="stat-card alert"><div class="val">${alerts}</div><div class="lbl">Alerts</div></div>
    <div class="stat-card"><div class="val">${(data.elapsed_sec || 0).toFixed(0)}s</div><div class="lbl">Runtime</div></div>
    <div class="stat-card"><div class="val">${data.contrastive_engine ? "ON" : "OFF"}</div><div class="lbl">Contrastive</div></div>
  `;

  renderLogs(data.decision_log || []);
  renderTimelineChart(data);
  renderAnswerChart(answers);
  renderAnswers(answers);
}

/* ── Timeline visualization (semantic alignment with engine reasoning) ── */

const TIER_COLORS = {
  NONE: "#94a3b8",
  WEAK: "#fbbf24",
  MODERATE: "#fb923c",
  STRONG: "#ef4444",
};

const TIER_POINT_RADIUS = {
  NONE: 2,
  WEAK: 4,
  MODERATE: 5,
  STRONG: 7,
};

const TIER_INTENSITY_SCALE = {
  NONE: 0,
  WEAK: 0.42,
  MODERATE: 0.72,
  STRONG: 1.0,
};

const STATUS_OVERLAY = {
  CLEAR: "rgba(74, 222, 128, 0.09)",
  AMBIGUOUS: "rgba(251, 191, 36, 0.13)",
  PROBABLE_SCRIPT_READING: "rgba(248, 113, 113, 0.15)",
};

const STATUS_LABEL = {
  CLEAR: "CLEAR",
  AMBIGUOUS: "AMBIGUOUS",
  PROBABLE_SCRIPT_READING: "PROBABLE",
};

let timelineWindowMeta = [];
let timelineMetaByTime = new Map();

function windowStartSec(w) {
  return Number(w.start_time ?? w.start_sec ?? 0);
}

function lookupWindowAtTime(x) {
  if (x == null || Number.isNaN(x)) return null;
  const key = Number(Number(x).toFixed(4));
  if (timelineMetaByTime.has(key)) return timelineMetaByTime.get(key);
  return timelineWindowMeta.find((w) => Math.abs(windowStartSec(w) - x) < 0.08) || null;
}

function collectTimelineWindows(data) {
  const logs = data?.window_logs || [];
  if (logs.length) {
    return [...logs].sort(
      (a, b) => (a.start_time ?? a.start_sec ?? 0) - (b.start_time ?? b.start_sec ?? 0),
    );
  }

  const out = [];
  for (const answer of data?.answers || []) {
    const windows = answer.contrastive?.windows || [];
    for (const w of windows) {
      out.push({
        ...w,
        answer_id: answer.answer_id,
        start_time: w.start_sec ?? w.start_time,
        end_time: w.end_sec ?? w.end_time,
      });
    }
  }
  return out.sort(
    (a, b) => (a.start_time ?? a.start_sec ?? 0) - (b.start_time ?? b.start_sec ?? 0),
  );
}

function windowTier(w) {
  if (w.suspicion_level) return w.suspicion_level;
  return w.suspicious_flag ? "WEAK" : "NONE";
}

function windowSuspicionIntensity(w) {
  const tier = windowTier(w);
  const contrastive = Number(w.contrastive_score ?? 0);
  const evidence = w.evidence_units;
  if (evidence != null && Number(evidence) > 0) {
    return Math.min(1, Number(evidence) * 2.8);
  }
  return contrastive * (TIER_INTENSITY_SCALE[tier] ?? 0);
}

function windowEwma(w) {
  if (w.ewma_after != null) return Number(w.ewma_after);
  if (w.ewma_score != null) return Number(w.ewma_score);
  return null;
}

function buildWindowTooltipLines(w, answer) {
  const tier = windowTier(w);
  const start = w.start_time ?? w.start_sec ?? 0;
  const end = w.end_time ?? w.end_sec ?? start;
  const ewma = windowEwma(w);
  const lines = [
    `Answer ${w.answer_id ?? answer?.answer_id ?? "?"}`,
    `Time: ${start.toFixed(2)}s – ${end.toFixed(2)}s`,
    `Contrastive: ${Number(w.contrastive_score ?? 0).toFixed(3)}`,
    `Suspicion tier: ${tier}`,
    `Suspicion intensity: ${windowSuspicionIntensity(w).toFixed(3)}`,
    `Naturality: ${Number(w.naturality_score ?? 0).toFixed(3)}`,
  ];
  if (ewma != null) lines.push(`EWMA (temporal): ${ewma.toFixed(3)}`);
  if (w.peak_ewma != null) lines.push(`Peak EWMA: ${Number(w.peak_ewma).toFixed(3)}`);
  if (w.evidence_units != null) lines.push(`Evidence units: ${Number(w.evidence_units).toFixed(4)}`);
  if (w.suspicious_flag != null) {
    lines.push(`Window activation: ${w.suspicious_flag ? "above tier threshold" : "none"}`);
  }
  if (w.is_benign_window) lines.push("Benign window (recovery guard)");
  if (w.consecutive_strong != null && w.consecutive_strong > 0) {
    lines.push(`Consecutive STRONG: ${w.consecutive_strong}`);
  }
  if (answer?.status) {
    lines.push(`Final answer status: ${answer.status}`);
    const comp = answer.contrastive?.composite_score ?? answer.contrastive?.ewma_score;
    if (comp != null) lines.push(`Answer composite: ${Number(comp).toFixed(3)}`);
  }
  return lines;
}

function buildGapAwareSeries(windows, valueFn) {
  const points = [];
  for (let i = 0; i < windows.length; i += 1) {
    const w = windows[i];
    if (i > 0 && windows[i - 1].answer_id !== w.answer_id) {
      points.push({ x: null, y: null });
    }
    const x = w.start_time ?? w.start_sec ?? 0;
    const y = valueFn(w);
    points.push({ x, y: y == null ? null : y });
  }
  return points;
}

const answerOverlayPlugin = {
  id: "answerOverlay",
  beforeDatasetsDraw(chart) {
    const answers = chart.options.plugins?.answerOverlay?.answers || [];
    const { ctx, chartArea, scales } = chart;
    if (!answers.length || !chartArea) return;

    ctx.save();

    for (const answer of answers) {
      const xStart = scales.x.getPixelForValue(answer.start_sec);
      const xEnd = scales.x.getPixelForValue(answer.end_sec);
      const width = Math.max(1, xEnd - xStart);
      const fill = STATUS_OVERLAY[answer.status] || STATUS_OVERLAY.CLEAR;

      ctx.fillStyle = fill;
      ctx.fillRect(xStart, chartArea.top, width, chartArea.bottom - chartArea.top);

      ctx.fillStyle = "rgba(15, 20, 25, 0.55)";
      ctx.fillRect(xStart, chartArea.top, width, 20);

      const contr = answer.contrastive || {};
      const composite = contr.composite_score ?? contr.ewma_score;
      const label = `#${answer.answer_id} ${STATUS_LABEL[answer.status] || answer.status}`
        + (composite != null ? ` · conf ${Number(composite).toFixed(2)}` : "");

      ctx.fillStyle = "#e7ecf3";
      ctx.font = "600 10px Segoe UI, system-ui, sans-serif";
      ctx.textBaseline = "middle";
      ctx.fillText(label, xStart + 6, chartArea.top + 10);
    }

    ctx.strokeStyle = "rgba(148, 163, 184, 0.45)";
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);
    for (const answer of answers) {
      for (const t of [answer.start_sec, answer.end_sec]) {
        const x = scales.x.getPixelForValue(t);
        ctx.beginPath();
        ctx.moveTo(x, chartArea.top);
        ctx.lineTo(x, chartArea.bottom);
        ctx.stroke();
      }
    }

    ctx.restore();
  },
};

function renderTimelineLegendHint(hasAnswers) {
  const el = $("#timelineLegendHint");
  if (!el) return;
  el.innerHTML = hasAnswers
    ? `<span class="legend-chip local">Local signals</span> window-level contrastive, tier-scaled suspicion intensity, per-answer EWMA`
    + ` &nbsp;·&nbsp; <span class="legend-chip global">Global interpretation</span> shaded answer bands = final CLEAR / AMBIGUOUS / PROBABLE`
    : "Run analysis with contrastive mode to populate the timeline.";
}

function renderTimelineChart(data) {
  const ctx = $("#timelineChart").getContext("2d");
  if (timelineChart) timelineChart.destroy();

  const answers = data?.answers || [];
  const windows = collectTimelineWindows(data);
  timelineWindowMeta = windows;
  timelineMetaByTime = new Map(windows.map((w) => [Number(windowStartSec(w).toFixed(4)), w]));

  renderTimelineLegendHint(answers.length > 0);

  if (!windows.length) {
    timelineChart = new Chart(ctx, {
      type: "line",
      data: { labels: ["—"], datasets: [{ label: "No window logs", data: [0] }] },
      options: { responsive: true, maintainAspectRatio: false },
    });
    return;
  }

  const answerById = Object.fromEntries(answers.map((a) => [a.answer_id, a]));
  const tierColors = windows.map((w) => TIER_COLORS[windowTier(w)] || TIER_COLORS.NONE);
  const tierRadii = windows.map((w) => TIER_POINT_RADIUS[windowTier(w)] || 3);

  const contrastivePoints = windows.map((w) => ({
    x: w.start_time ?? w.start_sec ?? 0,
    y: Number(w.contrastive_score ?? 0),
  }));

  const naturalityPoints = windows.map((w) => ({
    x: w.start_time ?? w.start_sec ?? 0,
    y: Number(w.naturality_score ?? 0),
  }));

  const intensityPoints = buildGapAwareSeries(windows, windowSuspicionIntensity);
  const ewmaPoints = buildGapAwareSeries(windows, windowEwma);

  timelineChart = new Chart(ctx, {
    type: "line",
    data: {
      datasets: [
        {
          label: "Contrastive (script − natural)",
          data: contrastivePoints,
          borderColor: "#3d8bfd",
          backgroundColor: "rgba(61, 139, 253, 0.08)",
          borderWidth: 2,
          pointRadius: 2,
          pointHoverRadius: 5,
          tension: 0.2,
          order: 2,
        },
        {
          label: "Naturality",
          data: naturalityPoints,
          borderColor: "#4ade80",
          backgroundColor: "rgba(74, 222, 128, 0.06)",
          borderWidth: 2,
          pointRadius: 2,
          pointHoverRadius: 5,
          tension: 0.2,
          order: 2,
        },
        {
          label: "Suspicion intensity (tier-scaled)",
          data: intensityPoints,
          borderColor: "#f87171",
          backgroundColor: "rgba(248, 113, 113, 0.18)",
          borderWidth: 2.5,
          fill: "origin",
          pointBackgroundColor: tierColors,
          pointBorderColor: tierColors,
          pointRadius: tierRadii,
          pointHoverRadius: tierRadii.map((r) => r + 2),
          tension: 0.15,
          spanGaps: false,
          order: 1,
        },
        {
          label: "Temporal confidence (EWMA)",
          data: ewmaPoints,
          borderColor: "#c084fc",
          borderWidth: 2,
          borderDash: [6, 4],
          pointRadius: 0,
          pointHoverRadius: 4,
          tension: 0.25,
          spanGaps: false,
          order: 0,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      parsing: false,
      interaction: { mode: "nearest", axis: "x", intersect: false },
      plugins: {
        answerOverlay: { answers },
        legend: {
          labels: { usePointStyle: true },
        },
        tooltip: {
          callbacks: {
            title(items) {
              const w = lookupWindowAtTime(items[0]?.parsed?.x);
              if (!w) return "";
              const start = windowStartSec(w);
              const end = w.end_time ?? w.end_sec ?? start;
              return `${start.toFixed(2)}s – ${end.toFixed(2)}s`;
            },
            label(item) {
              const label = item.dataset.label || "";
              const y = item.parsed.y;
              if (y == null) return `${label}: —`;
              return `${label}: ${y.toFixed(3)}`;
            },
            afterBody(items) {
              const w = lookupWindowAtTime(items[0]?.parsed?.x);
              if (!w) return [];
              const answer = answerById[w.answer_id];
              return buildWindowTooltipLines(w, answer);
            },
          },
        },
      },
      scales: {
        x: {
          type: "linear",
          title: { display: true, text: "Time (seconds)" },
          ticks: {
            callback(v) {
              return `${Number(v).toFixed(0)}s`;
            },
          },
        },
        y: {
          min: -0.05,
          max: 1,
          title: { display: true, text: "Score / intensity" },
        },
      },
    },
    plugins: [answerOverlayPlugin],
  });

  renderTimelineTierKey();
}

function renderTimelineTierKey() {
  const el = $("#timelineTierKey");
  if (!el) return;
  el.innerHTML = [
    ["WEAK", TIER_COLORS.WEAK, "small local spikes"],
    ["MODERATE", TIER_COLORS.MODERATE, "sustained partial suspicion"],
    ["STRONG", TIER_COLORS.STRONG, "high-severity windows"],
  ]
    .map(
      ([tier, color, hint]) =>
        `<span class="tier-key-item"><span class="tier-dot" style="background:${color}"></span>${tier}<span class="tier-hint">${hint}</span></span>`,
    )
    .join("");
}

function renderAnswerChart(answers) {
  const ctx = $("#answerChart").getContext("2d");
  if (answerChart) answerChart.destroy();

  const labels = answers.map((a) => `#${a.answer_id} (${a.start_sec?.toFixed(0)}s)`);
  const channels = ["acoustic", "linguistic", "gaze", "lip"];
  const colors = ["#3d8bfd", "#a78bfa", "#fbbf24", "#f472b6"];

  answerChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels,
      datasets: channels.map((ch, i) => ({
        label: ch,
        data: answers.map((a) => {
          const sig = a.signals || {};
          const bd = a.signal_breakdown?.[ch];
          return bd?.score ?? sig[ch] ?? 0;
        }),
        backgroundColor: colors[i] + "99",
      })),
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: { y: { min: 0, max: 1 } },
    },
  });
}

function normalizeExplanation(explanation) {
  if (!explanation) return [];
  if (Array.isArray(explanation)) return explanation.map(String);
  if (typeof explanation === "string") return explanation.trim() ? [explanation] : [];
  return [String(explanation)];
}

function renderAnswers(answers) {
  const list = $("#answersList");
  list.innerHTML = "";

  answers.forEach((a) => {
    const alert = a.status === "PROBABLE_SCRIPT_READING";
    const ambiguous = a.status === "AMBIGUOUS";
    const contr = a.contrastive || {};
    const reasons = normalizeExplanation(contr.decision_explanation)
      .map((r) => `<li>${escapeHtml(r)}</li>`)
      .join("");
    const sig = a.signals || {};
    const card = document.createElement("div");
    card.className = `answer-card${alert ? " alert" : ambiguous ? " ambiguous" : ""}`;
    card.innerHTML = `
      <div class="answer-head">
        <strong>Answer ${a.answer_id}</strong>
        <span class="status-pill ${alert ? "alert" : ambiguous ? "ambiguous" : "clear"}">${a.status}</span>
        <span class="conf">${a.confidence || ""} · ${a.start_sec?.toFixed(1)}–${a.end_sec?.toFixed(1)}s</span>
      </div>
      <div class="channels">
        Acoustic ${(sig.acoustic ?? 0).toFixed(3)}
        · Linguistic ${(sig.linguistic ?? 0).toFixed(3)}
        · Gaze ${(sig.gaze ?? 0).toFixed(3)}
        · Lip ${(sig.lip ?? 0).toFixed(3)}
        ${contr.composite_score != null ? ` · Composite ${contr.composite_score.toFixed(3)}` : contr.ewma_score != null ? ` · EWMA ${contr.ewma_score.toFixed(3)}` : ""}
        ${contr.weighted_evidence != null ? ` · Evidence ${Number(contr.weighted_evidence).toFixed(2)}` : ""}
        ${contr.strong_window_count != null ? ` · STRONG ${contr.strong_window_count}` : ""}
      </div>
      <div class="transcript">"${escapeHtml(a.transcript || "")}"</div>
      ${reasons ? `<ul class="decision-reasons">${reasons}</ul>` : ""}
    `;
    list.appendChild(card);
  });
}

checkHealth();
