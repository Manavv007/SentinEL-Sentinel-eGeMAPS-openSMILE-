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
  renderTimelineChart(data.window_logs || []);
  renderAnswerChart(answers);
  renderAnswers(answers);
}

function renderTimelineChart(windowLogs) {
  const ctx = $("#timelineChart").getContext("2d");
  if (timelineChart) timelineChart.destroy();

  if (!windowLogs.length) {
    timelineChart = new Chart(ctx, {
      type: "line",
      data: { labels: ["—"], datasets: [{ label: "No window logs", data: [0] }] },
      options: { responsive: true, maintainAspectRatio: false },
    });
    return;
  }

  const labels = windowLogs.map((w) => `${(w.start_time ?? w.start_sec ?? 0).toFixed(1)}s`);
  timelineChart = new Chart(ctx, {
    type: "line",
    data: {
      labels,
      datasets: [
        {
          label: "Contrastive (script − natural)",
          data: windowLogs.map((w) => w.contrastive_score ?? 0),
          borderColor: "#3d8bfd",
          tension: 0.2,
          yAxisID: "y",
        },
        {
          label: "Naturality",
          data: windowLogs.map((w) => w.naturality_score ?? 0),
          borderColor: "#4ade80",
          tension: 0.2,
          yAxisID: "y",
        },
        {
          label: "Suspicious",
          data: windowLogs.map((w) => (w.suspicious_flag ? 1 : 0)),
          borderColor: "#f87171",
          stepped: true,
          yAxisID: "y1",
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },
      scales: {
        y: { min: -0.2, max: 1, title: { display: true, text: "Score" } },
        y1: { position: "right", min: 0, max: 1, grid: { drawOnChartArea: false } },
      },
    },
  });
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

function renderAnswers(answers) {
  const list = $("#answersList");
  list.innerHTML = "";

  answers.forEach((a) => {
    const alert = a.status === "PROBABLE_SCRIPT_READING";
    const ambiguous = a.status === "AMBIGUOUS";
    const contr = a.contrastive || {};
    const reasons = (contr.decision_explanation || [])
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
