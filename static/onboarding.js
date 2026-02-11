(function () {
  const boot = window.PRESCIENCE_ONBOARDING;
  if (!boot) return;

  const sessionId = boot.sessionId;
  let session = boot.session;
  let summary = boot.summary;
  let candidates = [];
  let stage = "seed";
  let currentIndex = 0;
  let img = null;
  let boxes = [];
  let drawing = false;
  let startPt = null;
  let previewBox = null;

  const canvas = document.getElementById("label-canvas");
  const ctx = canvas.getContext("2d");

  function byId(id) {
    return document.getElementById(id);
  }

  function setText(id, value) {
    const el = byId(id);
    if (el) el.textContent = value;
  }

  function setProgress(value) {
    const bar = byId("training-progress-bar");
    if (!bar) return;
    const pct = Math.max(0, Math.min(100, Number(value || 0)));
    bar.style.width = `${pct}%`;
  }

  function getActiveCandidates() {
    if (session.state === "approval_labeling" || session.state === "train_stage2" || session.state === "complete") {
      stage = "approval";
      return session.approval_candidates || [];
    }
    stage = "seed";
    return session.seed_candidates || [];
  }

  function imageUrl(frameName) {
    return `/api/onboarding/${encodeURIComponent(sessionId)}/frame/${encodeURIComponent(frameName)}`;
  }

  function currentCandidate() {
    if (!candidates.length) return null;
    if (currentIndex >= candidates.length) currentIndex = candidates.length - 1;
    if (currentIndex < 0) currentIndex = 0;
    return candidates[currentIndex];
  }

  function draw() {
    if (!img) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, 0, 0);
    ctx.lineWidth = 2;
    for (const b of boxes) {
      ctx.strokeStyle = "#00a676";
      ctx.strokeRect(b.x1, b.y1, b.x2 - b.x1, b.y2 - b.y1);
    }
    if (previewBox) {
      ctx.strokeStyle = "#ff9f1c";
      ctx.strokeRect(previewBox.x1, previewBox.y1, previewBox.x2 - previewBox.x1, previewBox.y2 - previewBox.y1);
    }
  }

  function clampBox(box) {
    const x1 = Math.max(0, Math.min(canvas.width - 1, Math.min(box.x1, box.x2)));
    const y1 = Math.max(0, Math.min(canvas.height - 1, Math.min(box.y1, box.y2)));
    const x2 = Math.max(0, Math.min(canvas.width - 1, Math.max(box.x1, box.x2)));
    const y2 = Math.max(0, Math.min(canvas.height - 1, Math.max(box.y1, box.y2)));
    return { x1, y1, x2, y2 };
  }

  function toCanvasPoint(evt) {
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    return {
      x: Math.round((evt.clientX - rect.left) * scaleX),
      y: Math.round((evt.clientY - rect.top) * scaleY),
    };
  }

  function loadCurrentFrame() {
    const candidate = currentCandidate();
    if (!candidate) {
      setText("frame-label", "No candidates available for this stage.");
      img = null;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      return;
    }
    setText(
      "frame-label",
      `${candidate.frame_name} (${currentIndex + 1}/${candidates.length}) stage=${stage} status=${candidate.status || "pending"}`
    );
    boxes = (candidate.boxes || []).map((b) => ({ x1: b.x1, y1: b.y1, x2: b.x2, y2: b.y2 }));
    previewBox = null;
    const image = new Image();
    image.onload = () => {
      canvas.width = image.naturalWidth;
      canvas.height = image.naturalHeight;
      img = image;
      draw();
    };
    image.src = imageUrl(candidate.frame_name) + `?t=${Date.now()}`;
  }

  async function fetchSession() {
    const res = await fetch(`/api/onboarding/${encodeURIComponent(sessionId)}`);
    if (!res.ok) return;
    const payload = await res.json();
    session = payload.session;
    summary = payload.summary;
    candidates = getActiveCandidates();
    setText("session-state", summary.state);
    setText("seed-progress", `${summary.seed.labeled}/${summary.seed.total}`);
    setText("approval-progress", `${summary.approval.labeled}/${summary.approval.total}`);
    byId("seed-complete").disabled = !(session.state === "seed_labeling");
    byId("approval-complete").disabled = !(session.state === "approval_labeling");
    if (!currentCandidate()) currentIndex = 0;
    loadCurrentFrame();
  }

  async function saveLabel(status) {
    const candidate = currentCandidate();
    if (!candidate) return;
    const payload = {
      frame_name: candidate.frame_name,
      status,
      stage,
      boxes: boxes.map((b) => ({ x1: b.x1, y1: b.y1, x2: b.x2, y2: b.y2 })),
    };
    if (status === "positive" && !payload.boxes.length) {
      alert("Draw at least one box for positive label.");
      return;
    }
    const res = await fetch(`/api/onboarding/${encodeURIComponent(sessionId)}/labels`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!res.ok) {
      const text = await res.text();
      alert(`Failed to save label: ${text}`);
      return;
    }
    candidate.status = status;
    candidate.boxes = payload.boxes;
    if (currentIndex < candidates.length - 1) currentIndex += 1;
    await fetchSession();
  }

  function renderJobs(snapshot) {
    if (!snapshot) return;
    const root = byId("job-feed");
    if (!root) return;
    root.innerHTML = "";
    const jobs = snapshot.jobs || [];
    const latestJob = jobs.find((j) => j.job_id === session.latest_job_id) || jobs[0];
    if (latestJob) {
      setText("training-status", `${latestJob.status} · ${latestJob.step || "-"} · ${latestJob.message || ""}`);
      setProgress(latestJob.progress || 0);
    }
    for (const job of jobs.slice(0, 8)) {
      const row = document.createElement("div");
      row.className = "job-row";
      row.innerHTML = `
        <div><code>${job.type}</code> · ${job.status}</div>
        <div class="muted">${job.step || "-"} · ${job.message || ""}</div>
      `;
      root.appendChild(row);
    }
  }

  byId("save-positive").addEventListener("click", () => saveLabel("positive"));
  byId("save-negative").addEventListener("click", () => saveLabel("negative"));
  byId("save-skipped").addEventListener("click", () => saveLabel("skipped"));
  byId("undo-box").addEventListener("click", () => {
    boxes.pop();
    draw();
  });
  byId("clear-boxes").addEventListener("click", () => {
    boxes = [];
    draw();
  });
  byId("prev-frame").addEventListener("click", () => {
    if (currentIndex > 0) currentIndex -= 1;
    loadCurrentFrame();
  });
  byId("next-frame").addEventListener("click", () => {
    if (currentIndex < candidates.length - 1) currentIndex += 1;
    loadCurrentFrame();
  });

  byId("seed-complete").addEventListener("click", async () => {
    const res = await fetch(`/api/onboarding/${encodeURIComponent(sessionId)}/seed/complete`, { method: "POST" });
    if (!res.ok) {
      alert(await res.text());
      return;
    }
    await fetchSession();
  });

  byId("approval-complete").addEventListener("click", async () => {
    const res = await fetch(`/api/onboarding/${encodeURIComponent(sessionId)}/approvals/complete`, { method: "POST" });
    if (!res.ok) {
      alert(await res.text());
      return;
    }
    await fetchSession();
  });

  canvas.addEventListener("mousedown", (evt) => {
    if (!img) return;
    drawing = true;
    const p = toCanvasPoint(evt);
    startPt = p;
    previewBox = { x1: p.x, y1: p.y, x2: p.x, y2: p.y };
    draw();
  });

  canvas.addEventListener("mousemove", (evt) => {
    if (!drawing || !startPt) return;
    const p = toCanvasPoint(evt);
    previewBox = { x1: startPt.x, y1: startPt.y, x2: p.x, y2: p.y };
    draw();
  });

  canvas.addEventListener("mouseup", (evt) => {
    if (!drawing || !startPt) return;
    drawing = false;
    const p = toCanvasPoint(evt);
    const box = clampBox({ x1: startPt.x, y1: startPt.y, x2: p.x, y2: p.y });
    previewBox = null;
    if (Math.abs(box.x2 - box.x1) >= 2 && Math.abs(box.y2 - box.y1) >= 2) boxes.push(box);
    startPt = null;
    draw();
  });

  window.addEventListener("keydown", (evt) => {
    if (evt.key === "u") {
      boxes.pop();
      draw();
    } else if (evt.key === "c") {
      boxes = [];
      draw();
    }
  });

  const jobsStream = new EventSource(`/stream/jobs?sku_id=${encodeURIComponent(session.sku_id)}`);
  jobsStream.onmessage = async (event) => {
    const payload = JSON.parse(event.data);
    renderJobs(payload.snapshot);
    await fetchSession();
  };

  fetchSession();
})();
