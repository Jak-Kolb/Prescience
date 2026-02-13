(function () {
  const boot = window.PRESCIENCE_BOOTSTRAP || {};
  const lineId = boot.lineId || "line-1";
  const initialLive = boot.initialLive || {};
  const params = new URLSearchParams(window.location.search);

  const seedCanvas = document.getElementById("seed-review-canvas");
  const seedCtx = seedCanvas ? seedCanvas.getContext("2d") : null;

  const state = {
    seedReview: {
      sessionId: null,
      skuId: null,
      session: null,
      gate: null,
      candidates: [],
      index: 0,
      boxes: [],
      image: null,
      frameName: null,
      drawing: false,
      start: null,
      preview: null,
    },
    seedPromptedSessions: new Set(),
  };

  function byId(id) {
    return document.getElementById(id);
  }

  function setText(id, value) {
    const el = byId(id);
    if (el) el.textContent = value;
  }

  function renderSkuMap(map) {
    const skuList = byId("sku-list");
    if (!skuList) return;
    skuList.innerHTML = "";
    const entries = Object.entries(map || {});
    if (!entries.length) {
      const li = document.createElement("li");
      li.textContent = "No counts yet";
      skuList.appendChild(li);
      return;
    }
    entries.sort((a, b) => b[1] - a[1]);
    for (const [sku, value] of entries) {
      const li = document.createElement("li");
      li.textContent = `${sku}: ${value}`;
      skuList.appendChild(li);
    }
  }

  function updateLive(live) {
    if (!live) return;
    setText("total-count", live.totals.overall);
    setText("unknown-count", live.totals.unknown);
    setText("throughput", Number(live.throughput.units_per_min || 0).toFixed(2));
    setText("alerts-recent", live.alerts_recent);
    setText("hb-time", live.device_health.last_heartbeat || "n/a");
    setText("hb-fps", live.device_health.fps ?? "n/a");
    setText("hb-bright", live.device_health.brightness ?? "n/a");
    setText("hb-blur", live.device_health.blur_score ?? "n/a");
    renderSkuMap(live.totals.by_sku || {});
  }

  function showTrainingOverlay() {
    const overlay = byId("training-overlay");
    const text = byId("overlay-text");
    if (!overlay) return;
    const skuId = params.get("sku_id");
    if (text && skuId) {
      text.textContent = `Training started for ${skuId}. You can keep using the dashboard while this runs.`;
    }
    overlay.classList.remove("hidden");
    window.setTimeout(() => {
      overlay.classList.add("hidden");
      if (params.has("training_started")) {
        params.delete("training_started");
        params.delete("session_id");
        params.delete("sku_id");
        const query = params.toString();
        window.history.replaceState({}, "", query ? `/?${query}` : "/");
      }
    }, 2400);
  }

  function stageLabel(value) {
    const map = {
      extracting: "extracting",
      seed_labeling: "waiting for user approval",
      train_stage1: "training stage1",
      auto_labeling: "auto labeling",
      auto_ready_stage2: "queueing stage2",
      train_stage2: "training stage2",
      manual_required: "manual required",
      approval_labeling: "approval labeling",
      complete: "complete",
    };
    return map[value] || value || "idle";
  }

  function scoreSeedCandidate(item) {
    const boxesCount = ((item.boxes || []).length || (item.proposals || []).length || 0);
    let score = boxesCount * 100;
    if (!item.needs_review) score += 10;
    if (item.status === "pending") score += 1;
    return score;
  }

  function buildSeedQueue(session) {
    const all = session.seed_candidates || [];
    const pending = all.filter((item) => !["positive", "negative"].includes(item.status));
    const source = pending.length ? pending : all;
    return source
      .slice()
      .sort((a, b) => {
        const diff = scoreSeedCandidate(b) - scoreSeedCandidate(a);
        if (diff !== 0) return diff;
        return String(a.frame_name).localeCompare(String(b.frame_name));
      });
  }

  function currentSeedCandidate() {
    const review = state.seedReview;
    if (!review.candidates.length) return null;
    review.index = Math.max(0, Math.min(review.index, review.candidates.length - 1));
    return review.candidates[review.index];
  }

  function cloneBoxes(list) {
    return (list || []).map((b) => ({ x1: Number(b.x1), y1: Number(b.y1), x2: Number(b.x2), y2: Number(b.y2) }));
  }

  function drawSeedCanvas() {
    if (!seedCtx || !seedCanvas) return;
    const review = state.seedReview;
    if (!review.image) {
      seedCtx.clearRect(0, 0, seedCanvas.width || 1, seedCanvas.height || 1);
      return;
    }
    seedCtx.clearRect(0, 0, seedCanvas.width, seedCanvas.height);
    seedCtx.drawImage(review.image, 0, 0);
    seedCtx.lineWidth = 2;
    for (const box of review.boxes) {
      seedCtx.strokeStyle = "#00a676";
      seedCtx.strokeRect(box.x1, box.y1, box.x2 - box.x1, box.y2 - box.y1);
    }
    if (review.preview) {
      seedCtx.strokeStyle = "#ff9f1c";
      seedCtx.strokeRect(
        review.preview.x1,
        review.preview.y1,
        review.preview.x2 - review.preview.x1,
        review.preview.y2 - review.preview.y1
      );
    }
  }

  function setSeedStatus() {
    const review = state.seedReview;
    const required = review.gate?.required || 3;
    const reviewed = review.gate?.reviewed || 0;
    setText("seed-review-title", `Waiting For User Approval · ${review.skuId || ""}`);
    setText(
      "seed-review-status",
      `Review ${required} frames (${reviewed}/${required}) to continue. After this, training is autonomous for this SKU.`
    );
    const candidate = currentSeedCandidate();
    if (!candidate) {
      setText("seed-review-frame", "No seed frames available.");
      return;
    }
    const propCount = (candidate.boxes || []).length || (candidate.proposals || []).length || 0;
    const reason = candidate.reason ? ` · reason=${candidate.reason}` : "";
    setText(
      "seed-review-frame",
      `${candidate.frame_name} (${review.index + 1}/${review.candidates.length}) · proposals=${propCount}${reason}`
    );
  }

  async function fetchOnboardingSession(sessionId) {
    const res = await fetch(`/api/onboarding/${encodeURIComponent(sessionId)}`);
    if (!res.ok) return null;
    return res.json();
  }

  async function loadSeedFrame() {
    const review = state.seedReview;
    const candidate = currentSeedCandidate();
    if (!candidate || !seedCanvas) {
      review.image = null;
      drawSeedCanvas();
      return;
    }
    const frameName = String(candidate.frame_name);
    review.frameName = frameName;
    review.boxes = cloneBoxes(candidate.boxes || candidate.proposals || []);
    review.preview = null;
    setSeedStatus();

    const image = new Image();
    image.onload = () => {
      if (!seedCanvas) return;
      seedCanvas.width = image.naturalWidth;
      seedCanvas.height = image.naturalHeight;
      review.image = image;
      drawSeedCanvas();
    };
    image.src = `/api/onboarding/${encodeURIComponent(review.sessionId)}/frame/${encodeURIComponent(frameName)}?t=${Date.now()}`;
  }

  function showSeedOverlay() {
    const overlay = byId("seed-review-overlay");
    if (!overlay) return;
    overlay.classList.remove("hidden");
  }

  function closeSeedOverlay() {
    const overlay = byId("seed-review-overlay");
    if (!overlay) return;
    overlay.classList.add("hidden");
    state.seedReview = {
      sessionId: null,
      skuId: null,
      session: null,
      gate: null,
      candidates: [],
      index: 0,
      boxes: [],
      image: null,
      frameName: null,
      drawing: false,
      start: null,
      preview: null,
    };
  }

  async function openSeedReview(sessionId, skuId) {
    const payload = await fetchOnboardingSession(sessionId);
    if (!payload) {
      alert("Failed to load onboarding session.");
      return false;
    }
    const session = payload.session || {};
    if (session.state !== "seed_labeling") {
      return false;
    }
    state.seedReview.sessionId = sessionId;
    state.seedReview.skuId = skuId;
    state.seedReview.session = session;
    state.seedReview.gate = payload.seed_gate || { required: 3, reviewed: 0 };
    state.seedReview.candidates = buildSeedQueue(session);
    state.seedReview.index = 0;
    showSeedOverlay();
    await loadSeedFrame();
    return true;
  }

  async function saveSeedDecision(status) {
    const review = state.seedReview;
    const candidate = currentSeedCandidate();
    if (!candidate || !review.sessionId) return;
    if (status === "positive" && review.boxes.length === 0) {
      alert("Draw at least one box for positive approval.");
      return;
    }
    const payload = {
      frame_name: candidate.frame_name,
      status,
      stage: "seed",
      boxes: status === "positive" ? cloneBoxes(review.boxes) : [],
    };
    const res = await fetch(`/api/onboarding/${encodeURIComponent(review.sessionId)}/labels`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!res.ok) {
      const text = await res.text();
      alert(`Failed to save label: ${text}`);
      return;
    }

    const refreshed = await fetchOnboardingSession(review.sessionId);
    if (!refreshed) {
      closeSeedOverlay();
      return;
    }
    review.session = refreshed.session || review.session;
    review.gate = refreshed.seed_gate || review.gate;

    if (review.session.state !== "seed_labeling" || (review.gate.reviewed || 0) >= (review.gate.required || 3)) {
      closeSeedOverlay();
      return;
    }

    review.candidates = buildSeedQueue(review.session);
    review.index = 0;
    await loadSeedFrame();
  }

  function pointOnCanvas(evt) {
    const rect = seedCanvas.getBoundingClientRect();
    const scaleX = seedCanvas.width / rect.width;
    const scaleY = seedCanvas.height / rect.height;
    return {
      x: Math.round((evt.clientX - rect.left) * scaleX),
      y: Math.round((evt.clientY - rect.top) * scaleY),
    };
  }

  function clampBox(box) {
    const x1 = Math.max(0, Math.min(seedCanvas.width - 1, Math.min(box.x1, box.x2)));
    const y1 = Math.max(0, Math.min(seedCanvas.height - 1, Math.min(box.y1, box.y2)));
    const x2 = Math.max(0, Math.min(seedCanvas.width - 1, Math.max(box.x1, box.x2)));
    const y2 = Math.max(0, Math.min(seedCanvas.height - 1, Math.max(box.y1, box.y2)));
    return { x1, y1, x2, y2 };
  }

  function renderJobs(snapshot) {
    const root = byId("job-list");
    if (!root || !snapshot) return;
    const jobs = snapshot.jobs || [];
    const sessions = snapshot.sessions || [];
    const skus = snapshot.skus || [];

    root.innerHTML = "";
    if (!jobs.length) {
      const empty = document.createElement("div");
      empty.className = "muted";
      empty.textContent = "No workflow jobs yet.";
      root.appendChild(empty);
    } else {
      for (const job of jobs) {
        const row = document.createElement("div");
        row.className = "job-row";
        const progress = job.progress != null ? `${Number(job.progress).toFixed(0)}%` : "-";
        row.innerHTML = `
          <div><strong>${job.sku_id}</strong> · <code>${job.type}</code></div>
          <div class="muted">status=${job.status} step=${job.step || "-"} progress=${progress}</div>
          <div class="muted">${job.message || ""}</div>
        `;
        root.appendChild(row);
      }
    }

    const latestBySku = new Map();
    for (const job of jobs) {
      if (!latestBySku.has(job.sku_id)) latestBySku.set(job.sku_id, job);
    }
    const latestSessionBySku = new Map();
    for (const session of sessions) {
      if (!latestSessionBySku.has(session.sku_id)) latestSessionBySku.set(session.sku_id, session);
    }

    for (const sku of skus) {
      const mode = byId(`sku-mode-${sku.sku_id}`);
      if (mode && sku.metadata) mode.textContent = sku.metadata.labeling_mode || "bootstrap_review";
    }

    const cards = document.querySelectorAll("[id^='sku-state-']");
    let autoSeedSession = null;
    cards.forEach((node) => {
      const skuId = node.id.replace("sku-state-", "");
      const cta = byId(`manual-cta-${skuId}`);
      const session = latestSessionBySku.get(skuId);
      const job = latestBySku.get(skuId);

      const stage = session ? stageLabel(session.state) : "idle";
      const jobBits = job ? `${job.status}${job.step ? `/${job.step}` : ""}` : "idle";
      node.textContent = `Status: ${stage} · job=${jobBits}`;

      if (!cta) return;
      cta.innerHTML = "";
      if (!session) return;

      if (session.state === "manual_required") {
        cta.innerHTML = `<button class="ghost" data-session-id="${session.session_id}" data-action="manual-enter">Cloud training unavailable. Label manually?</button>`;
      } else if (session.state === "seed_labeling") {
        cta.innerHTML = `<button class="ghost" data-session-id="${session.session_id}" data-action="seed-review">waiting for user approval</button>`;
        if (!state.seedPromptedSessions.has(session.session_id) && !state.seedReview.sessionId) {
          autoSeedSession = { sessionId: session.session_id, skuId };
        }
      } else {
        return;
      }

      const button = cta.querySelector("button");
      if (!button) return;
      button.addEventListener("click", async () => {
        const sessionId = button.getAttribute("data-session-id");
        const action = button.getAttribute("data-action");
        if (!sessionId || !action) return;
        if (action === "seed-review") {
          await openSeedReview(sessionId, skuId);
          return;
        }
        const res = await fetch(`/api/onboarding/${encodeURIComponent(sessionId)}/manual/enter`, { method: "POST" });
        if (!res.ok) {
          const text = await res.text();
          alert(`Failed to enter manual mode: ${text}`);
          return;
        }
        window.location.href = `/ui/onboarding/${encodeURIComponent(sessionId)}`;
      });
    });

    if (autoSeedSession) {
      state.seedPromptedSessions.add(autoSeedSession.sessionId);
      openSeedReview(autoSeedSession.sessionId, autoSeedSession.skuId).then((opened) => {
        if (!opened) state.seedPromptedSessions.delete(autoSeedSession.sessionId);
      });
    }
  }

  if (params.get("training_started") === "1") showTrainingOverlay();

  updateLive(initialLive);

  const liveStream = new EventSource(`/stream?line_id=${encodeURIComponent(lineId)}`);
  liveStream.onmessage = (event) => {
    const payload = JSON.parse(event.data);
    updateLive(payload.live);
  };

  const jobStream = new EventSource("/stream/jobs");
  jobStream.onmessage = (event) => {
    const payload = JSON.parse(event.data);
    renderJobs(payload.snapshot);
  };

  const closeBtn = byId("seed-review-close");
  if (closeBtn) closeBtn.addEventListener("click", closeSeedOverlay);
  const posBtn = byId("seed-approve-positive");
  if (posBtn) posBtn.addEventListener("click", () => saveSeedDecision("positive"));
  const negBtn = byId("seed-approve-negative");
  if (negBtn) negBtn.addEventListener("click", () => saveSeedDecision("negative"));
  const undoBtn = byId("seed-undo");
  if (undoBtn) {
    undoBtn.addEventListener("click", () => {
      state.seedReview.boxes.pop();
      drawSeedCanvas();
    });
  }
  const clearBtn = byId("seed-clear");
  if (clearBtn) {
    clearBtn.addEventListener("click", () => {
      state.seedReview.boxes = [];
      drawSeedCanvas();
    });
  }
  const nextBtn = byId("seed-next");
  if (nextBtn) {
    nextBtn.addEventListener("click", async () => {
      if (!state.seedReview.candidates.length) return;
      state.seedReview.index = (state.seedReview.index + 1) % state.seedReview.candidates.length;
      await loadSeedFrame();
    });
  }

  if (seedCanvas && seedCtx) {
    seedCanvas.addEventListener("mousedown", (evt) => {
      if (!state.seedReview.image) return;
      state.seedReview.drawing = true;
      state.seedReview.start = pointOnCanvas(evt);
      state.seedReview.preview = {
        x1: state.seedReview.start.x,
        y1: state.seedReview.start.y,
        x2: state.seedReview.start.x,
        y2: state.seedReview.start.y,
      };
      drawSeedCanvas();
    });
    seedCanvas.addEventListener("mousemove", (evt) => {
      if (!state.seedReview.drawing || !state.seedReview.start) return;
      const p = pointOnCanvas(evt);
      state.seedReview.preview = { x1: state.seedReview.start.x, y1: state.seedReview.start.y, x2: p.x, y2: p.y };
      drawSeedCanvas();
    });
    seedCanvas.addEventListener("mouseup", (evt) => {
      if (!state.seedReview.drawing || !state.seedReview.start) return;
      state.seedReview.drawing = false;
      const p = pointOnCanvas(evt);
      const box = clampBox({
        x1: state.seedReview.start.x,
        y1: state.seedReview.start.y,
        x2: p.x,
        y2: p.y,
      });
      state.seedReview.preview = null;
      state.seedReview.start = null;
      if (Math.abs(box.x2 - box.x1) >= 2 && Math.abs(box.y2 - box.y1) >= 2) {
        state.seedReview.boxes.push(box);
      }
      drawSeedCanvas();
    });
  }
})();
