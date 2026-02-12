(function () {
  const boot = window.PRESCIENCE_BOOTSTRAP || {};
  const lineId = boot.lineId || "line-1";
  const initialLive = boot.initialLive || {};

  function byId(id) {
    return document.getElementById(id);
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
    if (byId("total-count")) byId("total-count").textContent = live.totals.overall;
    if (byId("unknown-count")) byId("unknown-count").textContent = live.totals.unknown;
    if (byId("throughput")) byId("throughput").textContent = Number(live.throughput.units_per_min || 0).toFixed(2);
    if (byId("alerts-recent")) byId("alerts-recent").textContent = live.alerts_recent;
    if (byId("hb-time")) byId("hb-time").textContent = live.device_health.last_heartbeat || "n/a";
    if (byId("hb-fps")) byId("hb-fps").textContent = live.device_health.fps ?? "n/a";
    if (byId("hb-bright")) byId("hb-bright").textContent = live.device_health.brightness ?? "n/a";
    if (byId("hb-blur")) byId("hb-blur").textContent = live.device_health.blur_score ?? "n/a";
    renderSkuMap(live.totals.by_sku || {});
  }

  function renderJobs(snapshot) {
    const root = byId("job-list");
    if (!root || !snapshot) return;
    const jobs = snapshot.jobs || [];
    root.innerHTML = "";
    if (!jobs.length) {
      const empty = document.createElement("div");
      empty.className = "muted";
      empty.textContent = "No workflow jobs yet.";
      root.appendChild(empty);
      return;
    }
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
})();
