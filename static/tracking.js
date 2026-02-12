(function () {
  const boot = window.PRESCIENCE_TRACKING || {};
  const modelsBySku = boot.modelsBySku || {};
  const latestVersionBySku = boot.latestVersionBySku || {};
  let activeSessionId = null;
  let statusTimer = null;

  function byId(id) {
    return document.getElementById(id);
  }

  function setTrackingStatus(text) {
    const el = byId("tracking-status");
    if (el) el.textContent = text;
  }

  function syncModelVersionOptions() {
    const sku = byId("tracking-sku").value;
    const versions = modelsBySku[sku] || [];
    const select = byId("tracking-model-version");
    select.innerHTML = "";
    if (!versions.length) {
      const option = document.createElement("option");
      option.value = "";
      option.textContent = "No model versions";
      select.appendChild(option);
      return;
    }
    for (const version of versions) {
      const option = document.createElement("option");
      option.value = String(version);
      option.textContent = `v${version}`;
      select.appendChild(option);
    }
    const latest = latestVersionBySku[sku];
    if (latest != null) {
      select.value = String(latest);
    }
  }

  async function syncDirectionFromLineConfig() {
    const lineId = byId("tracking-line-id").value || boot.lineId || "line-1";
    const res = await fetch(`/api/zone/${encodeURIComponent(lineId)}`);
    if (!res.ok) return;
    const payload = await res.json();
    const direction = payload.zone?.direction;
    if (typeof direction === "string" && direction.length) {
      byId("tracking-direction").value = direction;
    }
  }

  async function refreshStatus() {
    if (!activeSessionId) return;
    const res = await fetch(`/api/tracking/${encodeURIComponent(activeSessionId)}`);
    if (!res.ok) {
      setTrackingStatus(`status unavailable (${res.status})`);
      return;
    }
    const body = await res.json();
    const base = `status=${body.status} total=${body.total_count} fps=${Number(body.fps || 0).toFixed(2)}`;
    if (body.last_error) {
      setTrackingStatus(`${base} error=${body.last_error}`);
    } else if (body.message) {
      setTrackingStatus(
        `${base} model=${body.model_path || "n/a"} conf=${body.tracker_conf ?? "default"} direction=${body.direction || "default"} ${body.message}`
      );
    } else {
      setTrackingStatus(base);
    }
  }

  async function stopTracking() {
    if (!activeSessionId) return;
    await fetch(`/ui/tracking/${encodeURIComponent(activeSessionId)}/stop`, { method: "POST" });
    byId("tracking-stream").src = "";
    setTrackingStatus("Tracking stopped.");
    activeSessionId = null;
    if (statusTimer) {
      clearInterval(statusTimer);
      statusTimer = null;
    }
  }

  async function startTracking() {
    if (activeSessionId) {
      await stopTracking();
    }
    const payload = {
      sku_id: byId("tracking-sku").value,
      source: byId("tracking-source").value || "0",
      line_id: byId("tracking-line-id").value || boot.lineId || "line-1",
      device_id: byId("tracking-device-id").value || "device-1",
    };
    const version = byId("tracking-model-version").value;
    if (!version) {
      setTrackingStatus("No model version available for selected SKU.");
      return;
    }
    payload.model_path = `data/models/yolo/${payload.sku_id}_v${version}/best.pt`;
    const conf = Number.parseFloat(byId("tracking-conf").value);
    if (!Number.isNaN(conf)) payload.conf = conf;
    payload.direction = byId("tracking-direction").value;

    setTrackingStatus("Starting tracking session...");
    const res = await fetch("/ui/tracking/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!res.ok) {
      try {
        const body = await res.json();
        setTrackingStatus(body.detail || JSON.stringify(body));
      } catch (_err) {
        setTrackingStatus(await res.text());
      }
      return;
    }
    const out = await res.json();
    activeSessionId = out.session_id;
    const stream = byId("tracking-stream");
    stream.onerror = () => {
      setTrackingStatus("Stream has no frames yet. Check camera source, model, and zone config.");
    };
    stream.src = `${out.stream_url}?t=${Date.now()}`;

    await refreshStatus();
    setTrackingStatus(
      `Tracking started: model=${out.model_path} conf=${out.tracker_conf ?? "default"} direction=${out.direction || "default"}`
    );
    if (statusTimer) clearInterval(statusTimer);
    statusTimer = setInterval(() => {
      refreshStatus().catch((err) => setTrackingStatus(`Status failed: ${err}`));
    }, 1500);
  }

  byId("tracking-sku").addEventListener("change", syncModelVersionOptions);
  byId("tracking-line-id").addEventListener("change", () => {
    syncDirectionFromLineConfig().catch(() => {
      setTrackingStatus("Could not load direction from line config.");
    });
  });
  byId("tracking-start").addEventListener("click", () => {
    startTracking().catch((err) => setTrackingStatus(`Start failed: ${err}`));
  });
  byId("tracking-stop").addEventListener("click", () => {
    stopTracking().catch((err) => setTrackingStatus(`Stop failed: ${err}`));
  });

  syncModelVersionOptions();
  syncDirectionFromLineConfig().catch(() => {});
})();
