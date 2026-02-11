(function () {
  const boot = window.PRESCIENCE_TRACKING || {};
  let points = [];
  let frameImg = null;
  let activeSessionId = null;
  let statusTimer = null;

  const canvas = document.getElementById("zone-canvas");
  const ctx = canvas.getContext("2d");

  function byId(id) {
    return document.getElementById(id);
  }

  function setStatus(text) {
    const el = byId("zone-status");
    if (el) el.textContent = text;
  }

  function setTrackingStatus(text) {
    const el = byId("tracking-status");
    if (el) el.textContent = text;
  }

  function drawZone() {
    if (!frameImg) return;
    canvas.width = frameImg.naturalWidth;
    canvas.height = frameImg.naturalHeight;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(frameImg, 0, 0);
    if (!points.length) return;
    ctx.lineWidth = 2;
    ctx.strokeStyle = "#ff8c00";
    ctx.fillStyle = "#ff8c00";
    ctx.beginPath();
    ctx.moveTo(points[0][0], points[0][1]);
    for (let i = 1; i < points.length; i++) {
      ctx.lineTo(points[i][0], points[i][1]);
    }
    if (points.length >= 3) ctx.closePath();
    ctx.stroke();
    for (const [x, y] of points) {
      ctx.beginPath();
      ctx.arc(x, y, 4, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  function toCanvasPoint(evt) {
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    return [
      Math.round((evt.clientX - rect.left) * scaleX),
      Math.round((evt.clientY - rect.top) * scaleY),
    ];
  }

  async function loadFrame() {
    const source = byId("zone-source").value || "0";
    const img = new Image();
    img.onload = () => {
      frameImg = img;
      drawZone();
    };
    img.src = `/api/zone/frame?source=${encodeURIComponent(source)}&t=${Date.now()}`;
  }

  async function loadZone() {
    const lineId = byId("zone-line-id").value || boot.lineId || "line-1";
    const res = await fetch(`/api/zone/${encodeURIComponent(lineId)}`);
    if (!res.ok) return;
    const payload = await res.json();
    points = payload.zone?.polygon || [];
    byId("zone-direction").value = payload.zone?.direction || "left_to_right";
    drawZone();
  }

  async function saveZone() {
    const lineId = byId("zone-line-id").value || boot.lineId || "line-1";
    if (points.length < 3) {
      setStatus("Need at least 3 points.");
      return;
    }
    const payload = {
      polygon: points,
      direction: byId("zone-direction").value,
    };
    const res = await fetch(`/api/zone/${encodeURIComponent(lineId)}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!res.ok) {
      setStatus(await res.text());
      return;
    }
    setStatus(`Saved zone for ${lineId}.`);
  }

  async function startTracking() {
    const payload = {
      sku_id: byId("tracking-sku").value,
      source: byId("tracking-source").value || "0",
      line_id: byId("tracking-line-id").value || boot.lineId || "line-1",
      device_id: byId("tracking-device-id").value || "device-1",
    };
    const res = await fetch("/ui/tracking/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!res.ok) {
      setTrackingStatus(await res.text());
      return;
    }
    const out = await res.json();
    activeSessionId = out.session_id;
    byId("tracking-stream").src = out.stream_url;
    setTrackingStatus(`Tracking started: ${activeSessionId}`);
    if (statusTimer) clearInterval(statusTimer);
    statusTimer = setInterval(async () => {
      const s = await fetch(`/api/tracking/${encodeURIComponent(activeSessionId)}`);
      if (!s.ok) return;
      const body = await s.json();
      setTrackingStatus(`status=${body.status} total=${body.total_count} fps=${Number(body.fps || 0).toFixed(2)}`);
    }, 1500);
  }

  async function stopTracking() {
    if (!activeSessionId) return;
    await fetch(`/ui/tracking/${encodeURIComponent(activeSessionId)}/stop`, { method: "POST" });
    byId("tracking-stream").src = "";
    setTrackingStatus("Tracking stopped.");
    if (statusTimer) {
      clearInterval(statusTimer);
      statusTimer = null;
    }
  }

  byId("zone-load-frame").addEventListener("click", loadFrame);
  byId("zone-save").addEventListener("click", saveZone);
  byId("zone-undo").addEventListener("click", () => {
    points.pop();
    drawZone();
  });
  byId("zone-clear").addEventListener("click", () => {
    points = [];
    drawZone();
  });

  canvas.addEventListener("click", (evt) => {
    if (!frameImg) return;
    points.push(toCanvasPoint(evt));
    drawZone();
  });

  byId("tracking-start").addEventListener("click", startTracking);
  byId("tracking-stop").addEventListener("click", stopTracking);

  loadFrame().then(loadZone);
})();
