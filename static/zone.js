(function () {
  const boot = window.PRESCIENCE_ZONE || {};
  let points = [];
  let frameImg = null;

  const canvas = document.getElementById("zone-canvas");
  const ctx = canvas.getContext("2d");

  function byId(id) {
    return document.getElementById(id);
  }

  function setStatus(text) {
    const el = byId("zone-status");
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
    setStatus("Loading frame...");
    const url = `/api/zone/frame?source=${encodeURIComponent(source)}&t=${Date.now()}`;
    const img = new Image();
    img.onload = () => {
      frameImg = img;
      drawZone();
      setStatus(`Frame loaded (${img.naturalWidth}x${img.naturalHeight}).`);
    };
    img.onerror = async () => {
      let detail = "Could not decode frame image.";
      try {
        const res = await fetch(url);
        if (!res.ok) {
          const body = await res.json();
          detail = body.detail || detail;
        } else {
          detail = "Frame endpoint returned bytes, but browser could not decode image.";
        }
      } catch (_err) {
        detail = "Could not load frame from camera source.";
      }
      setStatus(detail);
    };
    img.src = url;
  }

  async function loadZone() {
    const lineId = byId("zone-line-id").value || boot.lineId || "line-1";
    const res = await fetch(`/api/zone/${encodeURIComponent(lineId)}`);
    if (!res.ok) {
      setStatus(`Could not load zone config for ${lineId}.`);
      return;
    }
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

  byId("zone-load-frame").addEventListener("click", () => {
    loadFrame().catch((err) => setStatus(`Load failed: ${err}`));
  });
  byId("zone-save").addEventListener("click", () => {
    saveZone().catch((err) => setStatus(`Save failed: ${err}`));
  });
  byId("zone-undo").addEventListener("click", () => {
    points.pop();
    drawZone();
  });
  byId("zone-clear").addEventListener("click", () => {
    points = [];
    drawZone();
  });
  byId("zone-line-id").addEventListener("change", () => {
    loadZone().catch(() => {
      setStatus("Could not refresh zone.");
    });
  });

  canvas.addEventListener("click", (evt) => {
    if (!frameImg) return;
    points.push(toCanvasPoint(evt));
    drawZone();
  });

  loadFrame()
    .then(loadZone)
    .catch((err) => setStatus(`Init failed: ${err}`));
})();
