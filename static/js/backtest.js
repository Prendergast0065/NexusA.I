function startBacktestWatcher(jobId) {
  const bar   = document.getElementById("bt-bar");
  const label = document.getElementById("bt-label");
  if (!bar || !label) return;
  bar.hidden = false;
  label.hidden = false;

  async function poll() {
    try {
      const r = await fetch(`/status/${jobId}`);
      if (!r.ok) throw new Error(r.status);
      const { pct } = await r.json();
      bar.value = pct;
      label.textContent = pct + " %";
      if (pct < 100) {
        setTimeout(poll, 1000);
      }
    } catch (e) {
      console.error("progress poll failed", e);
      setTimeout(poll, 3000);
    }
  }
  poll();
}
