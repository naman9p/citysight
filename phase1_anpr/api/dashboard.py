"""Static operator dashboard for the SIH Phase 1 demo (Step 14).

Plain HTML/CSS/vanilla JS served by the existing stdlib server. The page talks
ONLY to the public HTTP API (/observations, /plates/{plate}/observations) — it
has no knowledge of ANPR/persistence internals. All data values are rendered via
textContent (never innerHTML), so no raw HTML injection is possible.

No camera coordinates exist in config, so the map area is intentionally empty
with a clear notice rather than fabricated markers.
"""

DASHBOARD_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CitySight — Operator Dashboard</title>
<style>
  :root { color-scheme: light dark; }
  body { font-family: system-ui, sans-serif; margin: 0; background: #0f1115; color: #e6e6e6; }
  header { padding: 12px 20px; background: #171a21; border-bottom: 1px solid #262a33; }
  header h1 { font-size: 18px; margin: 0; }
  main { display: grid; grid-template-columns: 1fr 320px; gap: 16px; padding: 16px 20px; }
  section { background: #171a21; border: 1px solid #262a33; border-radius: 8px; padding: 14px; }
  h2 { font-size: 14px; margin: 0 0 10px; color: #9aa4b2; text-transform: uppercase; letter-spacing: .04em; }
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th, td { text-align: left; padding: 6px 8px; border-bottom: 1px solid #262a33; }
  th { color: #9aa4b2; font-weight: 600; }
  .status { font-weight: 600; padding: 2px 8px; border-radius: 10px; font-size: 12px; }
  .accepted { background: #10331f; color: #67e39a; }
  .review { background: #33300f; color: #e3d067; }
  .abstained { background: #2a2d33; color: #9aa4b2; }
  .muted { color: #6b7280; }
  input, button { font: inherit; padding: 6px 10px; border-radius: 6px; border: 1px solid #303542; background: #10131a; color: #e6e6e6; }
  button { cursor: pointer; }
  #map { height: 220px; display: flex; align-items: center; justify-content: center; border: 1px dashed #303542; border-radius: 6px; color: #6b7280; text-align: center; }
  form { display: flex; gap: 8px; margin-bottom: 10px; }
  form input { flex: 1; }
</style>
</head>
<body>
<header><h1>CitySight — Operator Dashboard <span class="muted" id="conn"></span></h1></header>
<main>
  <section>
    <h2>Plate search (exact)</h2>
    <form id="search-form">
      <input id="q" placeholder="e.g. MH12AB1234" autocomplete="off">
      <button type="submit">Search</button>
      <button type="button" id="clear">Clear</button>
    </form>
    <h2 id="feed-title">Recent observations</h2>
    <table>
      <thead><tr>
        <th>Time</th><th>Camera</th><th>Plate</th><th>Confidence</th><th>Status</th>
      </tr></thead>
      <tbody id="rows"><tr><td colspan="5" class="muted">Loading…</td></tr></tbody>
    </table>
  </section>
  <section>
    <h2>Camera map</h2>
    <div id="map">No camera locations configured</div>
    <h2 style="margin-top:16px">Watchlist</h2>
    <form id="wl-form">
      <input id="wl-plate" placeholder="plate e.g. MH12AB1234" autocomplete="off">
      <button type="submit">Add</button>
    </form>
    <div id="wl-error" class="muted"></div>
    <table>
      <thead><tr><th>Plate</th><th>Label</th><th></th></tr></thead>
      <tbody id="wl-rows"><tr><td colspan="3" class="muted">—</td></tr></tbody>
    </table>
    <h2 style="margin-top:16px">Recent alerts</h2>
    <table>
      <thead><tr><th>Time</th><th>Plate</th><th>Camera</th><th>Conf.</th></tr></thead>
      <tbody id="alert-rows"><tr><td colspan="4" class="muted">—</td></tr></tbody>
    </table>
  </section>
</main>
<script>
(function () {
  var rows = document.getElementById("rows");
  var conn = document.getElementById("conn");
  var searchMode = null; // null => recent feed; string => plate query

  function td(text) {
    var c = document.createElement("td");
    c.textContent = text; // safe: assigns text only, no markup parsing
    return c;
  }

  function statusCell(status) {
    var c = document.createElement("td");
    var span = document.createElement("span");
    var s = String(status || "");
    span.className = "status " + (["accepted", "review", "abstained"].indexOf(s) >= 0 ? s : "muted");
    span.textContent = s || "—";
    c.appendChild(span);
    return c;
  }

  function plateText(o) {
    // Never expose a plate identity for abstained observations.
    if (o.status === "abstained") return "—";
    return o.plate_normalized || o.plate_raw || "—";
  }

  function confText(o) {
    return (typeof o.confidence === "number") ? o.confidence.toFixed(2) : "—";
  }

  function render(list) {
    rows.replaceChildren();
    if (!Array.isArray(list) || list.length === 0) {
      var tr = document.createElement("tr");
      var c = td("No observations");
      c.colSpan = 5; c.className = "muted";
      tr.appendChild(c); rows.appendChild(tr);
      return;
    }
    list.forEach(function (o) {
      var tr = document.createElement("tr");
      tr.appendChild(td(o.timestamp || "—"));
      tr.appendChild(td(o.camera_id || "—"));
      tr.appendChild(td(plateText(o)));
      tr.appendChild(td(confText(o)));
      tr.appendChild(statusCell(o.status));
      rows.appendChild(tr);
    });
  }

  function load() {
    var url = searchMode
      ? "/plates/" + encodeURIComponent(searchMode) + "/observations"
      : "/observations?limit=50";
    fetch(url).then(function (r) {
      conn.textContent = r.ok ? "" : "(api error)";
      return r.ok ? r.json() : [];
    }).then(render).catch(function () { conn.textContent = "(offline)"; });
  }

  document.getElementById("search-form").addEventListener("submit", function (e) {
    e.preventDefault();
    var q = document.getElementById("q").value.trim();
    searchMode = q || null;
    document.getElementById("feed-title").textContent =
      q ? "Search results" : "Recent observations";
    load();
  });
  document.getElementById("clear").addEventListener("click", function () {
    document.getElementById("q").value = "";
    searchMode = null;
    document.getElementById("feed-title").textContent = "Recent observations";
    load();
  });

  load();
  setInterval(load, 5000); // periodic polling only; no streaming transport

  // --- watchlist + alerts ---------------------------------------------------
  var wlRows = document.getElementById("wl-rows");
  var alertRows = document.getElementById("alert-rows");
  var wlError = document.getElementById("wl-error");

  function fillEmpty(tbody, cols, text) {
    var tr = document.createElement("tr");
    var c = td(text); c.colSpan = cols; c.className = "muted";
    tr.appendChild(c); tbody.appendChild(tr);
  }

  function loadWatchlist() {
    fetch("/watchlist").then(function (r) { return r.ok ? r.json() : []; })
      .then(function (list) {
        wlRows.replaceChildren();
        if (!Array.isArray(list) || list.length === 0) {
          fillEmpty(wlRows, 3, "Empty"); return;
        }
        list.forEach(function (w) {
          var tr = document.createElement("tr");
          tr.appendChild(td(w.normalized_plate || "—"));
          tr.appendChild(td(w.label || "—"));
          var actionCell = document.createElement("td");
          var btn = document.createElement("button");
          btn.type = "button"; btn.textContent = "Disable";
          btn.addEventListener("click", function () {
            fetch("/watchlist/" + encodeURIComponent(w.watchlist_id),
                  { method: "DELETE" }).then(loadWatchlist);
          });
          actionCell.appendChild(btn);
          tr.appendChild(actionCell);
          wlRows.appendChild(tr);
        });
      }).catch(function () {});
  }

  function loadAlerts() {
    fetch("/alerts").then(function (r) { return r.ok ? r.json() : []; })
      .then(function (list) {
        alertRows.replaceChildren();
        if (!Array.isArray(list) || list.length === 0) {
          fillEmpty(alertRows, 4, "No alerts"); return;
        }
        list.forEach(function (a) {
          var tr = document.createElement("tr");
          tr.appendChild(td(a.timestamp || "—"));
          tr.appendChild(td(a.normalized_plate || "—"));
          tr.appendChild(td(a.camera_id || "—"));
          tr.appendChild(td((typeof a.confidence === "number") ? a.confidence.toFixed(2) : "—"));
          alertRows.appendChild(tr);
        });
      }).catch(function () {});
  }

  document.getElementById("wl-form").addEventListener("submit", function (e) {
    e.preventDefault();
    wlError.textContent = "";
    var plate = document.getElementById("wl-plate").value.trim();
    fetch("/watchlist", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ plate: plate }),
    }).then(function (r) {
      if (r.ok) { document.getElementById("wl-plate").value = ""; loadWatchlist(); }
      else { return r.json().then(function (j) { wlError.textContent = j.error || "invalid plate"; }); }
    }).catch(function () { wlError.textContent = "request failed"; });
  });

  function loadSecondary() { loadWatchlist(); loadAlerts(); }
  loadSecondary();
  setInterval(loadSecondary, 5000);
})();
</script>
</body>
</html>
"""
