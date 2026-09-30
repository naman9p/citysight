"""Static operator dashboard for the SIH Phase 1 demo (Step 14).

Plain HTML/CSS/vanilla JS served by the existing stdlib server. The page talks
ONLY to public HTTP APIs — it
has no knowledge of ANPR/persistence internals. All data values are rendered via
textContent (never innerHTML), so no raw HTML injection is possible.

When city topology is enabled, the dashboard renders a lightweight GIS camera
map directly from the configured latitude/longitude and directed links.
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
  #map { height: 220px; position: relative; overflow: hidden; border: 1px solid #303542; border-radius: 6px; color: #6b7280; text-align: center; background: #10131a; }
  #map svg { position:absolute; inset:0; width:100%; height:100%; }
  .camera-dot { position:absolute; width:12px; height:12px; border-radius:50%; background:#67e39a; border:2px solid #0f1115; transform:translate(-50%,-50%); }
  .camera-label { position:absolute; transform:translate(-50%, 8px); font-size:10px; color:#c5ceda; white-space:nowrap; }
  form { display: flex; gap: 8px; margin-bottom: 10px; }
  form input { flex: 1; }
  .phase3 { grid-column: 1 / -1; }
  .phase3-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; }
  .phase3-card { border: 1px solid #303542; border-radius: 6px; padding: 10px; min-width: 0; }
  .phase3-card h3 { margin: 0 0 8px; font-size: 13px; color: #c5ceda; }
  .phase3-card form { flex-wrap: wrap; }
  .phase3-card pre { min-height: 130px; max-height: 360px; overflow: auto; white-space: pre-wrap; word-break: break-word; background: #10131a; border-radius: 6px; padding: 8px; font-size: 12px; }
  .trajectory-boundary { display: flex; gap: 18px; margin: 0 0 12px; color: #9aa4b2; font-size: 13px; }
  .trajectory-boundary strong { color: #e6e6e6; }
  .policy-input { width: 58px; flex: 0 0 auto; }
  @media (max-width: 900px) { main { grid-template-columns: 1fr; } .phase3-grid { grid-template-columns: 1fr; } }
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
    <div id="map"><span id="map-empty">Loading camera topology…</span></div>
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
  <section class="phase3" id="phase3-panel">
    <h2>Phase 3 — Read-only evidence inspection</h2>
    <div class="trajectory-boundary">
      <span><strong>Exact trajectory</strong> — existing accepted-plate Phase 2 result</span>
      <span><strong>Inferred trajectory hypothesis</strong> — separate evidence-backed possibility</span>
    </div>
    <div class="phase3-grid">
      <div class="phase3-card">
        <h3>Event evidence</h3>
        <form id="p3-evidence-form">
          <input id="p3-event-id" placeholder="event ID" autocomplete="off">
          <button type="submit">Load evidence</button>
        </form>
        <div class="muted">Appearance metadata is shown without raw vectors.</div>
        <pre id="p3-evidence-output">Enter an event ID.</pre>
      </div>
      <div class="phase3-card">
        <h3>Stored query history and retrieval position</h3>
        <form id="p3-query-form">
          <input id="p3-query-id" placeholder="query ID" autocomplete="off">
          <button type="submit">Load history</button>
        </form>
        <div class="muted">Stored order is retrieval history, not a relevance rank.</div>
        <pre id="p3-query-output">Enter a query ID.</pre>
      </div>
      <div class="phase3-card">
        <h3>Inferred trajectory hypothesis</h3>
        <form id="p3-hypothesis-form">
          <input id="p3-root-id" placeholder="root event ID" autocomplete="off">
          <input id="p3-hypothesis-query-ids" placeholder="query IDs, comma separated" autocomplete="off">
          <input class="policy-input" id="p3-max-hops" type="number" min="1" max="32" value="8" title="Maximum hops">
          <input class="policy-input" id="p3-max-hypotheses" type="number" min="1" max="100" value="25" title="Maximum hypotheses">
          <input class="policy-input" id="p3-max-branching" type="number" min="1" max="25" value="10" title="Maximum branching per event">
          <button type="submit">Build read-only</button>
        </form>
        <div class="muted">Physical feasibility, trusted plate relation, attribute evidence and appearance similarity remain separate.</div>
        <pre id="p3-hypothesis-output">Enter a root event and explicit query IDs.</pre>
      </div>
    </div>
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

  function loadCameraMap() {
    var map = document.getElementById("map");
    var empty = document.getElementById("map-empty");
    Promise.all([
      fetch("/v1/cameras").then(function (r) { return r.ok ? r.json() : []; }),
      fetch("/v1/links").then(function (r) { return r.ok ? r.json() : []; })
    ]).then(function (data) {
      var cameras = Array.isArray(data[0]) ? data[0] : [];
      var links = Array.isArray(data[1]) ? data[1] : [];
      if (!cameras.length) {
        if (empty) empty.textContent = "Camera topology unavailable";
        return;
      }
      map.replaceChildren();
      var lats = cameras.map(function (x) { return Number(x.latitude); });
      var lons = cameras.map(function (x) { return Number(x.longitude); });
      var minLat = Math.min.apply(null, lats), maxLat = Math.max.apply(null, lats);
      var minLon = Math.min.apply(null, lons), maxLon = Math.max.apply(null, lons);
      var latSpan = Math.max(maxLat - minLat, 0.0001);
      var lonSpan = Math.max(maxLon - minLon, 0.0001);
      function pos(cam) {
        return {
          x: 8 + ((Number(cam.longitude) - minLon) / lonSpan) * 84,
          y: 92 - ((Number(cam.latitude) - minLat) / latSpan) * 84
        };
      }
      var byId = {};
      cameras.forEach(function (cam) { byId[cam.camera_id] = cam; });
      var svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
      links.forEach(function (link) {
        var a = byId[link.from_camera_id], b = byId[link.to_camera_id];
        if (!a || !b) return;
        var p1 = pos(a), p2 = pos(b);
        var line = document.createElementNS("http://www.w3.org/2000/svg", "line");
        line.setAttribute("x1", p1.x + "%"); line.setAttribute("y1", p1.y + "%");
        line.setAttribute("x2", p2.x + "%"); line.setAttribute("y2", p2.y + "%");
        line.setAttribute("stroke", "#4b5563"); line.setAttribute("stroke-width", "2");
        svg.appendChild(line);
      });
      map.appendChild(svg);
      cameras.forEach(function (cam) {
        var p = pos(cam);
        var dot = document.createElement("div");
        dot.className = "camera-dot";
        dot.style.left = p.x + "%"; dot.style.top = p.y + "%";
        dot.title = cam.name + " · " + cam.road_name;
        map.appendChild(dot);
        var label = document.createElement("div");
        label.className = "camera-label";
        label.style.left = p.x + "%"; label.style.top = p.y + "%";
        label.textContent = cam.camera_id;
        map.appendChild(label);
      });
    }).catch(function () {
      if (empty) empty.textContent = "Camera topology unavailable";
    });
  }

  load();
  loadCameraMap();
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

  // --- Phase 3 read-only inspection ---------------------------------------
  function phase3Request(url) {
    return fetch(url).then(function (r) {
      return r.json().then(function (payload) {
        if (!r.ok) throw new Error(payload.error || "Phase 3 request failed");
        return payload;
      });
    });
  }

  function showPhase3(node, payload) {
    node.textContent = JSON.stringify(payload, null, 2);
  }

  function showPhase3Error(node, error) {
    node.textContent = error && error.message ? error.message : "Phase 3 request failed";
  }

  document.getElementById("p3-evidence-form").addEventListener("submit", function (e) {
    e.preventDefault();
    var eventId = document.getElementById("p3-event-id").value.trim();
    var output = document.getElementById("p3-evidence-output");
    if (!eventId) { output.textContent = "Event ID is required."; return; }
    phase3Request("/v1/phase3/evidence/" + encodeURIComponent(eventId))
      .then(function (data) { showPhase3(output, data); })
      .catch(function (error) { showPhase3Error(output, error); });
  });

  document.getElementById("p3-query-form").addEventListener("submit", function (e) {
    e.preventDefault();
    var queryId = document.getElementById("p3-query-id").value.trim();
    var output = document.getElementById("p3-query-output");
    if (!queryId) { output.textContent = "Query ID is required."; return; }
    var base = "/v1/phase3/queries/" + encodeURIComponent(queryId);
    Promise.all([phase3Request(base), phase3Request(base + "/results")])
      .then(function (data) {
        showPhase3(output, { query: data[0], ordered_results: data[1] });
      }).catch(function (error) { showPhase3Error(output, error); });
  });

  document.getElementById("p3-hypothesis-form").addEventListener("submit", function (e) {
    e.preventDefault();
    var rootId = document.getElementById("p3-root-id").value.trim();
    var rawIds = document.getElementById("p3-hypothesis-query-ids").value.split(",");
    var queryIds = rawIds.map(function (value) { return value.trim(); })
      .filter(function (value) { return Boolean(value); });
    var output = document.getElementById("p3-hypothesis-output");
    if (!rootId || queryIds.length === 0) {
      output.textContent = "Root event ID and at least one query ID are required.";
      return;
    }
    var parameters = new URLSearchParams();
    queryIds.forEach(function (queryId) { parameters.append("query_id", queryId); });
    parameters.set("max_hops", document.getElementById("p3-max-hops").value);
    parameters.set("max_hypotheses", document.getElementById("p3-max-hypotheses").value);
    parameters.set("max_branching_per_event", document.getElementById("p3-max-branching").value);
    var url = "/v1/phase3/hypotheses/" + encodeURIComponent(rootId) + "?" + parameters.toString();
    phase3Request(url).then(function (data) { showPhase3(output, data); })
      .catch(function (error) { showPhase3Error(output, error); });
  });
})();
</script>
</body>
</html>
"""
