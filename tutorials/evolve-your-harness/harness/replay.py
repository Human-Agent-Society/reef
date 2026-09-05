"""The replay page: what one run of harness evolution did, from the files it left under work/.

``python3 harness/replay.py [work] [out.html]`` reads the commit log (every
step's verdict, its mutations and the entries each release holds), the
resident process's log (mounts, polls) and every session log (turns,
stages, tool calls), and writes one self contained HTML page: the release
chain with each step's verdict and the tree diff it made, the loop graph of
each release with a scrubber that replays a session's stage path over it,
the tool call ledger of each session, and the process timeline. No server,
no network: the page carries its data inline.
"""

import json
import sys
from pathlib import Path

WORK = Path(__file__).resolve().parent.parent / "work"
SIDECAR = ".reef-harness-release"


def _lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _diff(before, after):
    """Entry ids added, updated and removed between two entry lists."""
    old = {e["id"]: e for e in before}
    new = {e["id"]: e for e in after}
    added = [e for i, e in new.items() if i not in old]
    removed = [e for i, e in old.items() if i not in new]
    updated = [e for i, e in new.items() if i in old and old[i] != e]
    return {
        "added": [{"id": e["id"], "kind": e["name"]} for e in added],
        "updated": [{"id": e["id"], "kind": e["name"]} for e in updated],
        "removed": [{"id": e["id"], "kind": e["name"]} for e in removed],
    }


def collect(work: Path) -> dict:
    """Everything the page shows, as one JSON value."""
    releases = []
    seed_entries = []
    tree_dir = work / "tree" / "native"
    if (tree_dir / "tree.json").is_file():
        seed_entries = json.loads((tree_dir / "tree.json").read_text(encoding="utf-8"))
    commits = []
    for path in sorted((work / "agent-record").glob("*.commits.jsonl")) if (work / "agent-record").is_dir() else []:
        commits.extend(_lines(path))
    commits.sort(key=lambda row: row.get("recorded_at") or 0)
    previous = None
    first_entries = None
    for row in commits:
        metrics = row.get("metrics") or {}
        state = row.get("algorithm_state") or {}
        entries = state.get("entries") or []
        if first_entries is None:
            # The seed the first step started from: the served tree before any commit.
            first_entries = _seed_before(entries, metrics)
            previous = first_entries
            releases.append(
                {
                    "release_id": row.get("artifact_ref", {}).get("parent_release_id") or "seed",
                    "step": 0,
                    "kind": "seed",
                    "entries": first_entries,
                    "diff": None,
                    "verdict": None,
                    "recorded_at": None,
                }
            )
        ref = row.get("artifact_ref") or {}
        published = bool(metrics.get("published"))
        selection = metrics.get("selection") or {}
        releases.append(
            {
                "release_id": ref.get("release_id") if published else None,
                "parent_release_id": ref.get("parent_release_id"),
                "step": metrics.get("steps"),
                "kind": "published" if published else ("skipped" if metrics.get("skipped") else "rejected"),
                "skipped": metrics.get("skipped"),
                "recorded_at": row.get("recorded_at"),
                "proposal": metrics.get("proposal"),
                "mutations": metrics.get("mutations") or ([metrics["mutation"]] if metrics.get("mutation") else []),
                "verdict": {
                    "wins": metrics.get("wins"),
                    "losses": metrics.get("losses"),
                    "ties": metrics.get("ties"),
                    "candidate_score": metrics.get("candidate_score"),
                    "current_score": metrics.get("current_score"),
                    "reason": (selection.get("reason") if isinstance(selection, dict) else None),
                    "candidate_paths": metrics.get("candidate_paths"),
                    "current_paths": metrics.get("current_paths"),
                    "proposer_seconds": metrics.get("proposer_seconds"),
                },
                "entries": entries if published else previous,
                "diff": _diff(previous, entries) if published else None,
            }
        )
        if published:
            previous = entries
    if not releases and seed_entries:
        releases.append(
            {"release_id": "seed", "step": 0, "kind": "seed", "entries": seed_entries, "diff": None, "verdict": None}
        )

    sessions = []
    sessions_dir = tree_dir / "sessions"
    if sessions_dir.is_dir():
        for directory in sorted(p for p in sessions_dir.iterdir() if p.is_dir()):
            log = directory / "session.jsonl"
            if not log.is_file():
                continue
            events = _lines(log)
            agents = {}
            for path in sorted((directory / "agents").glob("*.jsonl")) if (directory / "agents").is_dir() else []:
                agents[path.stem] = _lines(path)
            header = events[0]["data"] if events and events[0]["type"] == "session" else {}
            sessions.append(
                {
                    "session": directory.name,
                    "release_id": header.get("release_id"),
                    "model": header.get("model"),
                    "tools": header.get("tools"),
                    "start": events[0]["time"] if events else None,
                    "end": events[-1]["time"] if events else None,
                    "events": events,
                    "agents": agents,
                }
            )
    process = _lines(sessions_dir / "serve.jsonl") if (sessions_dir / "serve.jsonl").is_file() else []
    sessions.sort(key=lambda s: s["start"] or 0)
    return {"releases": releases, "sessions": sessions, "process": process, "seed_entries": seed_entries}


def _seed_before(entries, metrics):
    """The entries before the first step: undo its mutations when it published, else the state itself."""
    if not metrics.get("published"):
        return entries
    mutations = metrics.get("mutations") or ([metrics["mutation"]] if metrics.get("mutation") else [])
    before = [dict(e) for e in entries]
    for mutation in reversed(mutations):
        op, entry_id = mutation.get("op"), mutation.get("id")
        if op == "create":
            before = [e for e in before if e.get("id") != entry_id]
        elif op == "remove":
            before.append({"id": entry_id, "name": "?", "config": {}})
        # An update's previous options are not in the record; the diff marks the entry updated.
    return before


PAGE = """<title>Harness Evolution Replay</title>
<style>
:root{--bg:#f6f4ee;--ink:#1f2321;--mute:#6b6f6a;--line:#d9d5c9;--card:#fffdf8;--accent:#0a6f5c;--accent-ink:#ffffff;--warn:#a1521a;--bad:#9b2c2c;--good:#2e7d4f;--code:#efece3}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#15171a;--ink:#e8e6df;--mute:#9a9d97;--line:#2e3236;--card:#1c1f23;--accent:#4fc3a8;--accent-ink:#0c1512;--warn:#e0a05a;--bad:#e07474;--good:#7fcf9a;--code:#22262b}}
:root[data-theme="dark"]{--bg:#15171a;--ink:#e8e6df;--mute:#9a9d97;--line:#2e3236;--card:#1c1f23;--accent:#4fc3a8;--accent-ink:#0c1512;--warn:#e0a05a;--bad:#e07474;--good:#7fcf9a;--code:#22262b}
body{background:var(--bg);color:var(--ink);font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;margin:0;padding:0 0 4rem}
main{max-width:1180px;margin:0 auto;padding:1.5rem 1.25rem}
h1{font-size:1.5rem;margin:0 0 .25rem;letter-spacing:-.01em}h2{font-size:1.05rem;margin:2rem 0 .75rem;text-transform:uppercase;letter-spacing:.06em;color:var(--mute)}
.sub{color:var(--mute);margin:0 0 1rem}
.chain{display:flex;gap:.75rem;overflow-x:auto;padding-bottom:.5rem}
.rel{flex:0 0 260px;background:var(--card);border:1px solid var(--line);border-radius:6px;padding:.75rem;cursor:pointer}
.rel.sel{border-color:var(--accent);box-shadow:0 0 0 2px var(--accent) inset}
.rel .id{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.8rem;color:var(--mute)}
.rel .k{font-weight:600}.k.published{color:var(--good)}.k.rejected{color:var(--bad)}.k.skipped{color:var(--warn)}
.rel ul{margin:.4rem 0 0;padding-left:1.1rem}.rel li{font-size:.85rem}
.tag{display:inline-block;font-size:.72rem;padding:0 .35rem;border:1px solid var(--line);border-radius:3px;margin-right:.25rem;color:var(--mute)}
.two{display:grid;grid-template-columns:minmax(0,3fr) minmax(0,2fr);gap:1rem}
@media (max-width:820px){.two{grid-template-columns:1fr}}
svg{width:100%;height:auto;background:var(--card);border:1px solid var(--line);border-radius:6px}
.node rect{fill:var(--card);stroke:var(--line);stroke-width:1.5}.node text{fill:var(--ink);font-size:12px}
.node.seen rect{stroke:var(--accent)}.node.now rect{fill:var(--accent);}.node.now text{fill:var(--accent-ink)}
.edge{stroke:var(--line);stroke-width:1.5;fill:none}.edge.seen{stroke:var(--accent)}.elabel{fill:var(--mute);font-size:10px}
.panel{background:var(--card);border:1px solid var(--line);border-radius:6px;padding:.75rem;min-height:12rem}
.panel pre{white-space:pre-wrap;word-break:break-word;background:var(--code);padding:.5rem;border-radius:4px;max-height:20rem;overflow:auto;font-size:.8rem}
.ctl{display:flex;gap:.75rem;align-items:center;margin:.5rem 0;flex-wrap:wrap}
input[type=range]{flex:1;min-width:200px;accent-color:var(--accent)}
select,button{font:inherit;padding:.25rem .5rem;border:1px solid var(--line);background:var(--card);color:var(--ink);border-radius:4px}
button:focus-visible,select:focus-visible,.rel:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
table{border-collapse:collapse;width:100%;font-size:.85rem}th,td{text-align:left;padding:.3rem .5rem;border-bottom:1px solid var(--line);vertical-align:top}
th{color:var(--mute);font-weight:600;font-size:.75rem;text-transform:uppercase;letter-spacing:.05em}
td.t{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;color:var(--mute);white-space:nowrap;font-variant-numeric:tabular-nums}
tr.host td{background:color-mix(in srgb,var(--accent) 12%,transparent)}
tr.mount td{background:color-mix(in srgb,var(--good) 14%,transparent)}tr.fail td{background:color-mix(in srgb,var(--bad) 12%,transparent)}
.wrap{overflow-x:auto}details summary{cursor:pointer;color:var(--mute)}
.empty{color:var(--mute);font-style:italic}
</style>
<main>
<h1>Harness evolution replay</h1>
<p class="sub" id="sub"></p>
<h2>Releases</h2>
<div class="chain" id="chain"></div>
<h2>Loop graph, replayed from a session</h2>
<div class="ctl">
  <label>release <select id="relsel"></select></label>
  <label>session <select id="sessel"></select></label>
  <button id="play">play</button>
  <input type="range" id="scrub" min="0" max="0" value="0" aria-label="event position">
  <span id="pos" class="t"></span>
</div>
<div class="two"><div id="graph"></div><div class="panel" id="detail"><span class="empty">Move the slider to walk the session event by event.</span></div></div>
<h2>Ledger of the selected session</h2>
<div class="wrap"><table id="ledger"><thead><tr><th>t (s)</th><th>step</th><th>kind</th><th>what</th></tr></thead><tbody></tbody></table></div>
<h2>Process timeline</h2>
<div class="wrap"><table id="proc"><thead><tr><th>t (s)</th><th>event</th><th>detail</th></tr></thead><tbody></tbody></table></div>
</main>
<script id="data" type="application/json">__DATA__</script>
<script>
const D = JSON.parse(document.getElementById('data').textContent);
const t0 = Math.min(...D.sessions.map(s => s.start || Infinity), ...D.process.map(e => e.time || Infinity));
const sec = ms => ((ms - t0) / 1000).toFixed(1);
const esc = s => String(s == null ? '' : s).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const short = id => id ? String(id).slice(0, 8) : '-';
const state = { release: 0, session: 0, pos: 0, timer: null };

document.getElementById('sub').textContent =
  `${D.releases.length} release(s), ${D.sessions.length} session(s), ${D.process.length} process event(s)` +
  (D.sessions[0] && D.sessions[0].model ? ` on ${D.sessions[0].model}` : '');

function graphOf(entries) {
  const g = (entries || []).find(e => e.name === 'native_graph' && e.config && e.config.name === 'main');
  return g ? g.config : null;
}
function renderChain() {
  const el = document.getElementById('chain'); el.innerHTML = '';
  D.releases.forEach((r, i) => {
    const d = document.createElement('div'); d.className = 'rel' + (i === state.release ? ' sel' : ''); d.tabIndex = 0;
    const v = r.verdict || {};
    const source = r.proposal ? `agent proposal, session ${short(r.proposal.session)}` : (r.step ? 'method proposal' : 'the served tree before any step');
    const muts = (r.mutations || []).map(m => `<li><span class="tag">${esc(m.op)}</span>${esc(m.id)} <span class="tag">${esc((m.options||{}).name||'')}</span></li>`).join('');
    const diff = r.diff ? ['added','updated','removed'].filter(k => r.diff[k].length).map(k => `<li>${k}: ${r.diff[k].map(e => esc(e.id)).join(', ')}</li>`).join('') : '';
    d.innerHTML = `<div class="id">${r.step ? 'step ' + r.step + ' · ' : ''}${esc(short(r.release_id) || '-')}</div>` +
      `<div class="k ${r.kind}">${esc(r.kind)}${r.skipped ? ': ' + esc(r.skipped) : ''}</div>` +
      `<div>${esc(source)}</div>` +
      (v.wins != null ? `<div>W / L / T ${v.wins} / ${v.losses} / ${v.ties} · candidate ${v.candidate_score} vs current ${v.current_score}</div>` : '') +
      (muts ? `<ul>${muts}</ul>` : '') + (diff ? `<ul>${diff}</ul>` : '') +
      `<div class="id">${(r.entries||[]).length} entries</div>`;
    d.onclick = () => { state.release = i; renderChain(); renderGraph(); };
    d.onkeydown = e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); d.onclick(); } };
    el.appendChild(d);
  });
  const sel = document.getElementById('relsel'); sel.innerHTML = '';
  D.releases.forEach((r, i) => { const o = document.createElement('option'); o.value = i; o.textContent = (r.step ? 'step ' + r.step + ' ' : 'seed ') + short(r.release_id); o.selected = i === state.release; sel.appendChild(o); });
}
function layout(g) {
  // Layers by breadth first walk from the start stage; a stage keeps the first layer it is reached in.
  const stages = Object.keys(g.stages); const layer = {}; const queue = [g.start]; layer[g.start] = 0;
  while (queue.length) { const s = queue.shift(); for (const e of g.edges) if (e.from === s && !(e.to in layer)) { layer[e.to] = layer[s] + 1; queue.push(e.to); } }
  stages.forEach(s => { if (!(s in layer)) layer[s] = Math.max(0, ...Object.values(layer)) + 1; });
  const cols = {}; stages.forEach(s => { (cols[layer[s]] = cols[layer[s]] || []).push(s); });
  const W = 160, H = 46, GX = 70, GY = 24; const pos = {};
  Object.entries(cols).forEach(([l, names]) => names.forEach((s, i) => { pos[s] = { x: 20 + l * (W + GX), y: 20 + i * (H + GY) }; }));
  const width = 40 + (Object.keys(cols).length) * (W + GX), height = 40 + Math.max(...Object.values(cols).map(c => c.length)) * (H + GY);
  return { pos, W, H, width, height };
}
function renderGraph() {
  const r = D.releases[state.release]; const g = graphOf(r.entries);
  const box = document.getElementById('graph');
  if (!g) { box.innerHTML = '<div class="panel"><span class="empty">This release carries no main graph (the seed graph runs).</span></div>'; return; }
  const L = layout(g); const s = D.sessions[state.session];
  const seen = new Set(), seenEdges = new Set(); let now = null;
  if (s) {
    const walk = s.events.slice(0, state.pos + 1); let prev = null;
    for (const e of walk) {
      if (e.type === 'stage/enter') { seen.add(e.data.stage); if (prev) seenEdges.add(prev + '>' + e.data.stage); now = e.data.stage; }
      if (e.type === 'stage/exit') { prev = e.data.stage; }
    }
  }
  let svg = `<svg viewBox="0 0 ${L.width} ${L.height}" role="img" aria-label="loop graph">`;
  svg += `<defs><marker id="arr" viewBox="0 0 10 10" refX="10" refY="5" markerWidth="8" markerHeight="8" orient="auto"><path d="M0 0L10 5L0 10z" fill="currentColor"/></marker></defs>`;
  for (const e of g.edges) {
    const a = L.pos[e.from], b = L.pos[e.to]; if (!a || !b) continue;
    const x1 = a.x + L.W, y1 = a.y + L.H / 2, x2 = b.x, y2 = b.y + L.H / 2;
    const back = b.x <= a.x; const c = back ? `M${a.x + L.W/2} ${a.y + L.H} C ${a.x + L.W/2} ${a.y + L.H + 40}, ${b.x + L.W/2} ${b.y + L.H + 40}, ${b.x + L.W/2} ${b.y + L.H}` : `M${x1} ${y1} C ${x1 + 30} ${y1}, ${x2 - 30} ${y2}, ${x2} ${y2}`;
    const cls = 'edge' + (seenEdges.has(e.from + '>' + e.to) ? ' seen' : '');
    svg += `<path class="${cls}" d="${c}" marker-end="url(#arr)" style="color:var(--line)"/>`;
    const mx = back ? (a.x + b.x + L.W) / 2 : (x1 + x2) / 2, my = back ? a.y + L.H + 34 : (y1 + y2) / 2 - 4;
    svg += `<text class="elabel" x="${mx}" y="${my}" text-anchor="middle">${esc(e.when)}</text>`;
  }
  for (const [name, p] of Object.entries(L.pos)) {
    const st = g.stages[name] || {}; const cls = 'node' + (seen.has(name) ? ' seen' : '') + (now === name ? ' now' : '');
    svg += `<g class="${cls}"><rect x="${p.x}" y="${p.y}" width="${L.W}" height="${L.H}" rx="6"/><text x="${p.x + 10}" y="${p.y + 19}" font-weight="600">${esc(name)}</text><text x="${p.x + 10}" y="${p.y + 36}">${esc(st.kind || '')}${st.check ? ' · ' + esc(st.check) : ''}${st.agent ? ' · ' + esc(st.agent) : ''}</text></g>`;
  }
  svg += '</svg>'; box.innerHTML = svg;
}
function describe(e) {
  const d = e.data || {};
  switch (e.type) {
    case 'session': return `session ${short(d.session)} on release ${short(d.release_id)}, tools ${(d.tools || []).join(', ')}`;
    case 'turn/start': return `turn ${d.turn}: ${d.prompt}`;
    case 'step/start': return `step ${d.step} of turn ${d.turn} starts`;
    case 'step/end': return `step ${d.step} of turn ${d.turn} ends`;
    case 'stage/enter': return `enter ${d.stage} (${d.kind})`;
    case 'stage/exit': return `exit ${d.stage} with ${d.outcome} to ${d.to}` + (d.timeout ? ` · timeout ${JSON.stringify(d.timeout)}` : '') + (d.pattern ? ` · pattern ${d.pattern}` : '');
    case 'request/header': return `request header: ${(d.tools || []).length} tool(s), system prompt ${String(d.system || '').length} chars`;
    case 'assistant/message': return (d.tool_calls && d.tool_calls.length) ? `assistant calls ${d.tool_calls.map(c => c.function && c.function.name).join(', ')}` : `assistant: ${d.content}`;
    case 'tool/call': return `${d.name}(${typeof d.arguments === 'string' ? d.arguments : JSON.stringify(d.arguments)})`;
    case 'tool/result': return `${d.name} -> ${d.is_error ? 'error: ' : ''}${d.content}`;
    case 'user/message': return `user message from ${d.source ? d.source.kind : '?'}: ${d.content}`;
    case 'harness/mount': return `mount ${short(d.release_id)} (${d.source}, ${d.entries} entries)`;
    case 'harness/mount-failed': return `mount failed ${short(d.release_id)}: ${d.error}`;
    case 'release/poll-failed': return `poll failed, retry in ${d.retry_in_s}s: ${d.error}`;
    case 'turn/end': return `turn ${d.turn} ended: ${JSON.stringify(d.reason)}`;
    default: return JSON.stringify(d);
  }
}
function renderDetail() {
  const s = D.sessions[state.session]; const el = document.getElementById('detail');
  if (!s || !s.events.length) { el.innerHTML = '<span class="empty">No session.</span>'; return; }
  const e = s.events[state.pos];
  el.innerHTML = `<div><span class="tag">${esc(e.type)}</span> <span class="t">t+${sec(e.time)}s</span> step ${esc((e.data||{}).step ?? '')}</div><pre>${esc(describe(e))}</pre>`;
  document.getElementById('pos').textContent = `${state.pos + 1} / ${s.events.length}`;
  const rows = document.querySelectorAll('#ledger tbody tr'); rows.forEach((tr, i) => tr.style.outline = i === state.pos ? '2px solid var(--accent)' : '');
}
function renderLedger() {
  const s = D.sessions[state.session]; const tb = document.querySelector('#ledger tbody'); tb.innerHTML = '';
  if (!s) return;
  s.events.forEach((e, i) => {
    const d = e.data || {}; const tr = document.createElement('tr');
    if (e.type === 'tool/call' && /^harness_/.test(d.name || '')) tr.className = 'host';
    if (e.type === 'harness/mount') tr.className = 'mount';
    if (e.type === 'tool/result' && d.is_error) tr.className = 'fail';
    tr.innerHTML = `<td class="t">${sec(e.time)}</td><td class="t">${esc(d.step ?? '')}</td><td>${esc(e.type)}</td><td>${esc(describe(e)).slice(0, 400)}</td>`;
    tr.onclick = () => { state.pos = i; sync(); }; tb.appendChild(tr);
  });
  const sel = document.getElementById('sessel'); sel.innerHTML = '';
  D.sessions.forEach((x, i) => { const o = document.createElement('option'); o.value = i; o.textContent = `${short(x.session)} on ${short(x.release_id)} (${x.events.length} events)`; o.selected = i === state.session; sel.appendChild(o); });
  const scrub = document.getElementById('scrub'); scrub.max = Math.max(0, s.events.length - 1); scrub.value = state.pos;
}
function renderProcess() {
  const tb = document.querySelector('#proc tbody'); tb.innerHTML = '';
  const rows = [...D.process.map(e => ({ t: e.time, e }))];
  D.sessions.forEach(s => { rows.push({ t: s.start, e: { type: 'turn', data: { session: s.session, release_id: s.release_id } } }); });
  rows.sort((a, b) => a.t - b.t);
  rows.forEach(({ t, e }) => {
    const tr = document.createElement('tr'); const d = e.data || {};
    if (e.type === 'harness/mount') tr.className = 'mount'; if (/failed/.test(e.type)) tr.className = 'fail';
    const what = e.type === 'turn' ? `turn in session ${short(d.session)} on release ${short(d.release_id)}` : describe(e);
    tr.innerHTML = `<td class="t">${sec(t)}</td><td>${esc(e.type)}</td><td>${esc(what)}</td>`; tb.appendChild(tr);
  });
  if (!rows.length) tb.innerHTML = '<tr><td colspan="3" class="empty">No process log: the episode form writes none.</td></tr>';
}
function sync() { document.getElementById('scrub').value = state.pos; renderGraph(); renderDetail(); }
document.getElementById('relsel').onchange = e => { state.release = +e.target.value; renderChain(); renderGraph(); };
document.getElementById('sessel').onchange = e => { state.session = +e.target.value; state.pos = 0; renderLedger(); sync(); };
document.getElementById('scrub').oninput = e => { state.pos = +e.target.value; renderGraph(); renderDetail(); };
document.getElementById('play').onclick = () => {
  const s = D.sessions[state.session]; if (!s) return;
  if (state.timer) { clearInterval(state.timer); state.timer = null; document.getElementById('play').textContent = 'play'; return; }
  document.getElementById('play').textContent = 'pause';
  state.timer = setInterval(() => { if (state.pos >= s.events.length - 1) { clearInterval(state.timer); state.timer = null; document.getElementById('play').textContent = 'play'; return; } state.pos += 1; sync(); }, 350);
};
renderChain(); renderLedger(); renderProcess(); renderGraph(); renderDetail();
</script>
"""


def render(data: dict) -> str:
    payload = json.dumps(data, ensure_ascii=True).replace("</", "<\\/")
    return PAGE.replace("__DATA__", payload)


def main(argv=None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    work = Path(args[0]) if args else WORK
    out = Path(args[1]) if len(args) > 1 else work / "replay.html"
    data = collect(work)
    out.write_text(render(data), encoding="utf-8")
    print(
        f"{out}: {len(data['releases'])} release(s), {len(data['sessions'])} session(s), "
        f"{len(data['process'])} process event(s)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
