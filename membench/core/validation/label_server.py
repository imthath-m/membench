"""Local browser tool for one human annotator to label judge-validation items.

`membench judge-label` serves a single page on localhost. Every label is
written immediately (atomically) to the annotator's JSONL in the format
`merge_annotations` reads: `{"item_id": str, "label": bool, "note"?: str}`.

Blinding: only `id`, `metric`, `rubric`, `evidence` and `prompt` ever reach
the browser; the judge verdict, model, prompt version, backend, pack and
length information stay on the server.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from membench.core.validation.labels import LabelsError, read_jsonl

VISIBLE_FIELDS = ("id", "metric", "rubric", "evidence", "prompt")

# Short reminders, condensed from docs/labelling-guidelines.md.
RULES = {
    "semantic_recall": "true if the evidence lets a reader answer as the statement says. If an "
    "older value is also present, true only when the evidence makes clear which value is "
    "current (later timestamp, 'moved to', 'now').",
    "contradiction_rate": "true = the contradiction IS present: two or more conflicting values "
    "both presented as current, with nothing saying which wins. A clearly marked update is "
    "NOT a contradiction: false.",
    "qa_accuracy": "true if the answer (evidence) gives the same answer as the reference: same "
    "entity/value/date/number; extra correct detail is fine; a hedge naming a wrong answer "
    "too is false; 'I don't know' is false unless the reference says it is unanswerable.",
    "qa_accuracy:longmemeval": "Same as qa_accuracy; follow the official prompt's "
    "question-type rule when stricter (e.g. off-by-one accepted for temporal questions; "
    "abstention items must say the question is unanswerable).",
}
GENERAL_RULE = ("Does the EVIDENCE clearly support the STATEMENT? Judge only the evidence "
                "shown. If you need to guess, label false. Broken item: false + note 'broken'.")


def order_items(items: list[dict], seed: int, annotator: str) -> list[dict]:
    """Deterministic shuffle keyed by seed + annotator (interleaves backends)."""
    key = int(hashlib.sha256(f"{seed}:{annotator}".encode()).hexdigest()[:16], 16)
    out = sorted(items, key=lambda item: item["id"])
    random.Random(key).shuffle(out)
    return out


class LabelStore:
    """One annotator's labels over a (filtered, ordered) sample, persisted to `out`."""

    def __init__(self, sample: list[dict], annotator: str, out: str | Path,
                 metrics: list[str] | None = None, seed: int = 0):
        if metrics:
            sample = [item for item in sample if item.get("metric") in metrics]
        if not sample:
            raise LabelsError("no items to label (empty sample or no item matches --metric)")
        self.annotator = annotator
        self.out = Path(out)
        self.items = order_items(sample, seed, annotator)
        self._by_id = {item["id"]: item for item in self.items}
        self._all_ids = {item["id"] for item in sample}
        self.labels: dict[str, dict] = {}
        self._lock = threading.Lock()
        self.load()

    def load(self) -> None:
        if not self.out.exists():
            return
        for n, row in enumerate(read_jsonl(self.out), 1):
            where = f"{self.out}:{n}"
            if row.get("item_id") not in self._all_ids:
                raise LabelsError(f"{where}: item_id {row.get('item_id')!r} is not in the sample "
                                  "(or excluded by --metric)")
            if not isinstance(row.get("label"), bool):
                raise LabelsError(f"{where}: `label` must be true or false")
            if row.get("note") is not None and not isinstance(row["note"], str):
                raise LabelsError(f"{where}: `note` must be a string")
            self.labels[row["item_id"]] = row

    def set(self, item_id: str, label: bool | None, note: str | None = None) -> None:
        if item_id not in self._by_id:
            raise LabelsError(f"unknown item_id {item_id!r}")
        if label is not None and not isinstance(label, bool):
            raise LabelsError("`label` must be true, false or null")
        if note is not None and not isinstance(note, str):
            raise LabelsError("`note` must be a string")
        with self._lock:
            if label is None:
                self.labels.pop(item_id, None)
            else:
                row = {"item_id": item_id, "label": label}
                if note and note.strip():
                    row["note"] = note.strip()
                self.labels[item_id] = row
            self.save()

    def clear(self, item_id: str) -> None:
        self.set(item_id, None)

    def rows(self) -> list[dict]:
        return [self.labels[i["id"]] for i in self.items if i["id"] in self.labels]

    def export(self) -> str:
        return "".join(json.dumps(r, sort_keys=True) + "\n" for r in self.rows())

    def save(self) -> None:
        self.out.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.out.parent, prefix=f".{self.out.name}.")
        try:
            with os.fdopen(fd, "w") as fh:
                fh.write(self.export())
            os.replace(tmp, self.out)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

    def first_unlabelled(self) -> int:
        for n, item in enumerate(self.items):
            if item["id"] not in self.labels:
                return n
        return 0

    def state(self) -> dict:
        return {
            "annotator": self.annotator,
            "items": [{k: item[k] for k in VISIBLE_FIELDS if k in item} for item in self.items],
            "labels": self.labels,
            "rules": {**RULES, "_general": GENERAL_RULE},
            "start": self.first_unlabelled(),
        }


def make_server(store: LabelStore, host: str = "127.0.0.1",
                port: int = 8765) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args) -> None:  # keep the terminal quiet
            pass

        def _send(self, code: int, body: str, ctype: str, extra: dict | None = None) -> None:
            data = body.encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(data)

        def _json(self, code: int, obj) -> None:
            self._send(code, json.dumps(obj), "application/json")

        def do_GET(self) -> None:
            path = self.path.split("?", 1)[0]
            if path == "/":
                self._send(200, PAGE, "text/html; charset=utf-8")
            elif path == "/api/state":
                self._json(200, store.state())
            elif path == "/export.jsonl":
                name = f"{store.annotator}.jsonl"
                self._send(200, store.export(), "application/x-ndjson",
                           {"Content-Disposition": f'attachment; filename="{name}"'})
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self) -> None:
            if self.path.split("?", 1)[0] != "/api/label":
                self._json(404, {"error": "not found"})
                return
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)))
                if not isinstance(body, dict):
                    raise LabelsError("body must be a JSON object")
                store.set(body.get("item_id"), body.get("label"), body.get("note"))
            except (ValueError, TypeError) as exc:
                self._json(400, {"error": str(exc)})
                return
            self._json(200, {"ok": True, "labels": store.labels})

    return ThreadingHTTPServer((host, port), Handler)


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>membench judge-label</title>
<style>
:root{--bg:#fafafa;--fg:#1d1d1f;--mut:#666;--card:#fff;--bd:#ddd;--yes:#1a7f37;--no:#cf222e}
@media (prefers-color-scheme: dark){:root{--bg:#161618;--fg:#e8e8ea;--mut:#999;--card:#202024;
--bd:#3a3a40;--yes:#3fb950;--no:#f85149}}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.45 system-ui,sans-serif}
main{max-width:1000px;margin:0 auto;padding:16px}
header{display:flex;flex-wrap:wrap;gap:12px;align-items:baseline;justify-content:space-between}
.mut{color:var(--mut)} .card{background:var(--card);border:1px solid var(--bd);
border-radius:8px;padding:12px 14px;margin:10px 0}
h2{font-size:13px;text-transform:uppercase;letter-spacing:.05em;margin:0 0 6px;color:var(--mut)}
pre{white-space:pre-wrap;word-break:break-word;max-height:45vh;overflow:auto;margin:0;
font:13px/1.4 ui-monospace,monospace}
.stmt{font-size:17px} .rule{font-size:13px}
.status{font-weight:600} .t{color:var(--yes)} .f{color:var(--no)}
button{font:inherit;padding:6px 12px;border-radius:6px;border:1px solid var(--bd);
background:var(--card);color:var(--fg);cursor:pointer}
button.t{border-color:var(--yes)} button.f{border-color:var(--no)}
input{font:inherit;width:100%;box-sizing:border-box;padding:6px;border-radius:6px;
border:1px solid var(--bd);background:var(--card);color:var(--fg)}
.row{display:flex;flex-wrap:wrap;gap:8px;align-items:center}
kbd{border:1px solid var(--bd);border-radius:4px;padding:0 4px;font-size:12px}
.legend{font-size:13px} a{color:inherit}
</style></head><body><main>
<header><div><b>membench judge-label</b> <span class="mut" id="who"></span></div>
<div id="progress" class="mut"></div><a href="/export.jsonl">Download JSONL</a></header>
<div class="card rule"><span id="general"></span></div>
<div class="row"><span id="pos" class="mut"></span><span class="mut">metric:</span>
<b id="metric"></b><span id="status" class="status"></span></div>
<div class="card rule"><h2>Rule for this metric</h2><div id="rule"></div></div>
<div class="card"><h2>Statement</h2><div id="stmt" class="stmt"></div></div>
<div class="card"><h2>Evidence</h2><pre id="evidence"></pre></div>
<div class="card" id="promptbox" hidden><h2>Official prompt</h2><pre id="prompt"></pre></div>
<div class="card"><div class="row">
<button class="t" id="btrue">true (y)</button><button class="f" id="bfalse">false (n)</button>
<button id="bskip">skip (s)</button><button id="bclear">clear (u)</button>
<button id="bprev">&larr; prev</button><button id="bnext">next &rarr;</button></div>
<p><input id="note" placeholder="optional note (/ or e to focus, Esc to leave)"></p>
<div class="legend mut"><kbd>y</kbd>/<kbd>1</kbd> true &middot; <kbd>n</kbd>/<kbd>2</kbd>
false &middot; <kbd>s</kbd> skip &middot; <kbd>&larr;</kbd>/<kbd>k</kbd> previous &middot;
<kbd>&rarr;</kbd>/<kbd>j</kbd> next &middot; <kbd>u</kbd> clear label &middot;
<kbd>/</kbd>/<kbd>e</kbd> note &middot; <kbd>Esc</kbd> leave note</div>
<div id="err" class="f"></div></div>
</main><script>
let S = null, i = 0;
const $ = id => document.getElementById(id);
async function load() {
  S = await (await fetch('/api/state')).json();
  i = S.start; $('who').textContent = 'annotator: ' + S.annotator;
  $('general').textContent = S.rules._general; render();
}
function progress() {
  const per = {};
  for (const it of S.items) {
    const p = per[it.metric] = per[it.metric] || [0, 0];
    p[1]++; if (S.labels[it.id]) p[0]++;
  }
  const done = Object.keys(S.labels).length;
  return `labelled ${done} / ${S.items.length} &mdash; ` +
    Object.entries(per).map(([m, p]) => `${m}: ${p[0]}/${p[1]}`).join(' &middot; ');
}
function render() {
  const it = S.items[i], lab = S.labels[it.id];
  $('progress').innerHTML = progress();
  $('pos').textContent = `#${i + 1}/${S.items.length}`;
  $('metric').textContent = it.metric;
  $('rule').textContent = S.rules[it.metric] || '';
  $('stmt').textContent = it.rubric;
  $('evidence').textContent = it.evidence; $('evidence').scrollTop = 0;
  $('promptbox').hidden = !it.prompt; $('prompt').textContent = it.prompt || '';
  $('status').textContent = lab ? 'labelled: ' + lab.label : 'unlabelled';
  $('status').className = 'status ' + (lab ? (lab.label ? 't' : 'f') : 'mut');
  $('note').value = lab && lab.note ? lab.note : '';
  $('err').textContent = '';
}
async function post(label) {
  const it = S.items[i];
  const r = await fetch('/api/label', {method: 'POST', headers: {'Content-Type':
    'application/json'}, body: JSON.stringify({item_id: it.id, label, note: $('note').value})});
  const body = await r.json();
  if (!r.ok) { $('err').textContent = 'Error: ' + body.error; return false; }
  S.labels = body.labels; return true;
}
function nextUnlabelled() {
  for (let k = 1; k <= S.items.length; k++) {
    const j = (i + k) % S.items.length;
    if (!S.labels[S.items[j].id]) return j;
  }
  return i;
}
async function label(v) { if (await post(v)) { i = nextUnlabelled(); render(); } }
async function clearLabel() { if (await post(null)) render(); }
function go(d) { i = Math.min(S.items.length - 1, Math.max(0, i + d)); render(); }
$('btrue').onclick = () => label(true); $('bfalse').onclick = () => label(false);
$('bskip').onclick = () => go(1); $('bclear').onclick = clearLabel;
$('bprev').onclick = () => go(-1); $('bnext').onclick = () => go(1);
document.addEventListener('keydown', e => {
  if (!S) return;
  if (document.activeElement === $('note')) {
    if (e.key === 'Escape') $('note').blur();
    if (e.key === 'Enter') {  // save the edited note of an already labelled item
      $('note').blur(); const lab = S.labels[S.items[i].id];
      if (lab) post(lab.label).then(ok => ok && render());
    }
    return;
  }
  if (e.ctrlKey || e.metaKey || e.altKey) return;
  const k = e.key;
  if (k === 'y' || k === '1') label(true);
  else if (k === 'n' || k === '2') label(false);
  else if (k === 's' || k === 'ArrowRight' || k === 'j') go(1);
  else if (k === 'ArrowLeft' || k === 'k') go(-1);
  else if (k === 'u') clearLabel();
  else if (k === '/' || k === 'e') { e.preventDefault(); $('note').focus(); }
  else return;
  e.preventDefault();
});
load();
</script></body></html>
"""
