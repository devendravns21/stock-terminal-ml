"""Local trainer dashboard: `python -m stock_terminal_ml.dashboard` → http://localhost:8765

Serves `dashboard/index.html` (a static page) plus a small JSON API it polls:

  GET  /api/info              datasets, presets, objectives, search space, defaults, paths
  GET  /api/main              is the main backend/frontend up, which models it serves
  GET  /api/runs              runs started from this dashboard (newest first)
  POST /api/runs              write config + run.sh, open a new Terminal window running the trainer
  GET  /api/runs/<id>         rolled-up progress (?symbol=S adds that symbol's trial series)
  POST /api/runs/<id>/stop    terminate the run's process group
  GET  /api/models            every artifact (full JSON) in the main repo's ml_models dir + unpublished run outputs

Run state lives in `outputs/runs/<run_id>/` (gitignored): config.json, run.sh, pid,
progress.jsonl, terminal.log, exit_code, cancelled. Stdlib only.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
import urllib.request
import webbrowser
from dataclasses import fields
from datetime import date, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from app.core.config import settings
from app.integrations.offline.store import DEFAULT_DATASET, list_datasets
from app.integrations.signals.config import Params

from .objectives import BY_NAME as OBJECTIVES
from .search_space import RANGES, TUNED_KEYS, WIDEN_KEYS
from .train import PRESETS

REPO = Path(__file__).resolve().parents[1]
PAGE = REPO / "dashboard" / "index.html"
RUNS = REPO / "outputs" / "runs"
MODELS_DIR = Path(settings.ml_models_dir)
MAIN_REPO = MODELS_DIR.parents[4] if len(MODELS_DIR.parents) > 4 else MODELS_DIR
BACKEND_JOBS = Path(settings.kite_session_file).parent / ".ml_jobs"
MAIN_BACKEND = os.environ.get("MAIN_BACKEND_URL", "http://localhost:8000")
MAIN_FRONTEND = os.environ.get("MAIN_FRONTEND_URL", "http://localhost:5173")

_MODEL_RE = re.compile(r"^dev_signal_ml(\d+)$")
_RUN_RE = re.compile(r"^[0-9]{8}-[0-9]{6}$")
_SYMBOL_RE = re.compile(r"^[A-Za-z0-9 &_\-]+$")

OBJECTIVE_HELP = {
    "expected_pnl_with_constraints": "Avg PnL% per trade, scaled down below 30 trades and above 10% drawdown (default, smooth).",
    "pure_pnl": "Net PnL% over the window. Can reward a few lucky trades.",
    "profit_factor": "Gross profit / gross loss.",
    "sharpe_like": "Mean / stdev of per-trade net %.",
}


# ---------------------------------------------------------------- helpers

def _read_json(p: Path, default=None):
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return default


def _pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _run_pid(rdir: Path) -> int | None:
    try:
        return int((rdir / "pid").read_text().strip())
    except (OSError, ValueError):
        return None


def _model_numbers() -> set[int]:
    """Every dev_signal_mlN taken: on disk, in our runs, and in the backend's own jobs."""
    taken: set[int] = set()
    for p in MODELS_DIR.glob("dev_signal_ml*.json"):
        m = _MODEL_RE.match(p.stem)
        if m:
            taken.add(int(m.group(1)))
    for root in (RUNS, BACKEND_JOBS):
        if not root.is_dir():
            continue
        for cfg in root.glob("*/config.json"):
            m = _MODEL_RE.match(str((_read_json(cfg, {}) or {}).get("model_id") or ""))
            if m:
                taken.add(int(m.group(1)))
    return taken


def next_model_id() -> str:
    taken = _model_numbers()
    return f"dev_signal_ml{(max(taken) + 1) if taken else 1}"


def _http_json(url: str, timeout: float = 1.0):
    with urllib.request.urlopen(url, timeout=timeout) as r:  # noqa: S310 — localhost only
        return json.loads(r.read().decode() or "null")


# ---------------------------------------------------------------- info

def info() -> dict:
    defaults = Params()
    datasets = []
    for m in list_datasets():
        datasets.append({
            "dataset": m.get("dataset"),
            "title": m.get("title"),
            "from": (m.get("from") or "")[:10],
            "to": (m.get("to") or "")[:10],
            "symbols": [
                {"symbol": s["symbol"], "exchange": s.get("exchange"), "kind": s.get("kind"),
                 "days": s.get("days"), "last_close": s.get("last_close")}
                for s in m.get("symbols", [])
            ],
        })
    space = []
    for key in TUNED_KEYS + WIDEN_KEYS:
        kind, lo, hi, log = RANGES[key]
        space.append({"key": key, "kind": kind, "low": lo, "high": hi, "log": log,
                      "widen_only": key in WIDEN_KEYS, "default": getattr(defaults, key)})
    return {
        "datasets": datasets,
        "default_dataset": DEFAULT_DATASET,
        "presets": PRESETS,
        "objectives": [{"name": k, "help": OBJECTIVE_HELP.get(k, "")} for k in OBJECTIVES],
        "search_space": space,
        "defaults": {f.name: getattr(defaults, f.name) for f in fields(Params)
                     if isinstance(getattr(defaults, f.name), (int, float, str, bool))},
        "next_model_id": next_model_id(),
        "cpu_count": os.cpu_count() or 1,
        "paths": {
            "ml_repo": str(REPO),
            "main_repo": str(MAIN_REPO),
            "models_dir": str(MODELS_DIR),
            "runs_dir": str(RUNS),
            "python": sys.executable,
        },
        "main_backend": MAIN_BACKEND,
        "main_frontend": MAIN_FRONTEND,
        "platform": sys.platform,
    }


def main_status() -> dict:
    out = {"backend_up": False, "frontend_up": False, "models": [], "backend_error": None}
    try:
        rows = _http_json(f"{MAIN_BACKEND}/api/ml/models")
        out["backend_up"] = True
        out["models"] = [r.get("id") for r in rows or []]
    except Exception as e:  # noqa: BLE001
        out["backend_error"] = str(e)
    try:
        with urllib.request.urlopen(MAIN_FRONTEND, timeout=1.0):  # noqa: S310
            out["frontend_up"] = True
    except Exception:  # noqa: BLE001
        pass
    return out


# ---------------------------------------------------------------- start a run

class BadRequest(ValueError):
    pass


def _validate(body: dict) -> dict:
    try:
        f = date.fromisoformat(body["from"])
        t = date.fromisoformat(body["to"])
    except (KeyError, TypeError, ValueError):
        raise BadRequest("from/to must be YYYY-MM-DD dates")
    if f >= t:
        raise BadRequest("'from' must be before 'to'")
    preset = str(body.get("preset") or "fast").lower()
    if preset not in PRESETS:
        raise BadRequest(f"unknown preset {preset!r}")
    objective = body.get("objective") or "expected_pnl_with_constraints"
    if objective not in OBJECTIVES:
        raise BadRequest(f"unknown objective {objective!r}")
    dataset = body.get("dataset") or DEFAULT_DATASET
    known = {d["dataset"]: d for d in info()["datasets"]}
    if dataset not in known:
        raise BadRequest(f"unknown dataset {dataset!r}")
    all_syms = [s["symbol"] for s in known[dataset]["symbols"]]
    symbols = body.get("symbols") or []
    if not isinstance(symbols, list) or not all(isinstance(s, str) and _SYMBOL_RE.match(s) for s in symbols):
        raise BadRequest("symbols must be a list of symbol names")
    unknown = [s for s in symbols if s not in all_syms]
    if unknown:
        raise BadRequest(f"not in dataset: {', '.join(unknown)}")
    if not symbols:
        raise BadRequest("pick at least one symbol")
    try:
        target = float(body.get("target_pct", 1.0))
        sl = float(body.get("sl_pct", 0.5))
        trials = int(body["trials"]) if body.get("trials") not in (None, "") else None
        jobs = int(body.get("jobs") or 0)
    except (TypeError, ValueError):
        raise BadRequest("target/stop/trials/jobs must be numbers")
    if not (0 < target <= 20 and 0 < sl <= 20):
        raise BadRequest("target % and stop % must be between 0 and 20")
    if trials is not None and not (1 <= trials <= 2000):
        raise BadRequest("trials must be 1–2000")
    jobs = max(0, min(jobs, os.cpu_count() or 1))
    label = str(body.get("label") or "").strip()[:60]
    return {
        "from": f.isoformat(), "to": t.isoformat(), "target_pct": target, "sl_pct": sl,
        "preset": preset, "trials": trials, "objective": objective, "dataset": dataset,
        "symbols": symbols, "all_symbols": len(symbols) == len(all_syms),
        "widen": bool(body.get("widen")), "jobs": jobs, "label": label,
        "publish": body.get("publish", True) is not False,
    }


def _terminal_script(rdir: Path, cmd: list[str], cfg: dict) -> str:
    q = shlex.quote
    banner = [
        f"Dev Signal ML trainer · run {rdir.name}",
        f"model    {cfg['model_id']}  ({cfg['label']})",
        f"window   {cfg['window']['from']} → {cfg['window']['to']}   target {cfg['target_pct']}%  stop {cfg['sl_pct']}%",
        f"search   {cfg['preset']} · {cfg['trials_per_symbol']} trials/symbol · {cfg['objective']}"
        + (" · widen" if cfg["widen"] else ""),
        f"symbols  {len(cfg['symbols'])} from {cfg['dataset']}",
        f"output   {cfg['output']}",
    ]
    lines = [
        "#!/bin/bash",
        f"cd {q(str(REPO))} || exit 1",
        f"echo $$ > {q(str(rdir / 'pid'))}",
        "export PYTHONUNBUFFERED=1 ML_PROGRESS_ECHO=1",
        "clear",
        "printf '\\033[1;35m%s\\033[0m\\n' " + q("━" * 64),
        *[f"echo {q(b)}" for b in banner],
        "printf '\\033[1;35m%s\\033[0m\\n' " + q("━" * 64),
        "echo",
        "echo " + q("$ " + " ".join(q(c) for c in cmd)),
        "echo",
        " ".join(q(c) for c in cmd) + f" 2>&1 | tee {q(str(rdir / 'terminal.log'))}",
        "rc=${PIPESTATUS[0]}",
        f"echo $rc > {q(str(rdir / 'exit_code'))}",
        "echo",
        'if [ "$rc" = "0" ]; then',
        "  printf '\\033[1;32m%s\\033[0m\\n' " + q(f"✔ Training finished. Model: {cfg['output']}"),
        "else",
        "  printf '\\033[1;31m%s\\033[0m\\n' \"✘ Trainer exited with code $rc\"",
        "fi",
        "echo 'Results are in the dashboard. You can close this window.'",
    ]
    return "\n".join(lines) + "\n"


def _open_terminal(script: Path) -> str:
    """Open a new visible terminal running `bash script`. Returns how it was launched."""
    if sys.platform == "darwin":
        cmd = f"bash {shlex.quote(str(script))}".replace("\\", "\\\\").replace('"', '\\"')
        apple = f'tell application "Terminal"\n  activate\n  do script "{cmd}"\nend tell'
        subprocess.run(["osascript", "-e", apple], check=True, capture_output=True, timeout=15)
        return "Terminal.app"
    for term in (["x-terminal-emulator", "-e"], ["gnome-terminal", "--"], ["xterm", "-e"]):
        try:
            subprocess.Popen([*term, "bash", str(script)], start_new_session=True)
            return term[0]
        except FileNotFoundError:
            continue
    # Headless fallback: still runs, output only visible in terminal.log / the dashboard.
    subprocess.Popen(["bash", str(script)], start_new_session=True,
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return "background"


def start_run(body: dict) -> dict:
    v = _validate(body)
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    while (RUNS / run_id).exists():
        time.sleep(1)
        run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    rdir = RUNS / run_id
    rdir.mkdir(parents=True)

    model_id = next_model_id()
    label = v["label"] or model_id.replace("dev_signal_ml", "Dev Signal ML")
    output = (MODELS_DIR / f"{model_id}.json") if v["publish"] else (rdir / f"{model_id}.json")
    progress = rdir / "progress.jsonl"
    trials = v["trials"] or PRESETS[v["preset"]]["trials"]
    workers = v["jobs"] or max(1, (os.cpu_count() or 2) - 1)

    cmd = [
        sys.executable, "-m", "stock_terminal_ml.train",
        "--window", f"{v['from']}:{v['to']}",
        "--target-pct", str(v["target_pct"]),
        "--sl-pct", str(v["sl_pct"]),
        "--preset", v["preset"],
        "--dataset", v["dataset"],
        "--objective", v["objective"],
        "--model-id", model_id,
        "--label", label,
        "--output", str(output),
        "--progress", str(progress),
        "--jobs", str(v["jobs"]),
    ]
    if v["trials"]:
        cmd += ["--trials", str(v["trials"])]
    if not v["all_symbols"]:
        cmd += ["--symbols", ",".join(v["symbols"])]
    if v["widen"]:
        cmd.append("--widen")

    cfg = {
        "run_id": run_id,
        "model_id": model_id,
        "label": label,
        "window": {"from": v["from"], "to": v["to"]},
        "target_pct": v["target_pct"],
        "sl_pct": v["sl_pct"],
        "preset": v["preset"],
        "trials_per_symbol": trials,
        "objective": v["objective"],
        "dataset": v["dataset"],
        "symbols": v["symbols"],
        "widen": v["widen"],
        "workers": workers,
        "publish": v["publish"],
        "output": str(output),
        "progress": str(progress),
        "command": cmd,
        "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    (rdir / "config.json").write_text(json.dumps(cfg, indent=2))
    script = rdir / "run.sh"
    script.write_text(_terminal_script(rdir, cmd, cfg))
    script.chmod(0o755)
    try:
        cfg["launched_in"] = _open_terminal(script)
    except (subprocess.SubprocessError, OSError) as e:
        err = getattr(e, "stderr", b"") or b""
        raise BadRequest(f"could not open a terminal: {err.decode(errors='replace') or e}")
    (rdir / "config.json").write_text(json.dumps(cfg, indent=2))
    return summarize(run_id)


# ---------------------------------------------------------------- progress

def _status(rdir: Path, cfg: dict) -> str:
    code = (rdir / "exit_code")
    if code.exists():
        try:
            return "done" if int(code.read_text().strip()) == 0 else "error"
        except ValueError:
            return "error"
    if (rdir / "cancelled").exists():
        return "cancelled"
    pid = _run_pid(rdir)
    if pid is None:
        started = cfg.get("started_at")
        try:
            age = time.time() - datetime.fromisoformat(started).timestamp()
        except (TypeError, ValueError):
            age = 1e9
        return "starting" if age < 60 else "interrupted"
    return "running" if _pid_alive(pid) else "interrupted"


def _tail(p: Path, n: int, max_bytes: int = 64_000) -> list[str]:
    try:
        with p.open("rb") as fh:
            fh.seek(0, 2)
            size = fh.tell()
            fh.seek(max(0, size - max_bytes))
            data = fh.read().decode(errors="replace")
    except OSError:
        return []
    return data.splitlines()[-n:]


def summarize(run_id: str, symbol: str | None = None) -> dict:
    rdir = RUNS / run_id
    cfg = _read_json(rdir / "config.json", {}) or {}
    status = _status(rdir, cfg)
    syms = {s: {"symbol": s, "trials": 0, "best": None, "last": None, "state": "queued",
                "pruned": 0, "failed": 0, "error": None} for s in cfg.get("symbols", [])}
    events: list[dict] = []
    series: list[dict] = []
    run_start = run_done = None
    try:
        raw = (rdir / "progress.jsonl").read_text().splitlines()
    except OSError:
        raw = []
    for line in raw:
        try:
            ev = json.loads(line)
        except ValueError:
            continue  # partial last line while the writer is mid-flush
        kind = ev.get("event")
        s = syms.get(ev.get("symbol")) if ev.get("symbol") else None
        if kind == "trial" and s is not None:
            s["trials"] += 1
            if s["state"] == "queued":
                s["state"] = "running"
            st = ev.get("state")
            if st == "pruned":
                s["pruned"] += 1
            elif st == "fail":
                s["failed"] += 1
            val = ev.get("value")
            if val is not None:
                s["last"] = val
                if s["best"] is None or val > s["best"]:
                    s["best"] = val
            if symbol and ev.get("symbol") == symbol:
                series.append({"trial": ev.get("trial"), "value": val, "state": st})
        elif kind == "symbol_done" and s is not None:
            s["state"] = "done"
            s["best"] = ev.get("best_value", s["best"])
        elif kind == "error":
            if s is not None:
                s["state"] = "error"
                s["error"] = ev.get("message")
        elif kind == "run_start":
            run_start = ev
        elif kind == "run_done":
            run_done = ev
        if kind != "trial":
            events.append(ev)
    if status in ("cancelled", "interrupted", "error"):
        for s in syms.values():
            if s["state"] in ("running", "queued"):
                s["state"] = "stopped"

    per_sym_trials = int(cfg.get("trials_per_symbol") or 0)
    total = per_sym_trials * len(syms)
    done_trials = sum(min(s["trials"], per_sym_trials) if per_sym_trials else s["trials"] for s in syms.values())
    started = cfg.get("started_at")
    try:
        t0 = datetime.fromisoformat(started).timestamp()
    except (TypeError, ValueError):
        t0 = None
    end_ts = None
    for name in ("exit_code", "cancelled"):
        p = rdir / name
        if p.exists():
            end_ts = p.stat().st_mtime
    elapsed = ((end_ts or time.time()) - t0) if t0 else None
    eta = None
    if status == "running" and elapsed and done_trials and total > done_trials:
        eta = elapsed / done_trials * (total - done_trials)

    output = Path(cfg.get("output", "")) if cfg.get("output") else None
    return {
        "run_id": run_id,
        "status": status,
        "config": cfg,
        "pid": _run_pid(rdir),
        "symbols": list(syms.values()),
        "symbols_total": len(syms),
        "symbols_done": sum(1 for s in syms.values() if s["state"] == "done"),
        "symbols_error": sum(1 for s in syms.values() if s["state"] == "error"),
        "trials_total": total,
        "trials_done": done_trials,
        "elapsed_s": elapsed,
        "eta_s": eta,
        "run_started": run_start is not None,
        "run_done": run_done,
        "artifact_exists": bool(output and output.is_file()),
        "events": events[-60:],
        "log_tail": _tail(rdir / "terminal.log", 80),
        "series": series if symbol else None,
        "series_symbol": symbol,
    }


def list_runs() -> list[dict]:
    if not RUNS.is_dir():
        return []
    out = []
    for d in sorted((p for p in RUNS.iterdir() if p.is_dir() and _RUN_RE.match(p.name)), reverse=True)[:30]:
        cfg = _read_json(d / "config.json", {}) or {}
        out.append({
            "run_id": d.name,
            "status": _status(d, cfg),
            "model_id": cfg.get("model_id"),
            "label": cfg.get("label"),
            "started_at": cfg.get("started_at"),
            "symbols": len(cfg.get("symbols", [])),
            "preset": cfg.get("preset"),
            "window": cfg.get("window"),
        })
    return out


def stop_run(run_id: str) -> dict:
    rdir = RUNS / run_id
    pid = _run_pid(rdir)
    if not _pid_alive(pid):
        raise BadRequest("run is not running")
    (rdir / "cancelled").write_text(datetime.now().isoformat())
    try:
        pgid = os.getpgid(pid)
        if pgid != os.getpgrp():
            os.killpg(pgid, signal.SIGTERM)  # bash + tee + trainer + pool workers
        else:
            os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    return summarize(run_id)


# ---------------------------------------------------------------- models

def _model_files() -> list[tuple[Path, bool]]:
    files = [(p, True) for p in sorted(MODELS_DIR.glob("dev_signal_ml*.json"))]
    if RUNS.is_dir():  # unpublished outputs (publish=false)
        files += [(p, False) for p in sorted(RUNS.glob("*/dev_signal_ml*.json"))]
    return files


def list_models() -> list[dict]:
    out = []
    for p, published in _model_files():
        data = _read_json(p)
        if not isinstance(data, dict):
            continue
        data["_file"] = str(p)
        data["_published"] = published
        data["_ref"] = p.stem if published else f"{p.parent.name}/{p.stem}"
        data["_mtime"] = p.stat().st_mtime
        out.append(data)
    out.sort(key=lambda d: d["_mtime"], reverse=True)
    return out


# ---------------------------------------------------------------- http

class Handler(BaseHTTPRequestHandler):
    server_version = "stock-terminal-ml-dashboard"

    def log_message(self, fmt, *args):  # quieter console: only errors
        if args and str(args[1] if len(args) > 1 else "").startswith(("4", "5")):
            super().log_message(fmt, *args)

    def _send(self, code: int, payload, ctype="application/json"):
        body = payload if isinstance(payload, bytes) else json.dumps(payload, default=str).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _route(self, method: str):
        url = urlparse(self.path)
        parts = [p for p in url.path.split("/") if p]
        qs = parse_qs(url.query)
        try:
            if method == "GET" and parts in ([], ["index.html"]):
                return self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            if parts[:1] != ["api"]:
                return self._send(404, {"detail": "not found"})
            rest = parts[1:]
            if method == "GET" and rest == ["info"]:
                return self._send(200, info())
            if method == "GET" and rest == ["main"]:
                return self._send(200, main_status())
            if rest == ["runs"]:
                if method == "GET":
                    return self._send(200, list_runs())
                if method == "POST":
                    n = int(self.headers.get("Content-Length") or 0)
                    body = json.loads(self.rfile.read(n) or b"{}")
                    return self._send(200, start_run(body))
            if len(rest) >= 2 and rest[0] == "runs":
                run_id = rest[1]
                if not _RUN_RE.match(run_id) or not (RUNS / run_id).is_dir():
                    return self._send(404, {"detail": "unknown run"})
                if method == "GET" and len(rest) == 2:
                    sym = (qs.get("symbol") or [None])[0]
                    return self._send(200, summarize(run_id, sym))
                if method == "POST" and rest[2:] == ["stop"]:
                    return self._send(200, stop_run(run_id))
            if method == "GET" and rest == ["models"]:
                return self._send(200, list_models())
            return self._send(404, {"detail": "not found"})
        except BadRequest as e:
            return self._send(400, {"detail": str(e)})
        except Exception as e:  # noqa: BLE001
            return self._send(500, {"detail": f"{type(e).__name__}: {e}"})

    def do_GET(self):
        self._route("GET")

    def do_POST(self):
        self._route("POST")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="stock-terminal-ml.dashboard")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args(argv)
    RUNS.mkdir(parents=True, exist_ok=True)
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://localhost:{args.port}/"
    print(f"Dev Signal ML dashboard → {url}  (Ctrl-C to stop)")
    print(f"  models dir: {MODELS_DIR}")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
