"""Task 6.1 / 6.2 — `train_meta` CLI flags, config precedence, and progress events."""
import json

import pytest

from stock_terminal_ml import artifact, train_meta
from stock_terminal_ml.meta import train_meta as meta_train

from ._toy import toy_artifact

SYMS = ["RELIANCE", "TCS"]


def _events(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_help_lists_meta_flags(capsys):
    with pytest.raises(SystemExit):
        train_meta.main(["--mode", "meta", "--help"])
    out = capsys.readouterr().out
    for flag in ("--mode", "--train-months", "--test-months", "--embargo-days",
                 "--model-family", "--calibrate", "--no-calibrate", "--threshold-objective"):
        assert flag in out


def test_params_mode_rejects_meta_flags():
    assert train_meta.main(["--mode", "params", "--window", "2026-01-01:2026-02-01",
                            "--train-months", "4"]) == 2


def test_params_mode_delegates_to_v1(monkeypatch):
    seen = {}
    monkeypatch.setattr(train_meta.v1, "main", lambda argv: seen.setdefault("argv", argv) and 0)
    train_meta.main(["--mode", "params", "--window", "2026-01-01:2026-02-01", "--jobs", "1"])
    assert seen["argv"] == ["--window", "2026-01-01:2026-02-01", "--jobs", "1"]


def test_meta_mode_events_and_config_precedence(tmp_path, monkeypatch):
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps({
        "mode": "meta", "window": {"from": "2026-01-01", "to": "2026-10-31"},
        "symbols": SYMS, "train_months": 4, "embargo_days": 2, "calibrate": False,
        "cv_mode": "sliding",
    }))
    out = tmp_path / "dev_signal_ml9.json"
    prog = tmp_path / "progress.jsonl"
    captured = {}

    def fake_run(cfg, progress):
        captured.update(cfg)
        progress.emit("stage", name="features", pct=0.2)
        progress.emit("fold", index=0, train_from="2026-01-01", train_to="2026-03-31",
                      test_from="2026-04-01", test_to="2026-04-30", metrics={"pr_auc": 0.6})
        a = toy_artifact(cfg["model_id"])
        payload = artifact.build_v2(model_id=a["id"], label=cfg["label"], primary=a["primary"],
                                    model=a["model"], features=a["features"], training=a["training"],
                                    evaluation=a["evaluation"], per_symbol={},
                                    global_fallback=a["global_fallback"])
        return artifact.write_atomic_v2(payload, cfg["output"])

    monkeypatch.setattr(meta_train, "run", fake_run)
    # --test-months is explicit (non-default) so it wins; train_months/embargo/calibrate come from config.
    rc = train_meta.main(["--config", str(cfg_path), "--output", str(out), "--progress", str(prog),
                          "--test-months", "2"])
    assert rc == 0
    assert captured["train_months"] == 4 and captured["test_months"] == 2
    assert captured["embargo_days"] == 2 and captured["calibrate"] is False
    assert captured["symbols"] == SYMS and captured["model_id"] == "dev_signal_ml9"
    assert captured["extra"] == {"cv_mode": "sliding"}
    ev = _events(prog)
    assert [e["event"] for e in ev] == ["run_start", "stage", "fold", "run_done"]
    assert ev[0]["symbols_total"] == 2 and ev[0]["mode"] == "meta"
    assert ev[-1]["artifact"] == str(out)

    from app.integrations.ml.runner import _aggregate
    rolled = _aggregate(ev)
    assert rolled["symbols_total"] == 2 and rolled["artifact"] == str(out)


def test_meta_mode_errors_on_no_samples(tmp_path):
    """With a symbol that isn't in the dataset, `run` can't produce samples and
    bubbles the error event back out through the CLI glue."""
    prog = tmp_path / "p.jsonl"
    rc = train_meta.main(["--mode", "meta", "--window", "2026-01-01:2026-10-31",
                          "--symbols", "__DOES_NOT_EXIST__", "--output", str(tmp_path / "m.json"),
                          "--progress", str(prog)])
    assert rc == 1
    events = [e["event"] for e in _events(prog)]
    assert events[0] == "run_start" and events[-1] == "error"
