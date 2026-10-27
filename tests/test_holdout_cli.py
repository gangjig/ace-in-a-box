"""Tests for `ace holdout report`: the release gate and its ledger."""

import json

from ace_in_a_box.cli import main
from ace_in_a_box.evalset import OVERFIT_WARNING, read_ledger, split_cases

CASES = [{"id": f"case-{i:02d}"} for i in range(20)]
SPLIT = split_cases(CASES)


def _write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
    )
    return str(path)


def _scores(drive_score, holdout_score, run=None):
    rows = []
    for case in SPLIT.drive:
        row = {"id": case["id"], "score": drive_score}
        if run:
            row["run"] = run
        rows.append(row)
    for case in SPLIT.holdout:
        row = {"id": case["id"], "score": holdout_score}
        if run:
            row["run"] = run
        rows.append(row)
    return rows


def _run_report(tmp_path, score_rows, *extra_args):
    cases_path = _write_jsonl(tmp_path / "eval.jsonl", CASES)
    scores_path = _write_jsonl(tmp_path / "scores.jsonl", score_rows)
    ledger_path = str(tmp_path / "holdout-ledger.jsonl")
    rc = main([
        "holdout", "report",
        "--cases", cases_path,
        "--scores", scores_path,
        "--ledger", ledger_path,
        *extra_args,
    ])
    return rc, ledger_path


def test_report_warns_when_drive_improves_but_holdout_does_not(
    tmp_path, capsys
):
    rows = _scores(0.5, 0.5, run="before") + _scores(0.9, 0.5, run="after")
    rc, ledger_path = _run_report(tmp_path, rows)
    out = capsys.readouterr().out
    assert rc == 0
    assert OVERFIT_WARNING in out
    assert "Δdrive +0.400" in out
    entries = read_ledger(ledger_path)
    assert len(entries) == 1
    assert entries[0]["verdict"] == OVERFIT_WARNING
    assert entries[0]["fingerprint"] == SPLIT.fingerprint
    assert entries[0]["delta_drive"] == 0.4
    assert entries[0]["delta_holdout"] == 0.0


def test_report_ok_when_holdout_confirms_the_gain(tmp_path, capsys):
    rows = _scores(0.5, 0.5, run="before") + _scores(0.8, 0.75, run="after")
    rc, _ = _run_report(tmp_path, rows)
    out = capsys.readouterr().out
    assert rc == 0
    assert "OK: holdout confirms the drive-set gain" in out
    assert "WARNING" not in out


def test_report_static_gap_without_run_labels(tmp_path, capsys):
    rc, _ = _run_report(tmp_path, _scores(0.9, 0.5))
    out = capsys.readouterr().out
    assert rc == 0
    assert "exceeds --max-gap" in out
    assert "Δdrive" not in out  # no before/after -> no deltas


def test_report_rejects_empty_partition(tmp_path, capsys):
    rc, _ = _run_report(tmp_path, _scores(0.5, 0.5), "--fraction", "0")
    captured = capsys.readouterr()
    assert rc == 2
    assert "empty partition" in captured.err


def test_report_rejects_unknown_score_id(tmp_path, capsys):
    rows = _scores(0.5, 0.5) + [{"id": "case-999", "score": 1.0}]
    rc, _ = _run_report(tmp_path, rows)
    captured = capsys.readouterr()
    assert rc == 2
    assert "unknown case id" in captured.err


def test_report_rejects_unscored_partition(tmp_path, capsys):
    drive_only = [
        {"id": case["id"], "score": 0.5} for case in SPLIT.drive
    ]
    rc, _ = _run_report(tmp_path, drive_only)
    captured = capsys.readouterr()
    assert rc == 2
    assert "no scored cases" in captured.err


def test_ledger_accumulates_a_history(tmp_path, capsys):
    rows = _scores(0.6, 0.55)
    rc1, ledger_path = _run_report(tmp_path, rows)
    capsys.readouterr()
    rc2, _ = _run_report(tmp_path, rows)
    capsys.readouterr()
    assert rc1 == rc2 == 0
    entries = read_ledger(ledger_path)
    assert len(entries) == 2
    assert {e["fingerprint"] for e in entries} == {SPLIT.fingerprint}
    assert all(e["verdict"].startswith("OK") for e in entries)
