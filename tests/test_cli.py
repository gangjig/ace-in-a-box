"""Smoke tests for the CLI: every subcommand through main() and the shell."""

import json
import subprocess
import sys

from ace_in_a_box import load_playbook
from ace_in_a_box.cli import main


def test_cli_full_workflow(tmp_path, capsys):
    path = str(tmp_path / "playbook.json")
    # init
    assert main(["init", path]) == 0
    capsys.readouterr()

    # add two bullets
    assert main(["add", path, "--section", "strategies",
                 "--content", "Run tests before committing"]) == 0
    first_id = capsys.readouterr().out.strip()
    assert first_id == "b-0001"
    assert main(["add", path, "--section", "strategies",
                 "--content", "Skip tests to save time"]) == 0
    second_id = capsys.readouterr().out.strip()

    # feedback: first bullet good (x2), second bullet harmful (x4)
    assert main(["feedback", path, first_id, "--helpful"]) == 0
    assert main(["feedback", path, first_id, "--helpful"]) == 0
    for _ in range(4):
        assert main(["feedback", path, second_id, "--harmful"]) == 0
    capsys.readouterr()

    playbook = load_playbook(path)
    assert playbook.get(first_id).net_score == 2
    assert playbook.get(second_id).net_score == -4

    # prune removes the harmful bullet
    assert main(["prune", path, "--threshold", "-2"]) == 0
    out = capsys.readouterr().out
    assert "pruned 1" in out
    playbook = load_playbook(path)
    assert len(playbook) == 1
    assert playbook.bullets[0].id == first_id

    # show + render produce the surviving bullet
    assert main(["show", path]) == 0
    assert "Run tests before committing" in capsys.readouterr().out
    assert main(["render", path, "--max-chars", "4000"]) == 0
    assert "Run tests before committing" in capsys.readouterr().out


def test_cli_init_refuses_overwrite(tmp_path, capsys):
    path = str(tmp_path / "playbook.json")
    assert main(["init", path]) == 0
    capsys.readouterr()
    assert main(["init", path]) == 2
    assert "already exists" in capsys.readouterr().err
    assert main(["init", path, "--force"]) == 0


def test_cli_feedback_unknown_bullet(tmp_path, capsys):
    path = str(tmp_path / "playbook.json")
    assert main(["init", path]) == 0
    capsys.readouterr()
    assert main(["feedback", path, "b-9999", "--helpful"]) == 2
    assert "no bullet with id" in capsys.readouterr().err


def test_cli_markdown_store_round_trip(tmp_path, capsys):
    path = str(tmp_path / "playbook.md")
    assert main(["init", path]) == 0
    assert main(["add", path, "--content", "Markdown stores work too"]) == 0
    capsys.readouterr()
    playbook = load_playbook(path)
    assert playbook.bullets[0].content == "Markdown stores work too"


def test_cli_module_invocation(tmp_path):
    """`python -m ace_in_a_box` works end to end in a subprocess."""
    path = str(tmp_path / "playbook.json")
    result = subprocess.run(
        [sys.executable, "-m", "ace_in_a_box", "init", path],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    with open(path) as fh:
        assert json.load(fh) == {"version": 1, "bullets": []}


def _playbook_with_bullet(tmp_path, capsys):
    path = str(tmp_path / "playbook.json")
    assert main(["init", path]) == 0
    assert main(["add", path, "--content",
                 "Run tests before committing"]) == 0
    capsys.readouterr()
    return path


def test_cli_render_with_brief(tmp_path, capsys):
    path = _playbook_with_bullet(tmp_path, capsys)
    assert main(["render", path,
                 "--goal", "Ship the fix",
                 "--verify", "the test suite passes",
                 "--effort", "one focused hour"]) == 0
    out = capsys.readouterr().out
    assert "# Task brief" in out
    assert "**Goal:** Ship the fix" in out
    assert "**Verify:** the test suite passes" in out
    assert "**Effort:** one focused hour" in out
    assert out.index("# Task brief") < out.index("Run tests before committing")


def test_cli_render_goal_only_brief(tmp_path, capsys):
    path = _playbook_with_bullet(tmp_path, capsys)
    assert main(["render", path, "--goal", "Ship the fix"]) == 0
    out = capsys.readouterr().out
    assert "**Goal:** Ship the fix" in out
    assert "**Verify:**" not in out
    assert "**Effort:**" not in out


def test_cli_render_verify_without_goal_is_usage_error(tmp_path, capsys):
    path = _playbook_with_bullet(tmp_path, capsys)
    assert main(["render", path, "--verify", "tests pass"]) == 2
    assert "--goal" in capsys.readouterr().err


def test_cli_render_effort_without_goal_is_usage_error(tmp_path, capsys):
    path = _playbook_with_bullet(tmp_path, capsys)
    assert main(["render", path, "--effort", "one hour"]) == 2
    assert "--goal" in capsys.readouterr().err
