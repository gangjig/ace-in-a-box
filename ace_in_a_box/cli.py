"""Command-line interface: ``ace`` (or ``python -m ace_in_a_box``).

The playbook file format is chosen by suffix: ``.json`` (lossless) or
``.md`` (human-editable).  Every mutating command loads the file, applies
the operation and saves it back.

Examples::

    ace init playbook.json
    ace add playbook.json --section strategies --content "Run the tests before committing"
    ace feedback playbook.json b-0001 --helpful
    ace show playbook.json
    ace prune playbook.json --threshold -2 --max-bullets 100
    ace render playbook.json --max-chars 4000
    ace render playbook.json --goal "Fix the parser" --verify "pytest passes" \
        --effort "a focused hour; stop if blocked"
    ace holdout report --cases eval.jsonl --scores scores.jsonl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .brief import TaskBrief
from .curator import Curator
from .evalset import (
    DEFAULT_SEED,
    append_ledger,
    build_report,
    load_cases,
    load_scores,
    split_cases,
)
from .persistence import load_playbook, save_playbook, to_markdown
from .playbook import Playbook
from .render import render_prompt


def _load(path: str) -> Playbook:
    if not Path(path).exists():
        raise SystemExit(f"error: playbook file not found: {path}")
    return load_playbook(path)


def _cmd_init(args: argparse.Namespace) -> int:
    path = Path(args.path)
    if path.exists() and not args.force:
        print(f"error: {path} already exists (use --force to overwrite)",
              file=sys.stderr)
        return 2
    save_playbook(Playbook(), path)
    print(f"created empty playbook at {path}")
    return 0


def _cmd_show(args: argparse.Namespace) -> int:
    playbook = _load(args.path)
    print(to_markdown(playbook), end="")
    print(f"({len(playbook)} bullet(s))", file=sys.stderr)
    return 0


def _cmd_add(args: argparse.Namespace) -> int:
    playbook = _load(args.path)
    bullet = Curator(playbook).add_bullet(args.section, args.content)
    save_playbook(playbook, args.path)
    print(bullet.id)
    return 0


def _cmd_feedback(args: argparse.Namespace) -> int:
    playbook = _load(args.path)
    try:
        bullet = Curator(playbook).apply_feedback(
            args.bullet_id, helpful=args.helpful
        )
    except KeyError as exc:
        print(f"error: {exc.args[0]}", file=sys.stderr)
        return 2
    save_playbook(playbook, args.path)
    print(
        f"{bullet.id}: helpful={bullet.helpful_count} "
        f"harmful={bullet.harmful_count} net={bullet.net_score:+d}"
    )
    return 0


def _cmd_prune(args: argparse.Namespace) -> int:
    playbook = _load(args.path)
    curator = Curator(playbook)
    merged = curator.merge_duplicates(args.merge_threshold) \
        if args.merge_threshold is not None else []
    pruned = curator.prune(args.threshold)
    evicted = curator.enforce_cap(args.max_bullets) \
        if args.max_bullets is not None else []
    save_playbook(playbook, args.path)
    print(
        f"merged {sum(len(m['absorbed']) for m in merged)}, "
        f"pruned {len(pruned)}, evicted {len(evicted)}; "
        f"{len(playbook)} bullet(s) remain"
    )
    return 0


def _cmd_render(args: argparse.Namespace) -> int:
    brief = None
    if args.goal is not None:
        try:
            brief = TaskBrief(
                goal=args.goal, verify=args.verify, effort=args.effort
            )
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
    elif args.verify is not None or args.effort is not None:
        print("error: --verify and --effort require --goal", file=sys.stderr)
        return 2
    playbook = _load(args.path)
    text = render_prompt(
        playbook,
        brief=brief,
        max_chars=args.max_chars,
        max_tokens=args.max_tokens,
    )
    print(text, end="")
    return 0


def _cmd_holdout_report(args: argparse.Namespace) -> int:
    for label, path in (("cases", args.cases), ("scores", args.scores)):
        if not Path(path).exists():
            print(f"error: {label} file not found: {path}", file=sys.stderr)
            return 2
    try:
        split = split_cases(
            load_cases(args.cases),
            seed=args.seed,
            holdout_fraction=args.fraction,
        )
        report = build_report(split, load_scores(args.scores),
                              max_gap=args.max_gap)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(
        f"eval set: {len(split.drive) + len(split.holdout)} case(s) — "
        f"drive {len(split.drive)}, holdout {len(split.holdout)} "
        f"(seed {split.seed!r}, fraction {split.holdout_fraction:g})"
    )
    print(f"fingerprint: {split.fingerprint}")
    for run in report.runs:
        label = f"run {run.run!r}: " if run.run else ""
        print(
            f"{label}drive {run.drive_mean:.3f} (n={run.drive_n})  "
            f"holdout {run.holdout_mean:.3f} (n={run.holdout_n})  "
            f"gap {run.gap:+.3f}"
        )
    if report.delta_drive is not None:
        print(
            f"Δdrive {report.delta_drive:+.3f}  "
            f"Δholdout {report.delta_holdout:+.3f}"
        )
    for warning in report.warnings:
        print(warning)
    if not report.warnings:
        print(report.verdict)
    if not args.no_ledger:
        append_ledger(args.ledger, report.ledger_entry())
        print(f"ledger: appended to {args.ledger}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ace",
        description="ace-in-a-box: maintain a self-evolving agent playbook",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("init", help="create an empty playbook file")
    p.add_argument("path")
    p.add_argument("--force", action="store_true", help="overwrite if it exists")
    p.set_defaults(func=_cmd_init)

    p = sub.add_parser("show", help="print the playbook as Markdown")
    p.add_argument("path")
    p.set_defaults(func=_cmd_show)

    p = sub.add_parser("add", help="add a bullet and print its id")
    p.add_argument("path")
    p.add_argument("--section", default="strategies")
    p.add_argument("--content", required=True)
    p.set_defaults(func=_cmd_add)

    p = sub.add_parser("feedback", help="record a helpful/harmful vote")
    p.add_argument("path")
    p.add_argument("bullet_id")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--helpful", action="store_true")
    group.add_argument("--harmful", action="store_false", dest="helpful")
    p.set_defaults(func=_cmd_feedback)

    p = sub.add_parser("prune", help="merge duplicates, prune and cap the playbook")
    p.add_argument("path")
    p.add_argument("--threshold", type=int, default=-2,
                   help="prune bullets with net score below this (default -2)")
    p.add_argument("--max-bullets", type=int, default=None,
                   help="evict lowest-value bullets beyond this size")
    p.add_argument("--merge-threshold", type=float, default=None,
                   help="if set, first merge duplicates at this similarity")
    p.set_defaults(func=_cmd_prune)

    p = sub.add_parser("render", help="render the playbook for a prompt")
    p.add_argument("path")
    p.add_argument("--max-chars", type=int, default=None)
    p.add_argument("--max-tokens", type=int, default=None)
    p.add_argument("--goal", default=None,
                   help="task brief: what to do (enables the brief header)")
    p.add_argument("--verify", default=None,
                   help="task brief: how to verify the work (requires --goal)")
    p.add_argument("--effort", default=None,
                   help="task brief: how much effort to spend (requires --goal)")
    p.set_defaults(func=_cmd_render)

    p = sub.add_parser(
        "holdout",
        help="monitor-holdout: score the cases the playbook never learns from",
    )
    holdout_sub = p.add_subparsers(dest="holdout_command", required=True)
    pr = holdout_sub.add_parser(
        "report",
        help="report drive vs holdout scores side by side (release gate)",
    )
    pr.add_argument("--cases", required=True,
                    help="eval set JSONL (one case object with an 'id' per line)")
    pr.add_argument("--scores", required=True,
                    help="scores JSONL ({'id', 'score' 0..1, optional 'run'})")
    pr.add_argument("--seed", default=DEFAULT_SEED,
                    help=f"split seed (default {DEFAULT_SEED})")
    pr.add_argument("--fraction", type=float, default=0.5,
                    help="holdout fraction (default 0.5)")
    pr.add_argument("--max-gap", type=float, default=0.1,
                    help="warn when the drive/holdout gap exceeds this "
                         "(default 0.1)")
    pr.add_argument("--ledger", default="holdout-ledger.jsonl",
                    help="ledger file to append the verdict to "
                         "(default holdout-ledger.jsonl)")
    pr.add_argument("--no-ledger", action="store_true",
                    help="do not append this report to the ledger")
    pr.set_defaults(func=_cmd_holdout_report)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
