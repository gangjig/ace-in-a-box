"""Persistence: save and load playbooks as JSON or human-readable Markdown.

JSON is the lossless format (every field, including timestamps, round-trips
exactly).  Markdown is the ACE-style format: a person — or an agent — can
read and hand-edit it in any editor.  The Markdown format preserves ids,
sections, contents and feedback counters; timestamps are *not* stored and
are re-stamped on load.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from .playbook import Bullet, Playbook

FORMAT_VERSION = 1

_BULLET_LINE_RE = re.compile(
    r"^- \[(?P<id>[^\]]+)\](?:\s+\(\+(?P<helpful>\d+)/-(?P<harmful>\d+)\))?"
    r"\s+(?P<content>.+)$"
)
_SECTION_RE = re.compile(r"^##\s+(?P<section>\S.*?)\s*$")


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------

def playbook_to_dict(playbook: Playbook) -> dict:
    return {
        "version": FORMAT_VERSION,
        "bullets": [b.to_dict() for b in playbook.bullets],
    }


def playbook_from_dict(data: dict) -> Playbook:
    if int(data.get("version", FORMAT_VERSION)) != FORMAT_VERSION:
        raise ValueError(f"unsupported playbook format version: {data.get('version')!r}")
    return Playbook([Bullet.from_dict(b) for b in data.get("bullets", [])])


def dumps_json(playbook: Playbook) -> str:
    return json.dumps(playbook_to_dict(playbook), indent=2, ensure_ascii=False) + "\n"


def loads_json(text: str) -> Playbook:
    return playbook_from_dict(json.loads(text))


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------

def to_markdown(playbook: Playbook) -> str:
    lines = ["# Playbook", ""]
    for section in playbook.sections():
        lines.append(f"## {section}")
        lines.append("")
        for bullet in playbook.by_section(section):
            lines.append(
                f"- [{bullet.id}] "
                f"(+{bullet.helpful_count}/-{bullet.harmful_count}) "
                f"{bullet.content}"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def from_markdown(text: str) -> Playbook:
    bullets: list[Bullet] = []
    section: str | None = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        section_match = _SECTION_RE.match(line)
        if section_match:
            section = section_match.group("section")
            continue
        bullet_match = _BULLET_LINE_RE.match(line)
        if bullet_match and section is not None:
            bullets.append(
                Bullet(
                    id=bullet_match.group("id"),
                    section=section,
                    content=bullet_match.group("content"),
                    helpful_count=int(bullet_match.group("helpful") or 0),
                    harmful_count=int(bullet_match.group("harmful") or 0),
                )
            )
    return Playbook(bullets)


# ---------------------------------------------------------------------------
# File helpers (format chosen by suffix: .json / .md / .markdown)
# ---------------------------------------------------------------------------

def save_playbook(playbook: Playbook, path: str | Path) -> Path:
    path = Path(path)
    if path.suffix == ".json":
        path.write_text(dumps_json(playbook), encoding="utf-8")
    elif path.suffix in (".md", ".markdown"):
        path.write_text(to_markdown(playbook), encoding="utf-8")
    else:
        raise ValueError(
            f"cannot infer playbook format from suffix {path.suffix!r}; "
            "use .json or .md"
        )
    return path


def load_playbook(path: str | Path) -> Playbook:
    path = Path(path)
    if path.suffix not in (".json", ".md", ".markdown"):
        raise ValueError(
            f"cannot infer playbook format from suffix {path.suffix!r}; "
            "use .json or .md"
        )
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".json":
        return loads_json(text)
    return from_markdown(text)
