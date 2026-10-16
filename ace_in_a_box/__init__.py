"""ace-in-a-box: a mini self-evolving playbook framework.

Implements the ACE (Agentic Context Engineering) pattern in pure Python
with zero required dependencies: an agent's context is a structured
*playbook* of bullets that a Generator uses, a Reflector distills execution
traces into candidate bullets, and a Curator integrates feedback —
incrementing counters, merging duplicates, pruning harmful bullets — so
the context itself evolves between runs.
"""

from .brief import TaskBrief, render_brief
from .curator import Curator, jaccard, tokenize
from .persistence import (
    from_markdown,
    load_playbook,
    loads_json,
    dumps_json,
    save_playbook,
    to_markdown,
)
from .playbook import SECTIONS, Bullet, Playbook
from .reflector import Candidate, Reflector
from .render import render_prompt

__version__ = "0.2.0"

__all__ = [
    "SECTIONS",
    "Bullet",
    "Playbook",
    "TaskBrief",
    "render_brief",
    "Curator",
    "jaccard",
    "tokenize",
    "Reflector",
    "Candidate",
    "render_prompt",
    "save_playbook",
    "load_playbook",
    "dumps_json",
    "loads_json",
    "to_markdown",
    "from_markdown",
    "__version__",
]
