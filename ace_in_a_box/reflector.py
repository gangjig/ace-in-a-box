"""The Reflector: distill execution traces into candidate bullets.

After a task run, the Reflector looks at the trace (what the agent did,
step by step) and the outcome, and *proposes* new playbook bullets.  The
Curator stays in charge of actually integrating them, so proposals are
always reviewable data, never silent mutations.

Two extraction strategies are provided:

* **Rule-based** (default, zero dependencies): errors become
  ``failure_modes`` bullets, explicit ``lesson`` fields in steps become
  ``strategies`` bullets, and a failed run with no recorded error still
  yields a failure-mode candidate pointing at the last step.
* **Pluggable LLM**: pass any callable ``llm(prompt: str) -> str``.  The
  reflector builds a summarising prompt and parses the reply, which must
  contain one candidate per line as ``section | content``.  If the reply
  cannot be parsed (or the callable raises), the reflector falls back to
  the rule-based strategy, so reflection never hard-fails a pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass

from .playbook import SECTIONS, clean_text

#: A trace is a list of step dicts.  Recognised keys: ``action`` (what the
#: agent did), ``observation``, ``error``, ``lesson`` and ``section``
#: (section override for a lesson).  Unknown keys are ignored.
Trace = list[dict]

#: An outcome is a dict like ``{"success": True, "feedback": "..."}``.
Outcome = dict


@dataclass
class Candidate:
    """A proposed bullet, awaiting Curator review/integration."""

    section: str
    content: str
    rationale: str = ""

    def __post_init__(self) -> None:
        self.content = clean_text(self.content)


class Reflector:
    def __init__(self, llm=None) -> None:
        """``llm`` is an optional callable taking a prompt string and
        returning a string of ``section | content`` lines."""
        self.llm = llm

    # -- public API -----------------------------------------------------------
    def reflect(self, trace: Trace, outcome: Outcome) -> list[Candidate]:
        """Propose candidate bullets from one finished task run."""
        if self.llm is not None:
            try:
                candidates = self._from_llm(trace, outcome)
            except Exception:
                candidates = []
            if candidates:
                return _dedupe(candidates)
        return _dedupe(self._from_rules(trace, outcome))

    # -- rule-based extraction --------------------------------------------------
    def _from_rules(self, trace: Trace, outcome: Outcome) -> list[Candidate]:
        candidates: list[Candidate] = []
        saw_error = False
        for step in trace:
            action = str(step.get("action", "a step")).strip() or "a step"
            error = step.get("error")
            if error:
                saw_error = True
                candidates.append(
                    Candidate(
                        section="failure_modes",
                        content=f"Avoid: {action} failed with error: {error}",
                        rationale=f"step {action!r} raised an error",
                    )
                )
            lesson = step.get("lesson")
            if lesson:
                candidates.append(
                    Candidate(
                        section=str(step.get("section", "strategies")),
                        content=str(lesson),
                        rationale=f"explicit lesson recorded at step {action!r}",
                    )
                )
        if not outcome.get("success", False) and trace and not saw_error:
            last_action = str(trace[-1].get("action", "the final step"))
            candidates.append(
                Candidate(
                    section="failure_modes",
                    content=(
                        f"Task failed without a recorded error; the last "
                        f"step was: {last_action}. Investigate before retrying."
                    ),
                    rationale="failed outcome with no error in trace",
                )
            )
        return candidates

    # -- LLM extraction -----------------------------------------------------------
    def build_prompt(self, trace: Trace, outcome: Outcome) -> str:
        lines = [
            "You are the Reflector in an ACE-style agent system. Read the",
            "execution trace below and propose reusable playbook bullets:",
            "lessons a future agent should remember. Reply with ONE bullet",
            "per line, formatted exactly as:  section | content",
            f"Valid sections: {', '.join(SECTIONS)}.",
            "Reply with no bullets (empty) if nothing generalises.",
            "",
            f"Outcome: {'success' if outcome.get('success') else 'failure'}"
            + (f" — {outcome['feedback']}" if outcome.get("feedback") else ""),
            "",
            "Trace:",
        ]
        for i, step in enumerate(trace, start=1):
            lines.append(f"{i}. {step}")
        return "\n".join(lines)

    def _from_llm(self, trace: Trace, outcome: Outcome) -> list[Candidate]:
        reply = self.llm(self.build_prompt(trace, outcome))
        candidates: list[Candidate] = []
        for line in str(reply).splitlines():
            if "|" not in line:
                continue
            section, _, content = line.partition("|")
            section, content = section.strip(), content.strip()
            if not content:
                continue
            if section not in SECTIONS:
                section = "strategies"
            candidates.append(
                Candidate(
                    section=section,
                    content=content,
                    rationale="proposed by LLM reflector",
                )
            )
        return candidates


def _dedupe(candidates: list[Candidate]) -> list[Candidate]:
    seen: set[tuple[str, str]] = set()
    unique: list[Candidate] = []
    for cand in candidates:
        key = (cand.section, cand.content.lower())
        if key not in seen:
            seen.add(key)
            unique.append(cand)
    return unique
