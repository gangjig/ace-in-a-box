"""Tests for the Reflector: rule-based extraction and the LLM interface."""

from ace_in_a_box import Reflector

TRACE_WITH_ERROR = [
    {"action": "run_migrations", "observation": "started"},
    {"action": "run_migrations",
     "error": "duplicate column name: email"},
    {"action": "write_report", "lesson": "Summarise counts, not raw rows"},
]
SUCCESS = {"success": True}
FAILURE = {"success": False}


def test_rules_error_becomes_failure_mode():
    candidates = Reflector().reflect(TRACE_WITH_ERROR, FAILURE)
    failure_modes = [c for c in candidates if c.section == "failure_modes"]
    assert len(failure_modes) == 1
    assert "duplicate column name: email" in failure_modes[0].content
    assert "run_migrations" in failure_modes[0].content


def test_rules_lesson_becomes_strategy():
    candidates = Reflector().reflect(TRACE_WITH_ERROR, SUCCESS)
    strategies = [c for c in candidates if c.section == "strategies"]
    assert len(strategies) == 1
    assert strategies[0].content == "Summarise counts, not raw rows"


def test_rules_lesson_section_override():
    trace = [{"action": "deploy", "lesson": "Tag releases before deploying",
              "section": "domain_facts"}]
    candidates = Reflector().reflect(trace, SUCCESS)
    assert candidates[0].section == "domain_facts"


def test_rules_failed_run_without_error_yields_generic_failure_mode():
    trace = [{"action": "click_submit"}, {"action": "wait_for_redirect"}]
    candidates = Reflector().reflect(trace, FAILURE)
    assert len(candidates) == 1
    assert candidates[0].section == "failure_modes"
    assert "wait_for_redirect" in candidates[0].content


def test_rules_successful_quiet_trace_yields_nothing():
    trace = [{"action": "read_file", "observation": "ok"}]
    assert Reflector().reflect(trace, SUCCESS) == []


def test_rules_duplicate_lessons_deduped():
    trace = [
        {"action": "a", "lesson": "Same lesson"},
        {"action": "b", "lesson": "Same lesson"},
    ]
    assert len(Reflector().reflect(trace, SUCCESS)) == 1


class FakeLLM:
    """Records the prompt it was given and returns a canned reply."""

    def __init__(self, reply):
        self.reply = reply
        self.prompts = []

    def __call__(self, prompt):
        self.prompts.append(prompt)
        return self.reply


def test_llm_candidates_parsed_from_lines():
    fake = FakeLLM(
        "strategies | Cache the schema lookup between steps\n"
        "failure_modes | Retrying a 400 error never helps\n"
        "this line has no separator and is ignored\n"
        "unknown_section | Falls back to strategies\n"
    )
    reflector = Reflector(llm=fake)
    candidates = reflector.reflect(TRACE_WITH_ERROR, FAILURE)
    assert [c.content for c in candidates] == [
        "Cache the schema lookup between steps",
        "Retrying a 400 error never helps",
        "Falls back to strategies",
    ]
    assert candidates[2].section == "strategies"
    assert all(c.rationale == "proposed by LLM reflector" for c in candidates)


def test_llm_receives_trace_and_outcome_in_prompt():
    fake = FakeLLM("strategies | something general")
    Reflector(llm=fake).reflect(TRACE_WITH_ERROR, FAILURE)
    prompt = fake.prompts[0]
    assert "run_migrations" in prompt
    assert "Outcome: failure" in prompt


def test_llm_unparseable_reply_falls_back_to_rules():
    fake = FakeLLM("I could not find any lessons, sorry!")
    candidates = Reflector(llm=fake).reflect(TRACE_WITH_ERROR, FAILURE)
    assert any(c.section == "failure_modes" for c in candidates)
    assert any("duplicate column" in c.content for c in candidates)


def test_llm_exception_falls_back_to_rules():
    def boom(prompt):
        raise RuntimeError("model unavailable")

    candidates = Reflector(llm=boom).reflect(TRACE_WITH_ERROR, FAILURE)
    assert any("duplicate column" in c.content for c in candidates)
