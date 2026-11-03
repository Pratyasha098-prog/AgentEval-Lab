"""Demo Mode works without credentials or network, and is always labelled."""

import pytest

import agent
import run_episode
from demo_agent import (
    DEMO_EXAMPLES,
    DemoTaskNotSupportedError,
    extract_event_details_demo,
)
from logs.trajectory_logger import load_trajectory


EXPECTED_STATUS = {
    "✅ Correct extraction (3/3)": ("success", 3),
    "⚠️ Wrong time (2/3)": ("success", 2),
    "⚠️ Wrong date (2/3)": ("success", 2),
    "❓ Missing information (clarification)": ("needs_clarification", None),
    "❌ Incomplete AI response": ("invalid_extraction", None),
    "📅 Calendar tool failure": ("calendar_error", None),
}


@pytest.mark.parametrize(
    "example", DEMO_EXAMPLES, ids=[e["label"] for e in DEMO_EXAMPLES]
)
def test_demo_examples_run_without_credentials(
    no_credentials, monkeypatch, log_path, example
):

    def must_not_call(*args, **kwargs):
        raise AssertionError("Demo Mode must not call the live agent")

    monkeypatch.setattr(run_episode, "extract_event_details", must_not_call)

    result = run_episode.run_single_episode(example["task"], mode="demo")

    status, score = EXPECTED_STATUS[example["label"]]

    assert result["status"] == status
    assert result["mode"] == "demo"
    if score is not None:
        assert result["score"]["score"] == score

    entries = load_trajectory(log_path)
    assert entries
    assert all(entry["mode"] == "demo" for entry in entries)


def test_every_example_has_an_expected_outcome():

    assert {e["label"] for e in DEMO_EXAMPLES} == set(EXPECTED_STATUS)


def test_demo_rejects_unknown_tasks(no_credentials):

    with pytest.raises(DemoTaskNotSupportedError):
        run_episode.run_single_episode("Book a flight to Paris", mode="demo")


def test_demo_responses_are_copies():

    first = extract_event_details_demo(DEMO_EXAMPLES[0]["task"])
    first["name"] = "changed"

    assert extract_event_details_demo(DEMO_EXAMPLES[0]["task"])["name"] != "changed"


def test_demo_mode_never_reports_live(no_credentials):

    result = run_episode.run_single_episode(DEMO_EXAMPLES[0]["task"], mode="demo")

    assert result["mode"] == "demo"
    assert agent.is_live_mode_configured() is False
