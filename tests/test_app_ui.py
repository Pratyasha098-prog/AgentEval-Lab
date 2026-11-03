"""Streamlit UI tests: run app.py headlessly (offline) and check that
every path renders without an unhandled exception."""

from pathlib import Path

import openai
import pytest
from streamlit.testing.v1 import AppTest

from conftest import status_error
from demo_agent import DEMO_EXAMPLES


APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")


def all_text(at):
    parts = []
    for kind in ("markdown", "info", "success", "warning", "error", "caption"):
        parts.extend(str(element.value) for element in getattr(at, kind))
    return "\n".join(parts)


def start_app():
    at = AppTest.from_file(APP_PATH, default_timeout=30).run()
    assert not at.exception
    return at


def run_task(at, task, mode="demo"):
    at.radio(key="mode").set_value(mode).run()
    at.text_area(key="user_task").input(task).run()
    at.button(key="run").click().run()
    assert not at.exception, at.exception
    return at


def test_app_loads_in_demo_mode_without_credentials(no_credentials):

    at = start_app()

    assert at.title[0].value == "🤖 AgentEval Lab"
    assert at.radio(key="mode").value == "demo"
    assert "results are simulated" in all_text(at)


@pytest.mark.parametrize(
    "example", DEMO_EXAMPLES, ids=[e["label"] for e in DEMO_EXAMPLES]
)
def test_every_demo_example_renders(no_credentials, example):

    at = run_task(start_app(), example["task"])

    text = all_text(at)

    assert "SIMULATED RESULT" in text
    assert "LIVE RESULT" not in text


def test_demo_success_and_failure_scores_are_shown(no_credentials):

    at = run_task(start_app(), DEMO_EXAMPLES[0]["task"])
    assert "Evaluation Passed — 3/3" in all_text(at)

    at = run_task(at, DEMO_EXAMPLES[1]["task"])
    assert "Evaluation Score — 2/3" in all_text(at)
    assert any(metric.value == "FAIL" for metric in at.metric)

    # The summary covers both demo runs: one passed, one failed.
    assert "1 evaluation(s) failed" in all_text(at)


def test_example_selector_fills_the_task(no_credentials):

    at = start_app()
    at.selectbox(key="example").set_value(DEMO_EXAMPLES[3]["label"]).run()

    assert at.text_area(key="user_task").value == DEMO_EXAMPLES[3]["task"]


def test_unknown_task_in_demo_mode_is_explained(no_credentials):

    at = run_task(start_app(), "Book a flight to Paris")

    assert "only has simulated responses" in all_text(at)


def test_empty_task_shows_a_warning(no_credentials):

    at = start_app()
    at.button(key="run").click().run()

    assert not at.exception
    assert "Please enter a user task" in all_text(at)


def test_live_mode_without_credentials_shows_error(no_credentials):

    at = run_task(start_app(), DEMO_EXAMPLES[0]["task"], mode="live")

    text = all_text(at)

    assert "Live AI Mode is not configured" in text
    assert "SIMULATED RESULT" not in text


def test_live_auth_failure_offers_retry(fake_ai):

    fake_ai.error = status_error(openai.AuthenticationError, 401)

    at = run_task(start_app(), DEMO_EXAMPLES[0]["task"], mode="live")

    assert "Authentication with OpenRouter failed" in all_text(at)
    assert "sk-test-dummy-key" not in all_text(at)
    assert fake_ai.calls == 1

    at.button(key="retry").click().run()

    assert not at.exception
    assert fake_ai.calls == 2


def test_live_success_is_labelled_live(fake_ai):

    fake_ai.content = (
        '{"status": "success", "name": "Team Meeting", '
        '"date": "2026-09-10", "time": "10:00"}'
    )

    at = run_task(start_app(), DEMO_EXAMPLES[0]["task"], mode="live")

    text = all_text(at)

    assert "LIVE RESULT" in text
    assert "SIMULATED RESULT" not in text
    assert "Evaluation Passed — 3/3" in text
