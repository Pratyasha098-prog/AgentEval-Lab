"""End-to-end episode tests (offline): every outcome of run_single_episode."""

import json

import openai
import pytest

import agent
import run_episode
from conftest import REQUEST, status_error
from logs.trajectory_logger import load_trajectory


TASK = "Schedule a Team Meeting on 2026-09-10 at 10:00"


def reply(**fields):
    return json.dumps({"status": "success", **fields})


def steps(log_path):
    return [entry["step"] for entry in load_trajectory(log_path)]


def test_successful_live_episode(fake_ai, log_path):

    fake_ai.content = reply(
        name="Team Meeting", date="2026-09-10", time="10:00"
    )

    result = run_episode.run_single_episode(TASK + "\n")

    assert result["status"] == "success"
    assert result["mode"] == "live"
    assert result["evaluated"] is True
    assert result["score"]["score"] == 3
    assert result["calendar_result"]["status"] == "created"
    assert steps(log_path) == ["ai_agent", "calendar_tool", "ai_agent_score"]


@pytest.mark.parametrize(
    "fields, failed_field",
    [
        ({"name": "Doctor Appointment", "date": "2026-09-10",
          "time": "10:00"}, "name_correct"),
        ({"name": "Team Meeting", "date": "2026-09-11",
          "time": "10:00"}, "date_correct"),
        ({"name": "Team Meeting", "date": "2026-09-10",
          "time": "14:00"}, "time_correct"),
    ],
    ids=["wrong-name", "wrong-date", "wrong-time"],
)
def test_incorrect_extraction_scores_2_of_3(fake_ai, fields, failed_field):

    fake_ai.content = reply(**fields)

    score = run_episode.run_single_episode(TASK)["score"]

    assert score["score"] == 2
    assert score[failed_field] is False


def test_missing_information_requests_clarification(fake_ai, log_path):

    fake_ai.content = json.dumps({
        "status": "needs_clarification",
        "missing": ["date", "time"],
        "message": "Please provide the date and time."
    })

    result = run_episode.run_single_episode("Schedule a team meeting")

    assert result["status"] == "needs_clarification"
    assert result["calendar_result"] is None
    assert steps(log_path) == ["ai_agent", "clarification_required"]


@pytest.mark.parametrize(
    "fields",
    [
        {"name": "Team Meeting", "date": "", "time": "10:00"},
        {"name": "Team Meeting", "date": "2026-09-10", "time": 10},
        {"name": "Team Meeting", "date": "2026-09-10"},
    ],
    ids=["empty-date", "non-string-time", "missing-time"],
)
def test_incomplete_extraction_never_reaches_calendar(fake_ai, log_path, fields):

    fake_ai.content = reply(**fields)

    result = run_episode.run_single_episode(TASK)

    assert result["status"] == "invalid_extraction"
    assert result["calendar_result"] is None
    assert steps(log_path) == ["ai_agent", "invalid_extraction"]


def test_unknown_status_is_invalid(fake_ai):

    fake_ai.content = json.dumps({"status": "maybe", "name": "x"})

    assert run_episode.run_single_episode(TASK)["status"] == "invalid_extraction"


def test_invalid_json_is_reported_and_logged(fake_ai, log_path):

    fake_ai.content = "Sorry, I cannot help with that."

    with pytest.raises(agent.AgentInvalidResponseError):
        run_episode.run_single_episode(TASK)

    entries = load_trajectory(log_path)
    assert [e["step"] for e in entries] == ["ai_agent_error"]
    assert entries[0]["details"]["error_type"] == "AgentInvalidResponseError"


@pytest.mark.parametrize(
    "error, expected",
    [
        (status_error(openai.AuthenticationError, 401),
         agent.AgentAuthenticationError),
        (openai.APITimeoutError(request=REQUEST), agent.AgentTimeoutError),
        (openai.APIConnectionError(request=REQUEST),
         agent.AgentConnectionError),
        (status_error(openai.RateLimitError, 429), agent.AgentRateLimitError),
    ],
    ids=["auth", "timeout", "connection", "rate-limit"],
)
def test_api_failures_are_raised_and_logged(fake_ai, log_path, error, expected):

    fake_ai.error = error

    with pytest.raises(expected):
        run_episode.run_single_episode(TASK)

    assert steps(log_path) == ["ai_agent_error"]


def test_live_episode_does_not_fall_back_to_demo(no_credentials, log_path):

    with pytest.raises(agent.AgentConfigurationError):
        run_episode.run_single_episode(TASK, mode="live")

    entries = load_trajectory(log_path)
    assert entries[0]["mode"] == "live"
    assert entries[0]["step"] == "ai_agent_error"


def test_calendar_tool_failure(fake_ai, log_path):

    fake_ai.content = reply(name="Board Meeting", date="2026-02-30", time="09:00")

    result = run_episode.run_single_episode(TASK)

    assert result["status"] == "calendar_error"
    assert result["calendar_result"]["status"] == "failed"
    assert "Invalid date" in result["calendar_result"]["error"]
    assert result["score"] is None
    assert steps(log_path) == ["ai_agent", "calendar_tool_error"]


def test_calendar_tool_exception_is_caught(fake_ai, monkeypatch):

    fake_ai.content = reply(name="Team Meeting", date="2026-09-10", time="10:00")

    def broken(**kwargs):
        raise RuntimeError("calendar service down")

    monkeypatch.setattr(run_episode, "create_event", broken)

    result = run_episode.run_single_episode(TASK)

    assert result["status"] == "calendar_error"
    assert result["calendar_result"]["error"] == "calendar service down"


def test_ground_truth_not_found(fake_ai, log_path):

    fake_ai.content = reply(name="Lunch", date="2026-10-01", time="13:00")

    result = run_episode.run_single_episode(
        "Schedule lunch on 2026-10-01 at 13:00"
    )

    assert result["status"] == "success"
    assert result["evaluated"] is False
    assert result["score"] is None
    assert steps(log_path) == ["ai_agent", "calendar_tool", "ground_truth_missing"]


def test_find_ground_truth_ignores_case_and_spacing():

    expected = run_episode.find_ground_truth(
        "  schedule a team   meeting on 2026-09-10 at 10:00 "
    )

    assert expected["name"] == "Team Meeting"

    with pytest.raises(run_episode.GroundTruthNotFoundError):
        run_episode.find_ground_truth("Schedule something else")


def test_unknown_mode_is_rejected():

    with pytest.raises(ValueError):
        run_episode.run_single_episode(TASK, mode="other")
