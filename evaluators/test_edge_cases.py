import json
from datetime import datetime, timedelta

from evaluators.evaluator import Evaluator
from agent import extract_event_details


def test_wrong_event_name():
    evaluator = Evaluator()

    episode = {
        "expected_arguments": {
            "name": "Team Meeting",
            "date": "2026-09-10",
            "time": "10:00"
        }
    }

    actual_result = {
        "name": "Doctor Appointment",
        "date": "2026-09-10",
        "time": "10:00"
    }

    score = evaluator.evaluate(
        episode,
        actual_result
    )

    assert score["name_correct"] is False
    assert score["score"] == 2


def test_wrong_date():
    evaluator = Evaluator()

    episode = {
        "expected_arguments": {
            "name": "Team Meeting",
            "date": "2026-09-10",
            "time": "10:00"
        }
    }

    actual_result = {
        "name": "Team Meeting",
        "date": "2026-09-12",
        "time": "10:00"
    }

    score = evaluator.evaluate(
        episode,
        actual_result
    )

    assert score["date_correct"] is False
    assert score["score"] == 2


def test_wrong_time():
    evaluator = Evaluator()

    episode = {
        "expected_arguments": {
            "name": "Team Meeting",
            "date": "2026-09-10",
            "time": "10:00"
        }
    }

    actual_result = {
        "name": "Team Meeting",
        "date": "2026-09-10",
        "time": "14:00"
    }

    score = evaluator.evaluate(
        episode,
        actual_result
    )

    assert score["time_correct"] is False
    assert score["score"] == 2


def test_all_arguments_wrong():
    evaluator = Evaluator()

    episode = {
        "expected_arguments": {
            "name": "Team Meeting",
            "date": "2026-09-10",
            "time": "10:00"
        }
    }

    actual_result = {
        "name": "Doctor Appointment",
        "date": "2026-09-12",
        "time": "16:00"
    }

    score = evaluator.evaluate(
        episode,
        actual_result
    )

    assert score["name_correct"] is False
    assert score["date_correct"] is False
    assert score["time_correct"] is False
    assert score["score"] == 0


# The two agent tests below use the offline `fake_ai` fixture (see
# conftest.py) with a realistic model reply, so they check how the agent
# parses and returns the response without calling OpenRouter.

def test_missing_date_and_time(fake_ai):

    fake_ai.content = json.dumps({
        "status": "needs_clarification",
        "missing": ["date", "time"],
        "message": "Please provide the date and time."
    })

    result = extract_event_details(
        "Schedule a team meeting"
    )

    assert result["status"] == "needs_clarification"
    assert "date" in result["missing"]
    assert "time" in result["missing"]


def test_relative_date_tomorrow(fake_ai):

    tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")

    fake_ai.content = json.dumps({
        "status": "success",
        "name": "team meeting",
        "date": tomorrow,
        "time": "10:00"
    })

    result = extract_event_details(
        "Schedule a team meeting tomorrow at 10 AM"
    )

    assert result["status"] == "success"
    assert result["date"] != ""
    assert result["time"] == "10:00"