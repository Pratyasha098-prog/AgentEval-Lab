"""Evaluation summary calculations."""

import pytest

from evaluators.evaluator import Evaluator
from evaluators.summary import summarize_evaluations


def score_entry(score, max_score=3, mode=None, step="ai_agent_score"):
    entry = {"step": step, "details": {"score": score, "max_score": max_score}}
    if mode:
        entry["mode"] = mode
    return entry


def test_summary_metrics():

    summary = summarize_evaluations([
        score_entry(3), score_entry(2), score_entry(3), score_entry(0)
    ])

    assert summary["total_evaluations"] == 4
    assert summary["passed"] == 2
    assert summary["failed"] == 2
    assert summary["success_rate"] == pytest.approx(50.0)
    assert summary["average_score"] == pytest.approx(8 / 12 * 100)


def test_summary_is_none_without_evaluations():

    assert summarize_evaluations([]) is None
    assert summarize_evaluations([{"step": "ai_agent", "details": {}}]) is None


def test_summary_separates_demo_and_live():

    trajectory = [
        score_entry(3, mode="demo"),
        score_entry(2, mode="demo"),
        score_entry(1, mode="live"),
        score_entry(3),  # legacy entry without a mode = live
    ]

    demo = summarize_evaluations(trajectory, mode="demo")
    live = summarize_evaluations(trajectory, mode="live")

    assert demo["total_evaluations"] == 2 and demo["passed"] == 1
    assert live["total_evaluations"] == 2 and live["passed"] == 1
    assert summarize_evaluations(trajectory)["total_evaluations"] == 4


def test_summary_counts_legacy_streamlit_steps():

    summary = summarize_evaluations([score_entry(3, step="streamlit_1_score")])

    assert summary["total_evaluations"] == 1


@pytest.mark.parametrize(
    "bad",
    [
        {"step": "ai_agent_score", "details": {"score": "3", "max_score": 3}},
        {"step": "ai_agent_score", "details": {"score": True, "max_score": 3}},
        {"step": "ai_agent_score", "details": {"score": 1, "max_score": 0}},
        {"step": "ai_agent_score", "details": {"score": 5, "max_score": 3}},
        {"step": "ai_agent_score", "details": "oops"},
        {"step": 7, "details": {"score": 1, "max_score": 3}},
        "not a dict",
    ],
)
def test_malformed_score_entries_are_ignored(bad):

    summary = summarize_evaluations([bad, score_entry(3)])

    assert summary["total_evaluations"] == 1
    assert summary["average_score"] == pytest.approx(100.0)


def test_evaluator_treats_equivalent_times_as_equal():

    score = Evaluator().evaluate(
        {"expected_arguments": {"name": "Interview", "date": "2026-09-15",
                                "time": "09:30"}},
        {"name": " interview ", "date": "2026-09-15", "time": "9:30"},
    )

    assert score["score"] == 3


def test_evaluator_scores_missing_fields_as_incorrect():

    score = Evaluator().evaluate(
        {"expected_arguments": {"name": "Team Meeting", "date": "2026-09-10",
                                "time": "10:00"}},
        {"name": None, "date": "2026-09-10"},
    )

    assert score["name_correct"] is False
    assert score["time_correct"] is False
    assert score["score"] == 1
