"""Evaluation summary across trajectory log entries."""

from numbers import Real


def _is_score_entry(entry):

    step = entry.get("step")
    details = entry.get("details")

    if not isinstance(step, str) or not isinstance(details, dict):
        return False

    if not (step.startswith("streamlit_") or step == "ai_agent_score"):
        return False

    score = details.get("score")
    max_score = details.get("max_score")

    # bool is a subclass of int; a True/False "score" is not a score.
    return (
        isinstance(score, Real) and not isinstance(score, bool)
        and isinstance(max_score, Real) and not isinstance(max_score, bool)
        and max_score > 0
        and 0 <= score <= max_score
    )


def entry_mode(entry):
    """Entries written before modes existed were all live runs."""
    return entry.get("mode") or "live"


def summarize_evaluations(trajectory, mode=None):
    """Summary metrics for scored episodes, optionally for one mode only
    so simulated (demo) scores are never mixed with live AI scores.
    Returns None when there are no evaluations."""

    scores = [
        entry["details"]
        for entry in trajectory
        if isinstance(entry, dict)
        and _is_score_entry(entry)
        and (mode is None or entry_mode(entry) == mode)
    ]

    if not scores:
        return None

    total = len(scores)
    passed = sum(1 for s in scores if s["score"] == s["max_score"])
    total_score = sum(s["score"] for s in scores)
    total_max_score = sum(s["max_score"] for s in scores)

    return {
        "total_evaluations": total,
        "passed": passed,
        "failed": total - passed,
        "success_rate": passed / total * 100,
        "average_score": total_score / total_max_score * 100,
    }
