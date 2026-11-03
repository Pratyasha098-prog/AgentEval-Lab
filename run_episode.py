import json

from agent import extract_event_details
from demo_agent import extract_event_details_demo
from tools.calendar_tool import create_event
from logs.trajectory_logger import TrajectoryLogger
from evaluators.evaluator import Evaluator
from evaluation_dataset import GROUND_TRUTH


class GroundTruthNotFoundError(LookupError):
    """No ground-truth entry matches the given user task."""


REQUIRED_FIELDS = ("name", "date", "time")

LIVE_MODE = "live"
DEMO_MODE = "demo"


def _normalize_task(task):
    return " ".join(task.split()).casefold()


_GROUND_TRUTH_BY_TASK = {
    _normalize_task(task): expected
    for task, expected in GROUND_TRUTH.items()
}


def find_ground_truth(user_task):
    """Look up the expected arguments for a task. Matching ignores case
    and repeated whitespace, so "schedule a team meeting on ..." finds
    the same entry as "Schedule a Team Meeting on ..."."""

    expected = _GROUND_TRUTH_BY_TASK.get(_normalize_task(user_task))

    if expected is None:
        raise GroundTruthNotFoundError(
            "No ground truth found for this task."
        )

    return expected


def run_single_episode(user_task, mode=LIVE_MODE):
    """Run one episode. mode="live" calls OpenRouter; mode="demo" uses
    pre-written sample responses (see demo_agent.py). Every result and
    log entry records which mode produced it."""

    if mode not in (LIVE_MODE, DEMO_MODE):
        raise ValueError(f"Unknown mode: {mode!r}")

    # st.text_area (unlike input()) returns the raw text exactly as
    # typed, including any trailing/leading whitespace or newline.
    user_task = user_task.strip()

    extract = (
        extract_event_details_demo
        if mode == DEMO_MODE
        else extract_event_details
    )

    logger = TrajectoryLogger(mode=mode)
    evaluator = Evaluator()

    def finish(status, details, calendar_result=None, score=None, **extra):
        return {
            "status": status,
            "mode": mode,
            "extracted_details": details,
            "calendar_result": calendar_result,
            "score": score,
            "log_saved": logger.save(),
            **extra
        }

    # Step 1: AI Agent extracts task details
    # Agent errors (auth, timeout, rate limit, invalid response, ...)
    # are re-raised for the caller to report, but are logged first so
    # failed runs still appear in the trajectory.
    try:
        details = extract(user_task)
    except Exception as error:
        logger.log(
            "ai_agent_error",
            {
                "user_task": user_task,
                "error_type": type(error).__name__,
                "error": str(error)
            }
        )
        logger.save()
        raise

    logger.log(
        "ai_agent",
        {
            "user_task": user_task,
            "extracted_details": details
        }
    )

    if not isinstance(details, dict):

        logger.log(
            "invalid_extraction",
            {
                "extracted_details": details,
                "missing_fields": list(REQUIRED_FIELDS)
            }
        )

        return finish("invalid_extraction", details)

    # Step 2: Check whether AI needs clarification
    if details.get("status") == "needs_clarification":

        logger.log(
            "clarification_required",
            details
        )

        return finish("needs_clarification", details)

    # Step 2b: Validate the extraction before creating a calendar event.
    # A malformed or incomplete "success" response (missing/empty
    # fields) must never reach the calendar tool.
    missing_fields = [
        field
        for field in REQUIRED_FIELDS
        if not isinstance(details.get(field), str)
        or not details[field].strip()
    ]

    if details.get("status") != "success" or missing_fields:

        logger.log(
            "invalid_extraction",
            {
                "extracted_details": details,
                "missing_fields": missing_fields
            }
        )

        return finish("invalid_extraction", details)

    # Step 3: Calendar Tool
    try:
        result = create_event(
            name=details["name"],
            date=details["date"],
            time=details["time"]
        )
    except Exception as error:

        logger.log(
            "calendar_tool_error",
            {
                "error_type": type(error).__name__,
                "error": str(error)
            }
        )

        return finish(
            "calendar_error",
            details,
            calendar_result={
                "status": "failed",
                "error": str(error)
            }
        )

    logger.log(
        "calendar_tool",
        result
    )

    # Step 4: Evaluation
    # Use independent ground truth instead of AI output. A task with no
    # ground-truth entry (e.g. a custom task in Live AI Mode) still
    # completes, but is reported as not evaluated rather than scored.
    try:
        expected = find_ground_truth(user_task)
    except GroundTruthNotFoundError:

        logger.log(
            "ground_truth_missing",
            {"user_task": user_task}
        )

        return finish(
            "success",
            details,
            calendar_result=result,
            evaluated=False
        )

    episode = {
        "id": "ai_agent_episode",
        "user_task": user_task,
        "expected_arguments": expected
    }

    score = evaluator.evaluate(
        episode,
        details
    )

    logger.log(
        "ai_agent_score",
        score
    )

    # Step 5: Save trajectory
    return finish(
        "success",
        details,
        calendar_result=result,
        score=score,
        evaluated=True,
        expected_arguments=expected
    )


def main():

    task = input(
        "Enter calendar task: "
    )

    result = run_single_episode(task)

    print("\nAI Agent Output:")

    print(
        json.dumps(
            result["extracted_details"],
            indent=4
        )
    )

    # Handle clarification
    if result["status"] == "needs_clarification":

        print("\n⚠️ Clarification Required:")

        print(
            result["extracted_details"].get(
                "message",
                "Please provide more information."
            )
        )

        print(
            "Missing:",
            ", ".join(
                str(item)
                for item in result["extracted_details"].get("missing", [])
            )
        )

        return

    # Handle calendar tool failure
    if result["status"] == "calendar_error":

        print(
            "\n❌ Calendar tool failed:",
            result["calendar_result"]["error"]
        )

        return

    # Handle invalid/incomplete extraction
    if result["status"] == "invalid_extraction":

        print(
            "\n❌ AI extraction was invalid or incomplete. "
            "No calendar event was created."
        )

        return

    print("\nCalendar Result:")

    print(
        result["calendar_result"]
    )

    print("\nEvaluation Score:")

    print(
        result["score"]
        if result["evaluated"]
        else "Not evaluated: no ground truth for this task."
    )

    print(
        "\nEpisode completed successfully!"
    )


if __name__ == "__main__":
    main()
