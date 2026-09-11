import json

from agent import extract_event_details
from tools.calendar_tool import create_event
from logs.trajectory_logger import TrajectoryLogger
from evaluators.evaluator import Evaluator
from evaluation_dataset import GROUND_TRUTH


def run_single_episode(user_task):

    logger = TrajectoryLogger()
    evaluator = Evaluator()

    # Step 1: AI Agent extracts task details
    details = extract_event_details(user_task)

    logger.log(
        "ai_agent",
        {
            "user_task": user_task,
            "extracted_details": details
        }
    )

    # Step 2: Check whether AI needs clarification
    if details.get("status") == "needs_clarification":

        logger.log(
            "clarification_required",
            details
        )

        logger.save()

        return {
            "status": "needs_clarification",
            "extracted_details": details,
            "calendar_result": None,
            "score": None
        }

    # Step 3: Calendar Tool
    result = create_event(
        name=details["name"],
        date=details["date"],
        time=details["time"]
    )

    logger.log(
        "calendar_tool",
        result
    )

    # Step 4: Evaluation
    # Use independent ground truth instead of AI output
    if user_task not in GROUND_TRUTH:
        raise ValueError(
            "No ground truth found for this task."
        )

    episode = {
        "id": "ai_agent_episode",
        "user_task": user_task,
        "expected_arguments": GROUND_TRUTH[user_task]
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
    logger.save()

    return {
        "status": "success",
        "extracted_details": details,
        "calendar_result": result,
        "score": score
    }


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
            result["extracted_details"]["message"]
        )

        print(
            "Missing:",
            ", ".join(
                result["extracted_details"]["missing"]
            )
        )

        return

    print("\nCalendar Result:")

    print(
        result["calendar_result"]
    )

    print("\nEvaluation Score:")

    print(
        result["score"]
    )

    print(
        "\nEpisode completed successfully!"
    )


if __name__ == "__main__":
    main()