"""Demo Mode: pre-written sample AI responses.

These are NOT live model output. They let the full pipeline (calendar
tool, ground-truth evaluation, trajectory logging) be demonstrated
without an API key or internet access. Some samples are deliberately
wrong so the evaluator's failure detection can be seen.
"""

import copy


class DemoTaskNotSupportedError(LookupError):
    """Demo Mode only has sample responses for the listed example tasks."""


# Each sample: the task, a short label for the UI, what it demonstrates,
# and the simulated AI response returned for it.
DEMO_EXAMPLES = [
    {
        "task": "Schedule a Team Meeting on 2026-09-10 at 10:00",
        "label": "✅ Correct extraction (3/3)",
        "demonstrates": "A correct extraction that scores 3/3.",
        "response": {
            "status": "success",
            "name": "Team Meeting",
            "date": "2026-09-10",
            "time": "10:00"
        }
    },
    {
        "task": "Schedule a meeting with Rahul on 2026-09-12 at 15:00",
        "label": "⚠️ Wrong time (2/3)",
        "demonstrates": (
            "A simulated AI mistake: the time is extracted as 16:00 "
            "instead of 15:00, so the evaluator marks Time as FAIL."
        ),
        "response": {
            "status": "success",
            "name": "meeting with Rahul",
            "date": "2026-09-12",
            "time": "16:00"
        }
    },
    {
        "task": "Schedule a project review on 2026-09-15 at 11:30",
        "label": "⚠️ Wrong date (2/3)",
        "demonstrates": (
            "A simulated AI mistake: the date is off by one day, so "
            "the evaluator marks Date as FAIL."
        ),
        "response": {
            "status": "success",
            "name": "project review",
            "date": "2026-09-16",
            "time": "11:30"
        }
    },
    {
        "task": "Schedule a team meeting",
        "label": "❓ Missing information (clarification)",
        "demonstrates": (
            "The request has no date or time, so the agent asks for "
            "clarification instead of guessing."
        ),
        "response": {
            "status": "needs_clarification",
            "missing": ["date", "time"],
            "message": "Please provide the date and time for the meeting."
        }
    },
    {
        "task": "Schedule a dentist appointment on 2026-09-20",
        "label": "❌ Incomplete AI response",
        "demonstrates": (
            "A simulated faulty AI response that claims success but "
            "leaves the time empty. Validation blocks it before the "
            "calendar tool runs."
        ),
        "response": {
            "status": "success",
            "name": "dentist appointment",
            "date": "2026-09-20",
            "time": ""
        }
    },
    {
        "task": "Schedule a Board Meeting on 2026-02-30 at 09:00",
        "label": "📅 Calendar tool failure",
        "demonstrates": (
            "February 30 does not exist, so the (simulated) calendar "
            "tool rejects the event and the failure is reported."
        ),
        "response": {
            "status": "success",
            "name": "Board Meeting",
            "date": "2026-02-30",
            "time": "09:00"
        }
    },
]


def _normalize(task):
    return " ".join(task.split()).casefold()


_RESPONSES = {
    _normalize(example["task"]): example["response"]
    for example in DEMO_EXAMPLES
}


def extract_event_details_demo(user_task):
    """Return the pre-written sample response for an example task."""

    response = _RESPONSES.get(_normalize(user_task))

    if response is None:
        raise DemoTaskNotSupportedError(
            "Demo Mode only has simulated responses for the example "
            "tasks. Pick one from the examples list, or switch to "
            "Live AI Mode to try your own task."
        )

    return copy.deepcopy(response)
