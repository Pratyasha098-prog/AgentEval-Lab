from datetime import datetime


class CalendarToolError(ValueError):
    """The calendar rejected the event (e.g. an impossible date)."""


def create_event(name: str, date: str, time: str):
    """Simulated calendar: validates the event like a real calendar API
    would, but never contacts an external calendar."""

    try:
        datetime.strptime(date, "%Y-%m-%d")
    except (TypeError, ValueError):
        raise CalendarToolError(
            f"Invalid date '{date}': expected a real date in "
            "YYYY-MM-DD format."
        )

    try:
        datetime.strptime(time, "%H:%M")
    except (TypeError, ValueError):
        raise CalendarToolError(
            f"Invalid time '{time}': expected 24-hour HH:MM format."
        )

    event = {
        "name": name,
        "date": date,
        "time": time,
        "status": "created",
        "simulated": True
    }

    return event
