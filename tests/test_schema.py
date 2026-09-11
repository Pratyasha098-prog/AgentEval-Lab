from schemas.validator import validate_calendar_event


def test_valid_calendar_event():
    data = {
        "name": "Rahul",
        "date": "2026-09-05",
        "time": "15:00"
    }

    assert validate_calendar_event(data) is True