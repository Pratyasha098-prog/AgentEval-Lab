from tools.calendar_tool import create_event


def test_create_event():
    event = create_event(
        name="Rahul",
        date="2026-09-05",
        time="15:00"
    )

    assert event["name"] == "Rahul"
    assert event["status"] == "created"