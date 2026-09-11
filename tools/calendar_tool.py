from datetime import datetime


def create_event(name: str, date: str, time: str):
    event = {
        "name": name,
        "date": date,
        "time": time,
        "status": "created"
    }

    return event