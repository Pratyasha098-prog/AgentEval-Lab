from jsonschema import validate
from schemas.calendar_schema import calendar_schema


def validate_calendar_event(data):
    validate(instance=data, schema=calendar_schema)
    return True