calendar_schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "date": {"type": "string"},
        "time": {"type": "string"}
    },
    "required": ["name", "date", "time"]
}