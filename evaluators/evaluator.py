from datetime import datetime


def _text(value):
    return value.strip() if isinstance(value, str) else ""


def _same_value(expected, actual, fmt):
    """Equal if both parse to the same date/time (so "9:30" matches
    "09:30"); otherwise fall back to exact string comparison."""

    expected, actual = _text(expected), _text(actual)

    try:
        return (
            datetime.strptime(expected, fmt)
            == datetime.strptime(actual, fmt)
        )
    except ValueError:
        return expected != "" and expected == actual


class Evaluator:
    def evaluate(self, expected, actual):

        expected_args = expected["expected_arguments"]

        # A missing or non-string field scores as incorrect rather
        # than crashing the evaluation.
        name_correct = (
            _text(expected_args.get("name")).lower()
            == _text(actual.get("name")).lower()
            != ""
        )

        date_correct = _same_value(
            expected_args.get("date"),
            actual.get("date"),
            "%Y-%m-%d"
        )

        time_correct = _same_value(
            expected_args.get("time"),
            actual.get("time"),
            "%H:%M"
        )

        score = sum([
            name_correct,
            date_correct,
            time_correct
        ])

        return {
            "name_correct": name_correct,
            "date_correct": date_correct,
            "time_correct": time_correct,
            "score": score,
            "max_score": 3
        }
