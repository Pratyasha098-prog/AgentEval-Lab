from evaluators.evaluator import Evaluator


def test_evaluator():

    expected = {
        "expected_arguments": {
            "name": "John",
            "date": "2026-09-10",
            "time": "10:00"
        }
    }

    actual = {
        "name": "John",
        "date": "2026-09-10",
        "time": "10:00"
    }

    evaluator = Evaluator()

    result = evaluator.evaluate(expected, actual)

    assert result["name_correct"] is True
    assert result["date_correct"] is True
    assert result["time_correct"] is True
    assert result["score"] == 3
    assert result["max_score"] == 3


def test_evaluator_detects_wrong_date():

    expected = {
        "expected_arguments": {
            "name": "John",
            "date": "2026-09-10",
            "time": "10:00"
        }
    }

    actual = {
        "name": "John",
        "date": "2026-09-11",
        "time": "10:00"
    }

    evaluator = Evaluator()

    result = evaluator.evaluate(expected, actual)

    assert result["name_correct"] is True
    assert result["date_correct"] is False
    assert result["time_correct"] is True
    assert result["score"] == 2
    assert result["max_score"] == 3