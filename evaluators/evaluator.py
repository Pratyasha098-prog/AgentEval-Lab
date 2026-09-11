class Evaluator:
    def evaluate(self, expected, actual):

        name_correct = (
            expected["expected_arguments"]["name"]
            == actual["name"]
        )

        date_correct = (
            expected["expected_arguments"]["date"]
            == actual["date"]
        )

        time_correct = (
            expected["expected_arguments"]["time"]
            == actual["time"]
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