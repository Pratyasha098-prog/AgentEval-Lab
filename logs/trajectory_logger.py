import json
import uuid
from datetime import datetime


def log_step(step, action, details):
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "episode_id": str(uuid.uuid4()),
        "step": step,
        "action": action,
        "details": details
    }

    with open("logs/trajectory.jsonl", "a") as file:
        file.write(json.dumps(log_entry) + "\n")

    return log_entry


class TrajectoryLogger:

    def __init__(self):
        self.episode_id = str(uuid.uuid4())
        self.entries = []

    def log(self, step, details):

        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "episode_id": self.episode_id,
            "step": step,
            "details": details
        }

        self.entries.append(log_entry)

    def save(self):

        with open("logs/trajectory.jsonl", "a") as file:

            for entry in self.entries:
                file.write(
                    json.dumps(entry) + "\n"
                )