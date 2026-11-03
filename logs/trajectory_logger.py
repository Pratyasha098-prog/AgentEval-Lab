import json
import os
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path


# =========================================================
# LOG LOCATION
# =========================================================
# Resolved relative to the project root (not the current working
# directory), so the app logs to the same file however it is launched.
# AGENTEVAL_LOG_PATH overrides it -- the test suite points it at a
# temporary file so test runs never touch the real trajectory history.

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_LOG_PATH = PROJECT_ROOT / "logs" / "trajectory.jsonl"

LOCK_TIMEOUT_SECONDS = 10

_thread_lock = threading.Lock()


def get_log_path():

    override = os.getenv("AGENTEVAL_LOG_PATH")

    return Path(override) if override else DEFAULT_LOG_PATH


@contextmanager
def _file_lock(log_path):
    """Exclusive cross-process lock (a sidecar .lock file) so that
    concurrent Streamlit sessions cannot interleave partial lines.
    If the lock cannot be acquired in time, writing proceeds anyway:
    a demo should never hang on logging."""

    lock_path = log_path.with_name(log_path.name + ".lock")

    with _thread_lock, open(lock_path, "a+") as lock_file:

        locked = False
        deadline = time.monotonic() + LOCK_TIMEOUT_SECONDS

        while not locked and time.monotonic() < deadline:
            try:
                if os.name == "nt":
                    import msvcrt
                    lock_file.seek(0)
                    msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
                locked = True
            except OSError:
                time.sleep(0.05)

        try:
            yield
        finally:
            if locked:
                if os.name == "nt":
                    import msvcrt
                    lock_file.seek(0)
                    msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(lock_file, fcntl.LOCK_UN)


def _append_entries(entries, log_path=None):
    """Append entries as JSON lines in a single locked write. Always
    appends -- existing history is never truncated."""

    log_path = Path(log_path) if log_path else get_log_path()
    log_path.parent.mkdir(parents=True, exist_ok=True)

    payload = "".join(
        json.dumps(entry, default=str) + "\n"
        for entry in entries
    )

    with _file_lock(log_path):
        with open(log_path, "a", encoding="utf-8") as file:
            file.write(payload)
            file.flush()


def load_trajectory(log_path=None):
    """Read all valid entries; blank, malformed or non-object lines are
    skipped rather than breaking the dashboard."""

    log_path = Path(log_path) if log_path else get_log_path()
    trajectory = []

    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(entry, dict):
                    trajectory.append(entry)
    except FileNotFoundError:
        return []

    return trajectory


def log_step(step, action, details):
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "episode_id": str(uuid.uuid4()),
        "step": step,
        "action": action,
        "details": details
    }

    _append_entries([log_entry])

    return log_entry


class TrajectoryLogger:

    def __init__(self, mode=None, log_path=None):
        self.episode_id = str(uuid.uuid4())
        self.mode = mode
        self.log_path = log_path
        self.entries = []

    def log(self, step, details):

        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "episode_id": self.episode_id,
            "step": step,
            "details": details
        }

        if self.mode:
            log_entry["mode"] = self.mode

        self.entries.append(log_entry)

    def save(self):
        """Write buffered entries. Returns False instead of raising if
        the log cannot be written (e.g. read-only disk), so a logging
        problem never breaks an agent run."""

        if not self.entries:
            return True

        try:
            _append_entries(self.entries, self.log_path)
        except OSError:
            return False

        self.entries = []
        return True
