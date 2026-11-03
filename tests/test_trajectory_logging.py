"""Trajectory logging: isolation, robustness, and concurrency."""

import json
import threading

from logs import trajectory_logger
from logs.trajectory_logger import (
    TrajectoryLogger,
    get_log_path,
    load_trajectory,
)


def test_tests_never_use_the_real_log(log_path):

    assert get_log_path() == log_path
    assert get_log_path() != trajectory_logger.DEFAULT_LOG_PATH


def test_default_log_path_is_project_root_relative(monkeypatch):

    monkeypatch.delenv("AGENTEVAL_LOG_PATH")

    path = get_log_path()

    assert path.is_absolute()
    assert path == trajectory_logger.PROJECT_ROOT / "logs" / "trajectory.jsonl"


def test_missing_log_directory_is_created(log_path):

    assert not log_path.parent.exists()

    logger = TrajectoryLogger(mode="demo")
    logger.log("ai_agent", {"x": 1})

    assert logger.save() is True
    assert load_trajectory(log_path)[0]["mode"] == "demo"


def test_episode_entries_share_an_id_and_append(log_path):

    log_path.parent.mkdir(parents=True)
    log_path.write_text(json.dumps({"step": "existing"}) + "\n", encoding="utf-8")

    logger = TrajectoryLogger()
    logger.log("ai_agent", {})
    logger.log("calendar_tool", {})
    logger.save()
    logger.save()  # a second save must not duplicate entries

    entries = load_trajectory(log_path)

    assert [e["step"] for e in entries] == ["existing", "ai_agent", "calendar_tool"]
    assert entries[1]["episode_id"] == entries[2]["episode_id"]


def test_malformed_lines_are_skipped(log_path):

    log_path.parent.mkdir(parents=True)
    log_path.write_text(
        '{"step": "good"}\n'
        "not json\n"
        "\n"
        '{"step": "trunc\n'
        "[1, 2, 3]\n"
        '{"step": "also good"}\n',
        encoding="utf-8"
    )

    assert [e["step"] for e in load_trajectory(log_path)] == ["good", "also good"]


def test_missing_log_file_returns_empty_list(log_path):

    assert load_trajectory(log_path) == []


def test_unwritable_log_does_not_raise(tmp_path):

    blocker = tmp_path / "not_a_dir"
    blocker.write_text("file, not a directory")

    logger = TrajectoryLogger(log_path=blocker / "trajectory.jsonl")
    logger.log("ai_agent", {})

    assert logger.save() is False


def test_concurrent_writes_do_not_corrupt_the_log(log_path):

    def worker(n):
        for i in range(20):
            logger = TrajectoryLogger()
            logger.log("step", {"worker": n, "i": i, "pad": "x" * 2000})
            logger.log("step", {"worker": n, "i": i})
            logger.save()

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    lines = log_path.read_text(encoding="utf-8").splitlines()

    assert len(lines) == 8 * 20 * 2
    assert all(json.loads(line)["step"] == "step" for line in lines)
