from logs.trajectory_logger import log_step


def test_log_step():
    result = log_step(
        step=1,
        action="tool_selected",
        details={"tool": "create_event"}
    )

    assert result["step"] == 1
    assert result["action"] == "tool_selected"
    assert result["details"]["tool"] == "create_event"