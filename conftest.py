"""Shared test setup.

Every test runs isolated from real data and real services:
- trajectory logs go to a temporary file, never logs/trajectory.jsonl;
- the OpenRouter key is replaced by a dummy value;
- creating a real OpenAI client is blocked, so no test can reach the
  network or consume API credits. Tests that exercise Live AI Mode use
  the `fake_ai` fixture instead.
"""

from types import SimpleNamespace

import httpx
import pytest

import agent


@pytest.fixture(autouse=True)
def isolated_environment(tmp_path, monkeypatch):

    monkeypatch.setenv(
        "AGENTEVAL_LOG_PATH",
        str(tmp_path / "logs" / "trajectory.jsonl")
    )
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-dummy-key")
    monkeypatch.delenv("OPENROUTER_MODEL", raising=False)

    def no_network(*args, **kwargs):
        raise AssertionError(
            "Tests must not create a real OpenRouter client; "
            "use the fake_ai fixture."
        )

    monkeypatch.setattr(agent, "OpenAI", no_network)
    agent._client_cache.clear()


@pytest.fixture
def log_path(tmp_path):
    return tmp_path / "logs" / "trajectory.jsonl"


@pytest.fixture
def no_credentials(monkeypatch):
    """No API key in the environment or in Streamlit secrets."""

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(agent.st, "secrets", {})


class FakeCompletions:

    def __init__(self):
        self.content = None
        self.error = None
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        if self.error is not None:
            raise self.error
        message = SimpleNamespace(content=self.content)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


@pytest.fixture
def fake_ai(monkeypatch):
    """Replace the OpenRouter client with an offline fake. Set
    `.content` to the model's reply, or `.error` to an exception."""

    completions = FakeCompletions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    monkeypatch.setattr(agent, "_get_client", lambda: client)
    return completions


REQUEST = httpx.Request(
    "POST", "https://openrouter.ai/api/v1/chat/completions"
)


def status_error(error_class, status_code, message="error", headers=None):
    """Build an OpenAI SDK HTTP error without making a request."""

    response = httpx.Response(
        status_code, request=REQUEST, headers=headers or {}
    )
    return error_class(message, response=response, body=None)
