"""Live AI Mode error handling, tested offline with a fake client."""

import json

import openai
import pytest

import agent
from conftest import REQUEST, status_error


def test_success_response_is_parsed(fake_ai):

    fake_ai.content = json.dumps({
        "status": "success",
        "name": "Team Meeting",
        "date": "2026-09-10",
        "time": "10:00"
    })

    result = agent.extract_event_details("Schedule a Team Meeting")

    assert result["name"] == "Team Meeting"
    assert fake_ai.calls == 1


def test_code_fenced_and_wrapped_json_is_parsed(fake_ai):

    fake_ai.content = (
        'Here you go:\n```json\n{"status": "success", "name": "A", '
        '"date": "2026-09-10", "time": "10:00"}\n```'
    )

    assert agent.extract_event_details("x")["name"] == "A"


def test_event_name_is_normalised_to_name(fake_ai):

    fake_ai.content = json.dumps({
        "status": "success",
        "event_name": "Team Meeting",
        "date": "2026-09-10",
        "time": "10:00"
    })

    result = agent.extract_event_details("x")

    assert result["name"] == "Team Meeting"
    assert "event_name" not in result


@pytest.mark.parametrize("content", ["not json at all", "{broken", "[1, 2]"])
def test_invalid_json_raises_invalid_response(fake_ai, content):

    fake_ai.content = content

    with pytest.raises(agent.AgentInvalidResponseError):
        agent.extract_event_details("x")


@pytest.mark.parametrize("content", ["", None, "   "])
def test_empty_response_raises_invalid_response(fake_ai, content):

    fake_ai.content = content

    with pytest.raises(agent.AgentInvalidResponseError):
        agent.extract_event_details("x")


@pytest.mark.parametrize(
    "error, expected",
    [
        (status_error(openai.AuthenticationError, 401),
         agent.AgentAuthenticationError),
        (status_error(openai.PermissionDeniedError, 403),
         agent.AgentAuthenticationError),
        (openai.APITimeoutError(request=REQUEST),
         agent.AgentTimeoutError),
        (openai.APIConnectionError(request=REQUEST),
         agent.AgentConnectionError),
        (status_error(openai.RateLimitError, 429),
         agent.AgentRateLimitError),
        (status_error(openai.NotFoundError, 404),
         agent.AgentModelUnavailableError),
        (status_error(openai.BadRequestError, 400,
                      "x is not a valid model ID"),
         agent.AgentModelUnavailableError),
        (status_error(openai.InternalServerError, 503),
         agent.AgentModelUnavailableError),
        (status_error(openai.APIStatusError, 402),
         agent.AgentAPIError),
        (status_error(openai.BadRequestError, 400, "bad input"),
         agent.AgentAPIError),
    ],
    ids=[
        "auth-401", "auth-403", "timeout", "connection", "rate-limit",
        "model-404", "model-invalid-400", "provider-503", "credits-402",
        "bad-request",
    ],
)
def test_api_errors_are_mapped(fake_ai, error, expected):

    fake_ai.error = error

    with pytest.raises(expected):
        agent.extract_event_details("x")


def test_rate_limit_message_includes_reset_time(fake_ai):

    fake_ai.error = status_error(
        openai.RateLimitError, 429,
        headers={"x-ratelimit-reset": "1788000000000"}
    )

    with pytest.raises(agent.AgentRateLimitError, match="resets at"):
        agent.extract_event_details("x")


def test_live_mode_without_credentials(no_credentials):

    assert agent.is_live_mode_configured() is False

    with pytest.raises(agent.AgentConfigurationError):
        agent.extract_event_details("Schedule a Team Meeting")


def test_api_key_is_redacted_from_error_messages(fake_ai, monkeypatch):

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-secretvalue")

    fake_ai.error = status_error(
        openai.APIStatusError, 418, "echo sk-or-v1-secretvalue"
    )

    with pytest.raises(agent.AgentAPIError) as raised:
        agent.extract_event_details("x")

    assert "sk-or-v1-secretvalue" not in str(raised.value)
    assert "[REDACTED]" in str(raised.value)


def test_key_with_whitespace_is_stripped(monkeypatch):

    monkeypatch.setenv("OPENROUTER_API_KEY", "  sk-test-key\n")

    assert agent.get_api_key() == "sk-test-key"


def test_model_can_be_overridden(monkeypatch):

    assert agent.get_model_name() == agent.DEFAULT_MODEL_NAME

    monkeypatch.setenv("OPENROUTER_MODEL", "some/other-model:free")

    assert agent.get_model_name() == "some/other-model:free"
