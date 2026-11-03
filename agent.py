import os
import json
from datetime import datetime, timedelta, timezone

from dotenv import load_dotenv
from openai import (
    OpenAI,
    AuthenticationError,
    PermissionDeniedError,
    NotFoundError,
    BadRequestError,
    APITimeoutError,
    APIConnectionError,
    RateLimitError,
    InternalServerError,
    APIStatusError,
    OpenAIError,
)
import streamlit as st


# =========================================================
# AGENT-LEVEL ERRORS
# =========================================================
# Wrapping the OpenAI SDK's exceptions in our own types lets
# callers (e.g. app.py) handle each failure mode separately
# without depending on the SDK's exception hierarchy directly.

class AgentError(RuntimeError):
    """Base class for every agent failure reported to the user."""


class AgentConfigurationError(AgentError):
    """No OpenRouter API key is configured, so Live AI Mode cannot run."""


class AgentAuthenticationError(AgentError):
    """OpenRouter rejected the request due to a bad/missing API key."""


class AgentTimeoutError(AgentError):
    """The OpenRouter request did not complete within the timeout."""


class AgentConnectionError(AgentError):
    """OpenRouter could not be reached over the network."""


class AgentRateLimitError(AgentError):
    """OpenRouter's rate limit for the free-tier model was hit."""


class AgentModelUnavailableError(AgentError):
    """The configured model does not exist or has no available provider."""


class AgentAPIError(AgentError):
    """Any other error returned by the OpenRouter API."""


class AgentInvalidResponseError(AgentError):
    """OpenRouter returned an empty, malformed, or non-JSON response."""


# =========================================================
# CONFIGURATION
# =========================================================
# Settings come from environment variables (a local .env file) or,
# on Streamlit Community Cloud, from App settings -> Secrets. They are
# read when a live request is made -- not at import -- so the app
# (and Demo Mode) still loads when no API key is configured.

load_dotenv()


def _read_setting(name):

    value = os.getenv(name)

    if not value:
        try:
            value = st.secrets.get(name)
        except Exception:
            value = None

    # Guard against whitespace/newlines that sneak in when a key is
    # copy-pasted into a .env file or the Streamlit Secrets editor.
    # A key like this is still "truthy" but produces an invalid
    # Authorization header, which OpenRouter reports as a 401
    # "Missing Authentication header" instead of a clearer error.
    if isinstance(value, str):
        value = value.strip()

    return value or None


def get_api_key():
    return _read_setting("OPENROUTER_API_KEY")


def is_live_mode_configured():
    """True if an API key is configured. Never exposes the key itself."""
    return get_api_key() is not None


# =========================================================
# OPENROUTER CLIENT
# =========================================================

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# OpenRouter uses these to attribute free-tier usage to this app;
# they are not secrets and have no effect on authentication.
OPENROUTER_DEFAULT_HEADERS = {
    "HTTP-Referer": "https://agenteval-lab-apueqkh5ta9vzfj3uqbddu.streamlit.app",
    "X-Title": "AgentEval Lab",
}

# Bounds how long a single call can take. Without this, the
# underlying HTTP client's default timeout (several minutes) plus
# automatic retries can leave the Streamlit spinner running for a
# very long time before anything is reported to the user.
REQUEST_TIMEOUT_SECONDS = 30

# Model selection notes (re-verified live against OpenRouter):
# - nvidia/nemotron-3.5-lightning:free took anywhere from ~45s to
#   140s+ (occasionally not responding at all within 120s), which is
#   what originally caused the Streamlit UI to appear stuck.
# - google/gemma-4-31b-it:free returned its own upstream 429
#   ("temporarily rate-limited upstream") on the very first call,
#   independent of OpenRouter's account-wide free quota.
# - nex-agi/nex-n2.5-mini:free consistently responded in ~1-4s across
#   many repeated live calls, with no provider-side throttling seen.
# It remains the default. Note that OpenRouter also enforces a
# separate, account-wide daily cap shared across every ":free" model
# (50 requests/day with no purchased credits) -- no choice of free
# model can raise or bypass that cap; see the RateLimitError handling
# below for how that specific case is reported.
# OPENROUTER_MODEL (env var or Streamlit secret) overrides it.
DEFAULT_MODEL_NAME = "nex-agi/nex-n2.5-mini:free"


def get_model_name():
    return _read_setting("OPENROUTER_MODEL") or DEFAULT_MODEL_NAME


_client_cache = {}


def _get_client():

    api_key = get_api_key()

    if not api_key:
        raise AgentConfigurationError(
            "Live AI Mode needs an OpenRouter API key, and none is "
            "configured. Set OPENROUTER_API_KEY in a local .env file, "
            "or in Streamlit Cloud under App settings -> Secrets, then "
            "restart the app. Demo Mode works without a key."
        )

    if api_key not in _client_cache:
        _client_cache.clear()
        _client_cache[api_key] = OpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=api_key,
            timeout=REQUEST_TIMEOUT_SECONDS,
            max_retries=1,
            default_headers=OPENROUTER_DEFAULT_HEADERS
        )

    return _client_cache[api_key]


def redact_secrets(text):
    """Remove the API key from any text shown to users or logged."""

    text = str(text)
    api_key = get_api_key()

    if api_key and api_key in text:
        text = text.replace(api_key, "[REDACTED]")

    return text


# =========================================================
# AI AGENT
# =========================================================

def _rate_limit_reset_message(error):
    """Best-effort human-readable reset time for a 429, read from
    OpenRouter's X-RateLimit-Reset response header (epoch ms)."""

    try:
        reset_ms = error.response.headers.get("x-ratelimit-reset")

        if not reset_ms:
            return ""

        reset_time = datetime.fromtimestamp(
            int(reset_ms) / 1000,
            tz=timezone.utc
        )

        return (
            f" It resets at "
            f"{reset_time.strftime('%Y-%m-%d %H:%M UTC')}."
        )
    except Exception:
        return ""


def _parse_json_object(content):
    """Parse the model's reply, tolerating code fences or a sentence
    of text around the JSON object."""

    if content.startswith("```"):
        content = content.replace("```json", "")
        content = content.replace("```", "")
        content = content.strip()

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    start = content.find("{")
    end = content.rfind("}")

    if start != -1 and end > start:
        try:
            return json.loads(content[start:end + 1])
        except json.JSONDecodeError:
            pass

    raise AgentInvalidResponseError(
        "OpenRouter returned a response that was not valid JSON. "
        "Please try again."
    )


def extract_event_details(user_task):

    client = _get_client()
    model_name = get_model_name()

    current_date = datetime.now().strftime("%Y-%m-%d")

    prompt = f"""
You are an intelligent calendar task extraction agent.

Today's date is:

{current_date}

Extract:

1. Event name
2. Date
3. Time

IMPORTANT RULES:

- Convert relative dates such as today, tomorrow,
  and next Monday into an exact date.
- Use today's date as the reference.
- Convert time into 24-hour HH:MM format.
- Never guess missing information.
- If required information is missing, return
  needs_clarification.
- Return ONLY valid JSON.
- Do not use Markdown.
- Do not wrap the JSON in code fences.

If all information is available, return:

{{
    "status": "success",
    "name": "event name",
    "date": "YYYY-MM-DD",
    "time": "HH:MM"
}}

If information is missing, return:

{{
    "status": "needs_clarification",
    "missing": ["date", "time"],
    "message": "Please provide the missing information."
}}

User request:

{user_task}
"""

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a reliable calendar assistant. "
                        "Never invent missing information. "
                        "Return only valid JSON."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0
        )
    except (AuthenticationError, PermissionDeniedError):
        raise AgentAuthenticationError(
            "OpenRouter rejected the API key (401/403). The "
            "OPENROUTER_API_KEY is invalid, revoked, or disabled. "
            "Create a new key at openrouter.ai/settings/keys and update "
            "your .env file (or Streamlit Cloud Secrets), then restart "
            "the app."
        )
    except NotFoundError:
        raise AgentModelUnavailableError(
            f"The model '{model_name}' is not available on OpenRouter "
            "(404). Set OPENROUTER_MODEL to an available model."
        )
    except RateLimitError as error:
        raise AgentRateLimitError(
            "OpenRouter's rate limit was reached. On the free tier "
            "this is usually the account-wide daily quota (50 "
            "requests/day, shared across all free models), so "
            "switching models will not bypass it -- add credits at "
            "openrouter.ai/settings/credits, or wait for the reset."
            + _rate_limit_reset_message(error)
        )
    except APITimeoutError:
        raise AgentTimeoutError(
            f"OpenRouter did not respond within "
            f"{REQUEST_TIMEOUT_SECONDS} seconds. This is usually a "
            "temporary network or model-availability issue on "
            "OpenRouter's side. Please try again."
        )
    except APIConnectionError:
        raise AgentConnectionError(
            "Could not reach OpenRouter (network connection error). "
            "Check your internet connection or OpenRouter's status "
            "page, then try again."
        )
    except BadRequestError as error:
        if "model" in str(error).lower():
            raise AgentModelUnavailableError(
                f"OpenRouter did not accept the model '{model_name}'. "
                "Set OPENROUTER_MODEL to a valid model ID."
            )
        raise AgentAPIError(
            f"OpenRouter rejected the request (400): {redact_secrets(error)}"
        )
    except InternalServerError:
        raise AgentModelUnavailableError(
            f"OpenRouter or the provider for '{model_name}' is "
            "temporarily unavailable (5xx). Please try again shortly."
        )
    except APIStatusError as error:
        if error.status_code == 402:
            raise AgentAPIError(
                "OpenRouter reports insufficient credits for this "
                "request (402). Add credits or use a free model."
            )
        raise AgentAPIError(
            f"OpenRouter request failed (HTTP {error.status_code}): "
            f"{redact_secrets(error)}"
        )
    except OpenAIError as error:
        raise AgentAPIError(
            f"OpenRouter request failed: {redact_secrets(error)}"
        )

    try:
        content = response.choices[0].message.content.strip()
    except (IndexError, AttributeError, TypeError) as error:
        raise AgentInvalidResponseError(
            "OpenRouter returned an empty or malformed response. "
            "Please try again."
        ) from error

    if not content:
        raise AgentInvalidResponseError(
            "OpenRouter returned an empty response. Please try again."
        )

    result = _parse_json_object(content)

    if not isinstance(result, dict):
        raise AgentInvalidResponseError(
            "OpenRouter returned an unexpected response format. "
            "Please try again."
        )

    # The rest of the app (calendar tool, evaluator, schema, ground
    # truth) uses "name" for the event name. Models sometimes reply
    # with "event_name" instead, so normalise it to the one schema.
    if "name" not in result and "event_name" in result:
        result["name"] = result.pop("event_name")

    return result


# =========================================================
# TEST MODE
# =========================================================

def evaluate_result(result):

    score = 0

    print("\nEvaluation:")

    if result.get("name") == "Team Meeting":
        print("Event name: PASS")
        score += 1
    else:
        print("Event name: FAIL")

    tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")

    if result.get("date") == tomorrow:
        print("Date: PASS")
        score += 1
    else:
        print("Date: FAIL")

    if result.get("time") == "10:00":
        print("Time: PASS")
        score += 1
    else:
        print("Time: FAIL")

    print(f"\nScore: {score}/3")


if __name__ == "__main__":

    print("=" * 50)
    print("AgentEval Lab - Calendar AI Agent")
    print("=" * 50)

    task = input("\nEnter calendar task: ")

    result = extract_event_details(task)

    print("\nAI Agent Output:")

    print(
        json.dumps(
            result,
            indent=4
        )
    )

    evaluate_result(result)