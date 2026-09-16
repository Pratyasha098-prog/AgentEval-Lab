import os
import json
from datetime import datetime, timezone

from dotenv import load_dotenv
from openai import (
    OpenAI,
    AuthenticationError,
    APITimeoutError,
    APIConnectionError,
    RateLimitError,
    OpenAIError,
)
import streamlit as st


# =========================================================
# AGENT-LEVEL ERRORS
# =========================================================
# Wrapping the OpenAI SDK's exceptions in our own types lets
# callers (e.g. app.py) handle each failure mode separately
# without depending on the SDK's exception hierarchy directly.

class AgentAuthenticationError(RuntimeError):
    """OpenRouter rejected the request due to a bad/missing API key."""


class AgentTimeoutError(RuntimeError):
    """The OpenRouter request did not complete within the timeout."""


class AgentConnectionError(RuntimeError):
    """OpenRouter could not be reached over the network."""


class AgentRateLimitError(RuntimeError):
    """OpenRouter's rate limit for the free-tier model was hit."""


class AgentAPIError(RuntimeError):
    """Any other error returned by the OpenRouter API."""


class AgentInvalidResponseError(RuntimeError):
    """OpenRouter returned an empty, malformed, or non-JSON response."""


# =========================================================
# LOAD API KEY
# =========================================================

load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")

if not api_key:
    try:
        api_key = st.secrets.get("OPENROUTER_API_KEY")
    except Exception:
        api_key = None

# Guard against whitespace/newlines that sneak in when a key is
# copy-pasted into a .env file or the Streamlit Secrets editor.
# A key like this is still "truthy" but produces an invalid
# Authorization header, which OpenRouter reports as a 401
# "Missing Authentication header" instead of a clearer error.
if api_key:
    api_key = api_key.strip()

if not api_key:
    raise ValueError(
        "OPENROUTER_API_KEY not found or empty. "
        "Set it in a local .env file, or in Streamlit Cloud "
        "under App settings -> Secrets, then reboot the app."
    )


# =========================================================
# OPENROUTER CLIENT
# =========================================================

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
# It remains the best available option. Note that OpenRouter also
# enforces a separate, account-wide daily cap shared across every
# ":free" model (50 requests/day with no purchased credits) -- no
# choice of free model can raise or bypass that cap; see the
# RateLimitError handling below for how that specific case is
# reported.
MODEL_NAME = "nex-agi/nex-n2.5-mini:free"

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
    timeout=REQUEST_TIMEOUT_SECONDS,
    max_retries=1
)


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


def extract_event_details(user_task):

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
            model=MODEL_NAME,
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
    except AuthenticationError:
        raise AgentAuthenticationError(
            "OpenRouter rejected the request (401 authentication "
            "error). The OPENROUTER_API_KEY is missing, invalid, "
            "or was not picked up by the deployment. In Streamlit "
            "Cloud, check App settings -> Secrets and reboot the "
            "app after saving."
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
    except RateLimitError as error:
        raise AgentRateLimitError(
            "OpenRouter's free-tier daily quota (50 requests/day, "
            "shared across all free models on this account) has "
            "been used up. This is an account-wide limit, not "
            "specific to the current model, so switching models "
            "will not bypass it -- add credits at "
            "openrouter.ai/settings/credits to raise the limit, or "
            "wait for the daily reset."
            + _rate_limit_reset_message(error)
        )
    except OpenAIError as error:
        raise AgentAPIError(
            f"OpenRouter request failed: {error}"
        )

    try:
        content = response.choices[0].message.content.strip()
    except (IndexError, AttributeError) as error:
        raise AgentInvalidResponseError(
            "OpenRouter returned an empty or malformed response. "
            "Please try again."
        ) from error

    if not content:
        raise AgentInvalidResponseError(
            "OpenRouter returned an empty response. Please try again."
        )

    if content.startswith("```"):
        content = content.replace("```json", "")
        content = content.replace("```", "")
        content = content.strip()

    try:
        result = json.loads(content)
    except json.JSONDecodeError as error:
        raise AgentInvalidResponseError(
            "OpenRouter returned a response that was not valid JSON. "
            "Please try again."
        ) from error

    if not isinstance(result, dict):
        raise AgentInvalidResponseError(
            "OpenRouter returned an unexpected response format. "
            "Please try again."
        )

    return result


# =========================================================
# TEST MODE
# =========================================================

if __name__ == "__main__":

    task = input("Enter calendar task: ")

    result = extract_event_details(task)

    print("\nAI Agent Output:")

    print(
        json.dumps(
            result,
            indent=4
        )
    )