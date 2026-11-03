
import os
import json
from datetime import datetime, timezone

import streamlit as st
from dotenv import load_dotenv
from openai import (
    OpenAI,
    AuthenticationError,
    APITimeoutError,
    APIConnectionError,
    RateLimitError,
    OpenAIError,
)


# =========================================================
# AGENT-LEVEL ERRORS
# =========================================================

class AgentAuthenticationError(RuntimeError):
    """OpenRouter rejected the API key."""


class AgentTimeoutError(RuntimeError):
    """OpenRouter request timed out."""


class AgentConnectionError(RuntimeError):
    """Could not connect to OpenRouter."""


class AgentRateLimitError(RuntimeError):
    """OpenRouter rate limit reached."""


class AgentAPIError(RuntimeError):
    """Other OpenRouter API error."""


class AgentInvalidResponseError(RuntimeError):
    """OpenRouter returned an invalid response."""


# =========================================================
# LOAD API KEY
# =========================================================

load_dotenv()

api_key = None
key_source = "not found"

# Prioritize Streamlit Cloud Secrets.
try:
    secret_key = st.secrets.get("OPENROUTER_API_KEY")

    if secret_key and str(secret_key).strip():
        api_key = str(secret_key).strip()
        key_source = "Streamlit Secrets"

except Exception:
    pass

# Fall back to environment variables / local .env.
if not api_key:
    env_key = os.getenv("OPENROUTER_API_KEY")

    if env_key and env_key.strip():
        api_key = env_key.strip()
        key_source = "Environment Variable"

if not api_key:
    raise ValueError(
        "OPENROUTER_API_KEY is missing. "
        "Add it to Streamlit Cloud Secrets or your local .env file."
    )

# Display source only. Never display the API key.
if os.getenv("STREAMLIT_SHARING_MODE") or hasattr(st, "secrets"):
    st.caption(f"API key source: {key_source}")


# =========================================================
# OPENROUTER CLIENT
# =========================================================

REQUEST_TIMEOUT_SECONDS = 30

MODEL_NAME = "nex-agi/nex-n2.5-mini:free"

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
    timeout=REQUEST_TIMEOUT_SECONDS,
    max_retries=1,
    default_headers={
        "HTTP-Referer": "https://agenteval-lab-apueqkh5ta9vzfj3uqbddu.streamlit.app",
        "X-Title": "AgentEval Lab",
    },
)


# =========================================================
# RATE LIMIT RESET
# =========================================================

def _rate_limit_reset_message(error):

    try:
        reset_ms = error.response.headers.get(
            "x-ratelimit-reset"
        )

        if not reset_ms:
            return ""

        reset_time = datetime.fromtimestamp(
            int(reset_ms) / 1000,
            tz=timezone.utc
        )

        return (
            " It resets at "
            f"{reset_time.strftime('%Y-%m-%d %H:%M UTC')}."
        )

    except Exception:
        return ""


# =========================================================
# AI AGENT
# =========================================================

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
  and next Monday into exact dates.
- Use today's date as the reference.
- Convert time into 24-hour HH:MM format.
- Never guess missing information.
- If information is missing, return needs_clarification.
- Return ONLY valid JSON.
- Do not use Markdown or code fences.

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
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0,
        )

    except AuthenticationError as error:

        raise AgentAuthenticationError(
            "OpenRouter returned HTTP 401. "
            "Check that the new API key is active and that "
            "Streamlit Secrets contains the correct key."
        ) from error

    except APITimeoutError as error:

        raise AgentTimeoutError(
            f"OpenRouter did not respond within "
            f"{REQUEST_TIMEOUT_SECONDS} seconds."
        ) from error

    except APIConnectionError as error:

        raise AgentConnectionError(
            "Could not connect to OpenRouter. "
            "Please try again."
        ) from error

    except RateLimitError as error:

        raise AgentRateLimitError(
            "OpenRouter rate limit reached. "
            "Check your account quota or credits."
            + _rate_limit_reset_message(error)
        ) from error

    except OpenAIError as error:

        raise AgentAPIError(
            f"OpenRouter request failed: {error}"
        ) from error

    # =====================================================
    # VALIDATE RESPONSE
    # =====================================================

    try:

        content = response.choices[0].message.content

        if not content:
            raise ValueError("Empty response")

        content = content.strip()

    except (IndexError, AttributeError, ValueError) as error:

        raise AgentInvalidResponseError(
            "OpenRouter returned an empty or malformed response."
        ) from error

    # Remove Markdown code fences if present.
    if content.startswith("```"):

        content = content.replace("```json", "")
        content = content.replace("```", "")
        content = content.strip()

    try:

        result = json.loads(content)

    except json.JSONDecodeError as error:

        raise AgentInvalidResponseError(
            "OpenRouter returned invalid JSON."
        ) from error

    if not isinstance(result, dict):

        raise AgentInvalidResponseError(
            "OpenRouter returned an unexpected response format."
        )

    return result


# =========================================================
# LOCAL TEST MODE
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
