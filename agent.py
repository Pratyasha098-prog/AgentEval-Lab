import os
import json
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI


# =========================================================
# LOAD ENVIRONMENT
# =========================================================

load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")

if not api_key:
    raise ValueError(
        "OPENROUTER_API_KEY not found. "
        "Please add it to your .env file."
    )


# =========================================================
# OPENROUTER CLIENT
# =========================================================

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key
)


# =========================================================
# AI AGENT
# =========================================================

def extract_event_details(user_task):

    # Current date is provided to the AI so that
    # words like "today", "tomorrow" and "next Monday"
    # can be interpreted correctly.

    current_date = datetime.now().strftime("%Y-%m-%d")

    prompt = f"""
You are an intelligent calendar task extraction agent.

Today's date is:
{current_date}

Your job is to extract calendar information from the
user's natural-language request.

Extract:

1. Event name
2. Date
3. Time

IMPORTANT RULES:

- Convert relative dates such as "tomorrow", "today",
  "next Monday", etc. into an exact date.
- Use today's date ({current_date}) as the reference.
- Convert time into 24-hour HH:MM format.
- If the user does not provide enough information,
  return an error instead of guessing.
- Do not invent missing information.
- Return ONLY valid JSON.

If all required information is available, return:

{{
    "status": "success",
    "name": "event name",
    "date": "YYYY-MM-DD",
    "time": "HH:MM"
}}

If information is missing or ambiguous, return:

{{
    "status": "needs_clarification",
    "missing": ["date", "time"],
    "message": "Please provide the missing information."
}}

User request:
{user_task}
"""

    response = client.chat.completions.create(
        model="nvidia/nemotron-3.5-lightning:free",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a reliable calendar assistant. "
                    "Never invent missing information."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0
    )

    content = response.choices[0].message.content.strip()

    # Remove markdown code fences if returned by the model
    if content.startswith("```"):
        content = content.replace("```json", "")
        content = content.replace("```", "")
        content = content.strip()

    result = json.loads(content)

    return result


# =========================================================
# TEST MODE
# =========================================================

if __name__ == "__main__":

    task = input(
        "Enter calendar task: "
    )

    result = extract_event_details(task)

    print("\nAI Agent Output:")

    print(
        json.dumps(
            result,
            indent=4
        )
    )