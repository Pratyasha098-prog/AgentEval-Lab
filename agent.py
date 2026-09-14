import os
import json
from datetime import datetime

from dotenv import load_dotenv
from openai import OpenAI
import streamlit as st


load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")

if not api_key:
    try:
        api_key = st.secrets["OPENROUTER_API_KEY"]
    except Exception:
        api_key = None

if not api_key:
    raise ValueError(
        "OPENROUTER_API_KEY not found. "
        "Please add it to Streamlit Secrets."
    )


client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key
)


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
- If required information is missing, return needs_clarification.
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

    response = client.chat.completions.create(
        model="nvidia/nemotron-3.5-lightning:free",
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

    content = response.choices[0].message.content.strip()

    if content.startswith("```"):
        content = content.replace("```json", "")
        content = content.replace("```", "")
        content = content.strip()

    result = json.loads(content)

    return result


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