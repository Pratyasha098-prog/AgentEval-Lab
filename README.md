# AgentEval-Lab

## LLM Agent Evaluation Framework for Tool-Using AI Agents

AgentEval-Lab is a small, practical evaluation framework for testing an AI calendar agent.

Instead of only checking whether an AI agent produces an answer, the project evaluates whether the agent produced the **correct result** using:

* Ground-truth evaluation
* Automated scoring
* Trajectory logging
* Episode/session tracking
* Edge-case testing
* Tool execution
* Clarification handling
* Streamlit visualization

The project demonstrates how an AI agent can be **tested, evaluated, and monitored**, rather than simply built.

---

## Why This Project?

LLM agents can produce different outputs for the same type of task.

A successful agent therefore needs more than:

> "The model gave an answer."

It needs a way to determine:

> "Was the answer actually correct?"

AgentEval-Lab addresses this problem with a simple evaluation pipeline:

```text
User Task
    ↓
AI Agent
    ↓
Extracted Information
    ↓
Calendar Tool
    ↓
Ground Truth
    ↓
Evaluator
    ↓
Score
    ↓
Trajectory Log
```

---

## What the Agent Does

The agent receives a natural-language calendar request such as:

```text
Schedule a Team Meeting on 2026-09-10 at 10:00
```

It extracts:

```json
{
    "status": "success",
    "name": "Team Meeting",
    "date": "2026-09-10",
    "time": "10:00"
}
```

The extracted information is then passed to the calendar tool.

The evaluator compares the AI output against an independent ground-truth dataset.

---

## Evaluation System

The evaluator checks three important fields:

| Field      | Evaluation          |
| ---------- | ------------------- |
| Event name | Correct / Incorrect |
| Date       | Correct / Incorrect |
| Time       | Correct / Incorrect |

Each correct field receives one point.

### Example

Expected:

```text
Name: Team Meeting
Date: 2026-09-10
Time: 10:00
```

AI output:

```text
Name: Team Meeting
Date: 2026-09-10
Time: 10:00
```

Result:

```text
3 / 3
```

If the AI produces the wrong date:

```text
Name: Team Meeting
Date: 2026-09-11
Time: 10:00
```

Result:

```text
2 / 3
```

This demonstrates that the evaluation framework can detect incorrect agent behavior.

---

## Ground Truth

The project maintains an independent evaluation dataset.

Example:

```python
GROUND_TRUTH = {
    "Schedule a Team Meeting on 2026-09-10 at 10:00": {
        "name": "Team Meeting",
        "date": "2026-09-10",
        "time": "10:00"
    }
}
```

The important design principle is:

```text
Ground Truth ≠ AI Output
```

The evaluator should never treat the model's own output as the expected answer.

---

## Trajectory Logging

Each agent execution receives a unique `episode_id`.

Example:

```text
episode_id:
5af1c5db-9a94-47a5-83ae-4e97477d9d70
```

The same episode ID is attached to the major steps of the execution:

```text
AI Agent
    ↓
Calendar Tool
    ↓
Evaluation
```

Example trajectory:

```json
{
    "episode_id": "5af1c5db-9a94-47a5-83ae-4e97477d9d70",
    "step": "ai_agent"
}
```

```json
{
    "episode_id": "5af1c5db-9a94-47a5-83ae-4e97477d9d70",
    "step": "calendar_tool"
}
```

```json
{
    "episode_id": "5af1c5db-9a94-47a5-83ae-4e97477d9d70",
    "step": "ai_agent_score"
}
```

This makes it possible to trace an individual agent episode from input through evaluation.

---

## Edge-Case Testing

The project includes automated tests for incorrect agent outputs and agent behavior.

Current evaluator tests include:

* Wrong event name
* Wrong date
* Wrong time
* All arguments wrong
* Correct extraction
* Missing date and time
* Relative date handling

For example:

```text
Correct result → 3/3
Wrong name     → 2/3
Wrong date     → 2/3
Wrong time     → 2/3
All wrong      → 0/3
```

---

## Test Results

Current automated test result:

```text
11 passed
```

The test suite covers:

```text
evaluators/test_edge_cases.py
evaluators/test_evaluator.py
tests/test_calendar.py
tests/test_logger.py
tests/test_schema.py
```

The project currently passes:

**11/11 tests**

---

## Clarification Handling

The agent is designed not to guess missing information.

For example:

```text
Schedule a team meeting
```

does not contain enough information to create a calendar event.

Instead, the agent returns:

```json
{
    "status": "needs_clarification"
}
```

This demonstrates a basic safety principle:

> Do not invent missing information when required task parameters are unavailable.

---

## Technology Stack

* Python
* OpenRouter
* LLM API
* FastAPI
* Streamlit
* Pytest
* JSON / JSONL
* UUID-based episode tracking
* Calendar tool
* Ground-truth evaluation

---

## Project Structure

```text
AgentEval-Lab/
│
├── agent.py
│
├── run_episode.py
│
├── evaluation_dataset.py
│
├── evaluators/
│   ├── evaluator.py
│   ├── test_evaluator.py
│   └── test_edge_cases.py
│
├── tools/
│   └── calendar_tool.py
│
├── logs/
│   ├── trajectory.jsonl
│   └── trajectory_logger.py
│
├── tests/
│   ├── test_calendar.py
│   ├── test_logger.py
│   └── test_schema.py
│
└── README.md
```

---

## Example End-to-End Run

Input:

```text
Schedule a Team Meeting on 2026-09-10 at 10:00
```

AI output:

```json
{
    "status": "success",
    "name": "Team Meeting",
    "date": "2026-09-10",
    "time": "10:00"
}
```

Calendar result:

```text
status: created
```

Evaluation:

```text
name_correct: True
date_correct: True
time_correct: True

score: 3
max_score: 3
```

---

## Running the Project

### 1. Activate the environment

Use the Python environment where the project dependencies are installed.

### 2. Run the automated tests

```powershell
python -m pytest
```

Expected result:

```text
11 passed
```

### 3. Run an agent episode

```powershell
python run_episode.py
```

Example:

```text
Enter calendar task:
Schedule a Team Meeting on 2026-09-10 at 10:00
```

---

## Streamlit Interface

The project also includes a Streamlit interface for interacting with the agent and viewing evaluation results.

The interface is intended to make the evaluation workflow easier to demonstrate without requiring users to interact directly with Python code.

---

## Key Engineering Concepts Demonstrated

This project demonstrates practical concepts relevant to modern AI engineering and agent evaluation:

### Agentic AI

An LLM receives a natural-language task, extracts structured information, and interacts with a tool.

### Tool Calling

The extracted parameters are passed to a calendar tool.

### LLM Evaluation

The AI output is compared against an independent expected result.

### Ground Truth

Expected results are defined separately from model output.

### Trajectory Logging

Agent steps are recorded for later inspection.

### Episode Tracking

Each execution receives a unique identifier.

### Automated Testing

Pytest is used to verify agent evaluation behavior and supporting components.

### Edge-Case Testing

The framework intentionally tests incorrect outputs and incomplete requests.

---

## Future Improvements

Possible future improvements include:

* More comprehensive evaluation datasets
* Additional scoring metrics
* Better trajectory visualization
* Failure analysis
* Evaluation history
* Larger collections of agent tasks
* More tool-calling scenarios
* Automated evaluation reports
* Deployment of the Streamlit application
* GitHub Actions for automated testing

---

## Project Goal

The goal of AgentEval-Lab is not simply to build an AI agent.

It is to demonstrate the complete loop:

```text
Build
  ↓
Run
  ↓
Observe
  ↓
Evaluate
  ↓
Detect Failure
  ↓
Improve
```

This reflects an important principle in AI engineering:

> **An AI agent is only useful when we can measure how well it performs.**
