import streamlit as st
import json

from run_episode import run_single_episode


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AgentEval Lab",
    page_icon="🤖",
    layout="wide"
)


# =========================================================
# HEADER
# =========================================================

st.title("🤖 AgentEval Lab")
st.caption("AI-Powered Calendar Agent Evaluation")

st.divider()


# =========================================================
# USER TASK INPUT
# =========================================================

st.subheader("📝 Enter User Task")

user_task = st.text_area(
    "User Task",
    placeholder="Example: Schedule a Team Meeting on 2026-09-10 at 10:00",
    height=100
)

st.divider()


# =========================================================
# LOAD TRAJECTORY LOGS
# =========================================================

def load_trajectory_logs():

    trajectory = []

    try:

        with open(
            "logs/trajectory.jsonl",
            "r"
        ) as file:

            for line in file:

                line = line.strip()

                if not line:
                    continue

                try:

                    entry = json.loads(line)

                    if isinstance(entry, dict):
                        trajectory.append(entry)

                except json.JSONDecodeError:

                    continue

    except FileNotFoundError:

        return []

    return trajectory


# =========================================================
# RUN AGENT
# =========================================================

if st.button(
    "▶ Run AI Agent",
    type="primary",
    use_container_width=True
):

    if not user_task.strip():

        st.warning(
            "⚠️ Please enter a user task."
        )

    else:

        try:

            with st.spinner(
                "🤖 AI agent is processing the task..."
            ):

                result = run_single_episode(
                    user_task
                )


            # =================================================
            # HANDLE CLARIFICATION
            # =================================================

            if result["status"] == "needs_clarification":

                st.warning(
                    "⚠️ More information is required."
                )

                details = result["extracted_details"]

                st.subheader(
                    "🧠 AI Agent"
                )

                st.info(
                    details.get(
                        "message",
                        "Please provide more information."
                    )
                )

                missing = details.get(
                    "missing",
                    []
                )

                if missing:

                    st.write(
                        "**Missing information:**"
                    )

                    for item in missing:

                        st.write(
                            f"• {item.title()}"
                        )

                st.divider()

                st.subheader(
                    "🔄 Agent Workflow"
                )

                st.write(
                    """
                    **User Task**
                    ↓
                    **OpenRouter AI Agent**
                    ↓
                    **Missing Information Detected**
                    ↓
                    **Clarification Required**
                    """
                )

                st.success(
                    "✅ Calendar Tool was not executed."
                )

                st.caption(
                    "The agent correctly avoided creating "
                    "an incomplete calendar event."
                )


            # =================================================
            # HANDLE SUCCESS
            # =================================================

            else:

                st.success(
                    "✅ AI agent execution completed!"
                )

                st.divider()


                # =================================================
                # AI EXTRACTED DETAILS
                # =================================================

                st.subheader(
                    "🧠 AI Agent — Extracted Arguments"
                )

                details = result[
                    "extracted_details"
                ]

                col1, col2, col3 = st.columns(3)

                with col1:

                    st.metric(
                        "Event",
                        details["name"]
                    )

                with col2:

                    st.metric(
                        "Date",
                        details["date"]
                    )

                with col3:

                    st.metric(
                        "Time",
                        details["time"]
                    )

                st.json(details)

                st.divider()


                # =================================================
                # CALENDAR ACTION
                # =================================================

                st.subheader(
                    "📅 Calendar Tool"
                )

                calendar_result = result[
                    "calendar_result"
                ]

                st.json(
                    calendar_result
                )

                if calendar_result.get(
                    "status"
                ) == "created":

                    st.success(
                        "✅ Calendar event created successfully."
                    )

                else:

                    st.error(
                        "❌ Calendar event creation failed."
                    )

                st.divider()


                # =================================================
                # EVALUATION
                # =================================================

                st.subheader(
                    "🎯 Agent Evaluation"
                )

                score = result[
                    "score"
                ]

                col1, col2, col3, col4 = st.columns(4)

                with col1:

                    st.metric(
                        "Overall Score",
                        f"{score['score']}/{score['max_score']}"
                    )

                with col2:

                    st.metric(
                        "Event Name",
                        "PASS"
                        if score["name_correct"]
                        else "FAIL"
                    )

                with col3:

                    st.metric(
                        "Date",
                        "PASS"
                        if score["date_correct"]
                        else "FAIL"
                    )

                with col4:

                    st.metric(
                        "Time",
                        "PASS"
                        if score["time_correct"]
                        else "FAIL"
                    )


                if (
                    score["score"]
                    == score["max_score"]
                ):

                    st.success(
                        f"🎉 Evaluation Passed — "
                        f"{score['score']}/"
                        f"{score['max_score']}"
                    )

                else:

                    st.warning(
                        f"⚠️ Evaluation Score — "
                        f"{score['score']}/"
                        f"{score['max_score']}"
                    )

                st.divider()


                # =================================================
                # AGENT WORKFLOW
                # =================================================

                st.subheader(
                    "🔄 Agent Workflow"
                )

                st.write(
                    """
                    **User Task**
                    ↓
                    **OpenRouter AI Agent**
                    ↓
                    **Structured Arguments**
                    ↓
                    **Calendar Tool**
                    ↓
                    **Trajectory Logging**
                    ↓
                    **Evaluation**
                    """
                )

                st.divider()


            # =====================================================
            # TRAJECTORY LOG
            # =====================================================

            st.subheader(
                "📊 Trajectory Log"
            )

            trajectory = load_trajectory_logs()

            if trajectory:

                for entry in reversed(
                    trajectory
                ):

                    step = entry.get(
                        "step",
                        "Unknown"
                    )

                    timestamp = entry.get(
                        "timestamp",
                        "Unknown"
                    )

                    with st.expander(
                        f"🔹 {step} — {timestamp}"
                    ):

                        st.json(entry)

            else:

                st.info(
                    "No trajectory logs available yet."
                )

            st.divider()


            # =====================================================
            # EVALUATION SUMMARY
            # =====================================================

            st.subheader(
                "📈 Evaluation Summary"
            )

            score_entries = [

                entry

                for entry in trajectory

                if isinstance(
                    entry.get("step"),
                    str
                )

                and (
                    entry["step"].startswith(
                        "streamlit_"
                    )

                    or entry["step"]
                    == "ai_agent_score"
                )

                and isinstance(
                    entry.get("details"),
                    dict
                )

                and "score"
                in entry["details"]

                and "max_score"
                in entry["details"]

            ]


            if score_entries:

                total_evaluations = len(
                    score_entries
                )

                passed_evaluations = sum(

                    1

                    for entry in score_entries

                    if entry["details"]["score"]
                    ==
                    entry["details"]["max_score"]

                )

                failed_evaluations = (
                    total_evaluations
                    - passed_evaluations
                )

                total_score = sum(

                    entry["details"]["score"]

                    for entry in score_entries

                )

                total_max_score = sum(

                    entry["details"]["max_score"]

                    for entry in score_entries

                )

                success_rate = (

                    passed_evaluations
                    / total_evaluations
                    * 100

                )

                average_score = (

                    total_score
                    / total_max_score
                    * 100

                )


                col1, col2, col3, col4 = st.columns(4)

                with col1:

                    st.metric(
                        "Total Evaluations",
                        total_evaluations
                    )

                with col2:

                    st.metric(
                        "Passed",
                        passed_evaluations
                    )

                with col3:

                    st.metric(
                        "Failed",
                        failed_evaluations
                    )

                with col4:

                    st.metric(
                        "Success Rate",
                        f"{success_rate:.1f}%"
                    )


                st.write(
                    "**Average Evaluation Score**"
                )

                st.progress(
                    min(
                        max(
                            average_score / 100,
                            0
                        ),
                        1
                    )
                )

                st.caption(
                    f"{average_score:.1f}% "
                    f"average score across "
                    f"{total_evaluations} evaluation(s)"
                )


                if failed_evaluations == 0:

                    st.success(
                        "🎉 All evaluations passed successfully."
                    )

                else:

                    st.warning(
                        f"⚠️ {failed_evaluations} "
                        f"evaluation(s) failed."
                    )

            else:

                st.info(
                    "No evaluations available yet."
                )


        except Exception as error:

            st.error(
                "❌ Agent execution failed."
            )

            st.exception(error)