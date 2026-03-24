from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data.generator import generate_dataset
from detection.scorer import compute_fps
from detection.signals import score_all_signals
from detection.tripwires import (
    check_submission_velocity,
    check_word_count_trap,
    evaluate_tripwires,
)
from rl.agent import QLearningAgent
from rl.environment import ExpertVerificationEnv
from rl.trainer import train

st.set_page_config(page_title="Expert Verification Engine", page_icon="EVE", layout="wide")

QUESTIONS = [
    {
        "text": "Describe your experience with distributed data pipelines. Answer in 100 words.",
        "limit": 100,
    },
    {
        "text": "Tell me about a production failure you caused and how you resolved it. Answer in 120 words.",
        "limit": 120,
    },
    {
        "text": "How do you decide between Kafka and a simpler queue for event streaming? Answer in 80 words.",
        "limit": 80,
    },
    {
        "text": "Describe a time when you had to significantly refactor existing code. Answer in 100 words.",
        "limit": 100,
    },
    {
        "text": "What is your experience with Kubernetes in a real production environment? Answer in 90 words.",
        "limit": 90,
    },
]

SIGNAL_LABELS = {
    "S1_hedging": "Hedging Language",
    "S2_failure_absence": "Lack of Failure Narratives",
    "S3_temporal_vagueness": "Temporal Vagueness",
    "S4_tribal_vocab": "Low Insider Vocabulary",
    "S5_tool_polarity": "Neutral Tool Opinions",
    "S6_length_uniformity": "Length Uniformity",
    "S7_structural_symmetry": "Structural Symmetry",
    "S8_depth_breadth": "Wide but Shallow Coverage",
}

BAND_EXPLANATION = {
    "CLEAN": "Low fraud risk. Signals look mostly human-authored.",
    "BORDERLINE": "Mixed indicators. A follow-up question is recommended.",
    "HIGH_SUSPICION": "Several risk indicators are elevated. Manual review advised.",
    "FRAUD": "Strong fraud pattern. Flag for reviewer decision.",
}


def init_state() -> None:
    if "answers" not in st.session_state:
        st.session_state.answers = [""] * len(QUESTIONS)
    if "durations" not in st.session_state:
        st.session_state.durations = [0.0] * len(QUESTIONS)
    if "question_index" not in st.session_state:
        st.session_state.question_index = 0
    if "question_start" not in st.session_state:
        st.session_state.question_start = time.time()
    if "completed" not in st.session_state:
        st.session_state.completed = False
    if "results" not in st.session_state:
        st.session_state.results = None


def _risk_color(band: str) -> str:
    return {
        "CLEAN": "#2A9D8F",
        "BORDERLINE": "#E9C46A",
        "HIGH_SUSPICION": "#F4A261",
        "FRAUD": "#E76F51",
    }.get(band, "#7A7A7A")


def _build_result() -> dict:
    answers = st.session_state.answers
    durations = st.session_state.durations

    t1_results = []
    t2_results = []
    for idx, answer in enumerate(answers):
        t1_results.append(check_word_count_trap(answer, QUESTIONS[idx]["limit"]))
        t2_results.append(check_submission_velocity(answer, durations[idx]))

    tripwire = evaluate_tripwires(t1_results, t2_results)
    signals = score_all_signals(answers, domain="data_engineering")
    fps = compute_fps(signals, tripwire["tripwire_result"])

    return {
        "answers": answers,
        "durations": durations,
        "t1_results": t1_results,
        "t2_results": t2_results,
        "tripwire": tripwire,
        "signals": signals,
        "fps": fps,
    }


def _question_diagnostics(result: dict) -> pd.DataFrame:
    rows = []
    for i in range(len(QUESTIONS)):
        t1 = result["t1_results"][i]
        t2 = result["t2_results"][i]
        rows.append(
            {
                "Question": i + 1,
                "Word Count": t1["word_count"],
                "Limit": QUESTIONS[i]["limit"],
                "Delta": t1["delta"],
                "Time (s)": round(result["durations"][i], 2),
                "Sec/Word": round(t2["seconds_per_word"], 2),
                "T1 Near Limit": "Yes" if t1["fired"] else "No",
                "T2 Very Fast": "Yes" if t2["fired"] else "No",
            }
        )
    return pd.DataFrame(rows)


def _screening_analytics_chart(result: dict) -> go.Figure:
    df = _question_diagnostics(result)
    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=df["Question"],
            y=df["Word Count"],
            name="Word Count",
            marker_color="#457B9D",
            yaxis="y",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["Question"],
            y=df["Time (s)"],
            name="Time (s)",
            mode="lines+markers",
            line=dict(color="#E76F51", width=3),
            yaxis="y2",
        )
    )

    fig.update_layout(
        title="Screening Analytics: Word Count vs Time Per Question",
        xaxis=dict(title="Question Number", dtick=1),
        yaxis=dict(title="Word Count"),
        yaxis2=dict(title="Time (seconds)", overlaying="y", side="right"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        height=380,
    )
    return fig


def _gauge_chart(fps: float) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=fps,
            number={"valueformat": ".2f"},
            gauge={
                "axis": {"range": [0, 1]},
                "bar": {"color": "#264653"},
                "steps": [
                    {"range": [0, 0.35], "color": "#95D5B2"},
                    {"range": [0.35, 0.60], "color": "#E9D8A6"},
                    {"range": [0.60, 0.80], "color": "#F4A261"},
                    {"range": [0.80, 1.00], "color": "#E76F51"},
                ],
                "threshold": {"line": {"color": "#1D3557", "width": 4}, "value": fps},
            },
            title={"text": "Fraud Probability Score"},
        )
    )
    fig.update_layout(height=330, margin=dict(l=10, r=10, t=50, b=10))
    return fig


def _signal_chart(signals: dict[str, float]) -> go.Figure:
    df = pd.DataFrame(
        {
            "Signal": [SIGNAL_LABELS.get(k, k) for k in signals.keys()],
            "Score": list(signals.values()),
        }
    ).sort_values("Score", ascending=True)

    fig = px.bar(
        df,
        x="Score",
        y="Signal",
        orientation="h",
        color="Score",
        color_continuous_scale=["#2A9D8F", "#E9C46A", "#E76F51"],
        range_color=(0, 1),
    )
    fig.update_layout(height=420, xaxis=dict(range=[0, 1]), coloraxis_showscale=False)
    return fig


def run_screening_tab() -> None:
    st.subheader("Candidate Screening")
    init_state()

    st.caption("Collect responses only. Detailed timing and word analytics are shown in Results.")

    if st.session_state.completed:
        st.success("Screening complete. Open the Results tab for interpretation.")
        if st.button("Start New Screening"):
            for key in ["answers", "durations", "question_index", "question_start", "completed", "results"]:
                st.session_state.pop(key, None)
            st.rerun()
        return

    idx = st.session_state.question_index
    q = QUESTIONS[idx]

    st.progress((idx + 1) / len(QUESTIONS), text=f"Question {idx + 1} of {len(QUESTIONS)}")
    st.markdown(f"### {q['text']}")

    answer = st.text_area("Candidate answer", height=220, key=f"answer_{idx}")

    if st.button("Submit Answer", type="primary"):
        elapsed = max(time.time() - st.session_state.question_start, 0.1)
        st.session_state.answers[idx] = answer
        st.session_state.durations[idx] = elapsed

        if idx < len(QUESTIONS) - 1:
            st.session_state.question_index += 1
            st.session_state.question_start = time.time()
        else:
            st.session_state.completed = True
            st.session_state.results = _build_result()
        st.rerun()


def run_results_tab() -> None:
    st.subheader("Results Dashboard")
    result = st.session_state.get("results")

    if not result:
        st.info("Complete screening first to generate results.")
        return

    fps = result["fps"]
    tripwire = result["tripwire"]
    signals = result["signals"]

    color = _risk_color(fps["band"])
    st.markdown(
        (
            f"<div style='padding:12px 14px;border-radius:8px;background:{color};color:white;'>"
            f"<b>Verdict: {fps['band']}</b><br>{BAND_EXPLANATION[fps['band']]}"
            "</div>"
        ),
        unsafe_allow_html=True,
    )

    with st.expander("How to read this dashboard", expanded=True):
        st.write("Fraud Probability Score is a weighted score from 0 to 1.")
        st.write("Tripwire checks are behavioral clues: strict word-limit obedience and unusually fast submission speed.")
        st.write("Signal scores are linguistic clues. Higher value means more fraud-like behavior.")

    col_gauge, col_summary = st.columns([1.2, 1.0])
    with col_gauge:
        st.plotly_chart(_gauge_chart(fps["fps"]), use_container_width=True)

    with col_summary:
        st.metric("Fraud Probability Score", f"{fps['fps']:.2f}")
        st.metric("Recommended Action", fps["recommended_action"])
        st.metric("Tripwire Result", tripwire["tripwire_result"])
        st.metric("Tripwire Multiplier", f"x{tripwire['multiplier']:.1f}")

    st.markdown("### Signal Breakdown")
    st.plotly_chart(_signal_chart(signals), use_container_width=True)

    contrib_df = pd.DataFrame(
        {
            "Signal": [SIGNAL_LABELS.get(k, k) for k in fps["contributions"].keys()],
            "Weighted Contribution": [round(v, 4) for v in fps["contributions"].values()],
        }
    )
    st.markdown("### Top Drivers")
    st.dataframe(contrib_df, use_container_width=True, hide_index=True)

    st.markdown("### Screening Analytics")
    st.plotly_chart(_screening_analytics_chart(result), use_container_width=True)

    st.markdown("### Per-Question Diagnostics")
    st.dataframe(_question_diagnostics(result), use_container_width=True, hide_index=True)


def run_training_tab() -> None:
    st.subheader("RL Training Monitor")
    st.caption("Train a policy for PASS / FLAG / FOLLOW_UP using synthetic labeled candidates.")

    episodes = st.slider("Episodes", min_value=200, max_value=5000, value=1200, step=200)
    n_fraud = st.slider("Fraud examples", min_value=100, max_value=1000, value=500, step=100)
    n_genuine = st.slider("Genuine examples", min_value=100, max_value=1000, value=500, step=100)

    if st.button("Train Agent", type="primary"):
        with st.spinner("Training in progress..."):
            dataset = generate_dataset(n_fraud=n_fraud, n_genuine=n_genuine)
            env = ExpertVerificationEnv(dataset)
            agent = QLearningAgent()
            metrics = train(agent, env, n_episodes=episodes, verbose_every=max(episodes // 6, 1))

        st.success(f"Training complete. Model saved to {Path('models/q_table.pkl').resolve()}")

        rewards_df = pd.DataFrame(
            {
                "episode": list(range(1, len(metrics["episode_rewards"]) + 1)),
                "reward": metrics["episode_rewards"],
            }
        )
        perf_df = pd.DataFrame(
            {
                "episode": list(range(1, len(metrics["fraud_caught"]) + 1)),
                "fraud_caught": metrics["fraud_caught"],
                "false_positives": metrics["false_positives"],
            }
        )

        reward_fig = go.Figure(data=[go.Scatter(x=rewards_df["episode"], y=rewards_df["reward"], mode="lines")])
        reward_fig.update_layout(title="Episode Reward Trend", height=320)

        perf_fig = go.Figure()
        perf_fig.add_trace(go.Scatter(x=perf_df["episode"], y=perf_df["fraud_caught"], mode="lines", name="Fraud Caught"))
        perf_fig.add_trace(go.Scatter(x=perf_df["episode"], y=perf_df["false_positives"], mode="lines", name="False Positives"))
        perf_fig.update_layout(title="Detection vs False Positives", height=320)

        st.plotly_chart(reward_fig, use_container_width=True)
        st.plotly_chart(perf_fig, use_container_width=True)


def main() -> None:
    tab1, tab2, tab3 = st.tabs(["Screening", "Results", "RL Training"])
    with tab1:
        run_screening_tab()
    with tab2:
        run_results_tab()
    with tab3:
        run_training_tab()


if __name__ == "__main__":
    main()
