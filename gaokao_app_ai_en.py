
import os
import math
import json
import csv
import time
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from datetime import datetime

# ---------------------------
# Page Setup
# ---------------------------
st.set_page_config(page_title="Gaokao Math Capability Profiler (Demo)", layout="wide")
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False

st.title("Gaokao Math Capability Profiler — Research Demo (English UI)")
st.caption("Upload data → score six capabilities → visualize → chat with AI tutor → collect feedback and adapt.")

# ---------------------------
# Helper: Optional LLM
# ---------------------------
def call_llm(system_prompt: str, user_prompt: str) -> str:
    """
    Tries to use OpenAI if OPENAI_API_KEY is present. Otherwise falls back to a rule-based response.
    The function is intentionally simple so this script remains self-contained for research demo.
    """
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if api_key:
        try:
            import openai
            openai.api_key = api_key
            # Chat Completions (compatible with most OpenAI Python SDK versions)
            resp = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
                max_tokens=600
            )
            return resp.choices[0].message["content"].strip()
        except Exception as e:
            return f"(LLM error: {e})\nI will provide a lightweight heuristic answer instead.\n" + rule_based_reply(user_prompt)
    else:
        return rule_based_reply(user_prompt)

def rule_based_reply(prompt: str) -> str:
    """Minimal heuristic tutor if no API key is available."""
    p = prompt.lower()
    bullets = []
    if "logic" in p or "reason" in p:
        bullets.append("Your logic score reflects performance on harder/new-type items. Focus on multi-step proofs, parameter problems, and function monotonicity tasks.")
    if "time" in p or "speed" in p:
        bullets.append("Practice under timed conditions. Target ~60s (easy), 90s (medium), 120s (hard). Stabilize pacing across mixed sets.")
    if "probability" in p or "statistic" in p:
        bullets.append("Prioritize classical probability models, distribution properties, and expectation/variance drills from past Gaokao papers.")
    if "vector" in p or "analytic" in p or "geometry" in p:
        bullets.append("Revisit vector operations and line/conic parameterization; practice locus/range problems with parameter constraints.")
    if "calculus" in p or "derivative" in p or "limit" in p:
        bullets.append("Strengthen derivative-based optimization and inequality proofs; practice limit evaluation and L'Hospital when applicable.")
    if "expression" in p or "write" in p or "solution" in p:
        bullets.append("Use clear step labels, standard notation, and minimal leaps. Ensure key transitions are justified (definition, lemma, conclusion).")
    if not bullets:
        bullets.append("Start from your lowest capability dimension. Drill 10–15 curated items per day and review errors within 24 hours. Stretch with 1–2 new-type items.")
    return "Tutor tips:\n- " + "\n- ".join(bullets)

# ---------------------------
# Upload Section (center)
# ---------------------------
st.header("Upload Data")
required_cols = ["student_id","question_id","subject","topic","correct","time_spent_sec","attempts","question_level","is_new_type"]
optional_cols = ["essay_len"]
all_cols = required_cols + optional_cols

# Template for convenience
template_df = pd.DataFrame({
    "student_id": ["S001","S001","S002"],
    "question_id": ["Q0001","Q0002","Q0001"],
    "subject": ["MATH","MATH","MATH"],
    "topic": ["Functions","Probability","Geometry"],
    "correct": [1,0,1],
    "time_spent_sec": [60,95,80],
    "attempts": [1,2,1],
    "question_level": [1,3,2],
    "is_new_type": [0,1,0],
    "essay_len": [0,0,0],
})
st.download_button(
    "Download CSV Template",
    data=template_df.to_csv(index=False).encode("utf-8"),
    file_name="student_data_template.csv",
    mime="text/csv"
)

uploaded = st.file_uploader("Drop a CSV/XLSX here or click to select", type=["csv","xlsx"], accept_multiple_files=False)

df = None
if uploaded is not None:
    try:
        if uploaded.name.lower().endswith(".csv"):
            df = pd.read_csv(uploaded)
            st.success(f"Loaded CSV: {uploaded.name}")
        else:
            df = pd.read_excel(uploaded)
            st.success(f"Loaded Excel: {uploaded.name}")
    except Exception as e:
        st.error(f"Failed to read file: {e}")
        st.stop()
else:
    # Use local demo file if exists
    demo_path = "student_data.csv"
    if os.path.exists(demo_path):
        df = pd.read_csv(demo_path)
        st.info("Loaded demo data: student_data.csv (upload your own to replace)")
    else:
        st.info("Please upload a dataset first.")
        st.stop()

# Preview + validation
st.subheader("Preview & Validation")
missing = [c for c in required_cols if c not in df.columns]
extra = [c for c in df.columns if c not in all_cols]

c1, c2 = st.columns([3,2], gap="large")
with c1:
    st.dataframe(df.head(20), use_container_width=True)
with c2:
    if missing:
        st.error(f"Missing required columns: {missing}")
    else:
        st.success("All required columns present.")
    if extra:
        st.warning(f"Detected extra columns (kept as-is): {extra}")
    st.caption("Required: " + ", ".join(required_cols) + " | Optional: " + ", ".join(optional_cols))

if missing:
    st.stop()

# Keep math only
if "subject" in df.columns:
    df["subject"] = df["subject"].astype(str).str.upper()
    math_df = df[df["subject"] == "MATH"].copy()
else:
    math_df = df.copy()

if math_df.empty:
    st.error("No MATH records found.")
    st.stop()

# Coerce numerics and clip difficulty
for col in ["correct","time_spent_sec","attempts","question_level","is_new_type"]:
    if col in math_df.columns:
        math_df[col] = pd.to_numeric(math_df[col], errors="coerce")
math_df["question_level"] = math_df["question_level"].clip(lower=1, upper=5)

# ---------------------------
# Sidebar Weights
# ---------------------------
st.sidebar.header("Weights (%)")
def wslider(label, default):
    return st.sidebar.slider(label, 0, 100, default, 1)

W = {
    "Knowledge": wslider("Knowledge", 25),
    "Logic": wslider("Logic", 20),
    "Strategy": wslider("Strategy", 15),
    "Expression": wslider("Expression", 15),
    "SelfControl": wslider("SelfControl", 15),
    "Emotion": wslider("Emotion", 10),
}
W_sum = sum(W.values())
st.sidebar.caption(f"Total weight: {W_sum} (we normalize when computing overall)")

# ---------------------------
# Student Selection
# ---------------------------
students = sorted(math_df["student_id"].dropna().astype(str).unique())
sid = st.selectbox("Select a student", students, index=0)
sdf = math_df[math_df["student_id"].astype(str) == sid].copy()
if sdf.empty:
    st.error("Selected student has no records.")
    st.stop()

# ---------------------------
# Scoring functions
# ---------------------------
def score_knowledge(g):
    mask = g["question_level"].isin([1,2])
    score = g.loc[mask, "correct"].mean()*100 if mask.any() else g["correct"].mean()*100
    return 0.0 if pd.isna(score) else float(score)

def score_logic(g):
    mask = g["question_level"]>=3
    score = g.loc[mask, "correct"].mean()*100 if mask.any() else g["correct"].mean()*100
    return 0.0 if pd.isna(score) else float(score)

def score_strategy(g):
    mask = g["is_new_type"]==1
    if mask.any():
        score = g.loc[mask, "correct"].mean()*100
    else:
        score = score_logic(g)
    return 0.0 if pd.isna(score) else float(score)

def score_expression(g):
    total_correct = int((g["correct"]==1).sum())
    if total_correct == 0:
        return 0.0
    one_try = int(((g["correct"]==1) & (g["attempts"]==1)).sum())
    return float(one_try/total_correct*100)

def score_time(g):
    if len(g)==0:
        return 0.0
    baseline = {1:60, 2:90, 3:120, 4:135, 5:150}
    ideal = sum(baseline.get(int(x), 90) for x in g["question_level"])
    actual = g["time_spent_sec"].replace(0, np.nan).sum()
    if actual<=0:
        return 100.0 if ideal>0 else 0.0
    score = min(100.0, max(0.0, ideal/actual*100.0))
    return float(score)

def score_emotion(g):
    if len(g)==0:
        return 0.0
    order = g.copy()
    if "question_id" in order.columns:
        try:
            order["ord"] = order["question_id"].astype(str).str.extract(r"(\d+)").astype(float)
            order = order.sort_values("ord")
        except:
            order = order.reset_index(drop=True)
    else:
        order = order.reset_index(drop=True)
    longest = cur = 0
    for c in order["correct"]:
        if int(c)==0:
            cur += 1
            longest = max(longest, cur)
        else:
            cur = 0
    if longest <= 1:
        return 100.0
    return float(max(0.0, 100.0 - (longest-1)*20.0))

scores = {
    "Knowledge": score_knowledge(sdf),
    "Logic": score_logic(sdf),
    "Strategy": score_strategy(sdf),
    "Expression": score_expression(sdf),
    "SelfControl": score_time(sdf),
    "Emotion": score_emotion(sdf),
}
scores_int = {k:int(round(v)) for k,v in scores.items()}
overall = (sum(scores[k]*W[k] for k in scores) / (W_sum if W_sum>0 else 1.0)) if W_sum>0 else 0.0
overall_int = int(round(overall))

# ---------------------------
# Display
# ---------------------------
left, right = st.columns([2,1])
with left:
    st.subheader(f"Scores for {sid}")
    st.write(pd.Series({**scores_int, "Overall": overall_int}).to_frame("Score"))
# Radar
dims = list(scores_int.keys())
vals = list(scores_int.values())
angles = [n/float(len(dims))*2*math.pi for n in range(len(dims))]
angles += angles[:1]; radar_vals = vals + vals[:1]
fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111, polar=True)
ax.plot(angles, radar_vals, linewidth=2)
ax.fill(angles, radar_vals, alpha=0.25)
ax.set_xticks(angles[:-1]); ax.set_xticklabels(dims)
ax.set_ylim(0,100); ax.set_title("Capability Radar", pad=14)
with right:
    st.pyplot(fig, use_container_width=True)

st.markdown("---")

# ---------------------------
# Feedback & Lightweight Adaptation
# ---------------------------
st.subheader("Feedback")
c1, c2 = st.columns(2)
fb = None
with c1:
    if st.button("✅ Looks correct"):
        fb = "Yes"
        st.success("Logged: fits the student.")
with c2:
    if st.button("❌ Not accurate"):
        fb = "No"
        st.warning("Logged: needs adjustment.")
reason = st.text_input("If not accurate, briefly say what's off (e.g., 'underestimates Logic').")

if fb is not None:
    log = "feedback_history_en.csv"
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row = {
        "time": ts,
        "student_id": sid,
        "fit": fb,
        "reason": reason,
        **{f"W_{k}": v for k,v in W.items()}
    }
    write_header = not os.path.exists(log)
    with open(log, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    st.success("Feedback & current weights saved to feedback_history_en.csv")

# micro adaptation: parse reason keywords to nudge weights (in-memory only)
if reason:
    r = reason.lower()
    nudges = {"knowledge":0,"logic":0,"strategy":0,"expression":0,"selfcontrol":0,"emotion":0}
    for k in nudges:
        if k in r:
            nudges[k] += 2  # tiny nudge (2%)
    if any(v != 0 for v in nudges.values()):
        for k in scores.keys():
            key = k.lower()
            W[k] = min(100, max(0, W[k] + nudges.get(key, 0)))
        st.info(f"Applied slight weight nudges from feedback: {nudges}")

# ---------------------------
# AI Tutor Panel
# ---------------------------
st.markdown("---")
st.subheader("AI Tutor (optional)")
st.caption("Ask the tutor to explain scores, plan study sessions, or generate a practice focus aligned with Gaokao distributions.")
default_q = "Explain my weakest dimensions and give me a 7-day practice plan aligned with Gaokao trends."
user_q = st.text_area("Your question", value=default_q, height=120)
if st.button("Ask AI"):
    sys = (
        "You are an expert Gaokao math tutor. "
        "Explain capability scores (Knowledge, Logic, Strategy, Expression, SelfControl, Emotion) in clear English, "
        "and produce actionable plans linked to typical Gaokao question types over the last decade."
    )
    # Provide the model with structured context
    ctx = {
        "student_id": sid,
        "scores": scores_int,
        "overall": overall_int,
        "weights": W,
        "data_columns": list(df.columns),
    }
    prompt = f"Context: {json.dumps(ctx, ensure_ascii=False)}\n\nUser: {user_q}"
    reply = call_llm(sys, prompt)
    st.markdown("**AI Reply:**")
    st.write(reply)
    # persist chat for traceability
    chatlog = "ai_chat_history_en.jsonl"
    with open(chatlog, "a", encoding="utf-8") as f:
        f.write(json.dumps({"time": datetime.now().isoformat(), "user": user_q, "reply": reply, "ctx": ctx}) + "\n")
    st.caption("Chat saved to ai_chat_history_en.jsonl")

