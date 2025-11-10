import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import openai

st.title("Gaokao Math Capabilities")

# 1. File upload (CSV or XLSX):contentReference[oaicite:0]{index=0}
uploaded_file = st.file_uploader("Upload Gaokao math data", type=["csv","xlsx"])
if uploaded_file is not None:
    # Read the file into a DataFrame (CSV or Excel)
    file_name = uploaded_file.name.lower()
    try:
        if file_name.endswith(".xlsx") or file_name.endswith(".xls"):
            df = pd.read_excel(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Failed to read file: {e}")
        st.stop()

    # 2. Filter for subject 'MATH'
    df = df[df['subject'].str.upper() == 'MATH']

    # 3. Student selection
    student_ids = sorted(df['student_id'].unique().tolist())
    selected_student = st.selectbox("Select student ID", student_ids)

    if selected_student:
        student_df = df[df['student_id'] == selected_student]
        
        # 4. Compute capability scores
        # Knowledge: average correctness on level 1-2 questions
        know_df = student_df[student_df['question_level'].isin([1,2])]
        knowledge = know_df['correct'].mean() if len(know_df) > 0 else 0

        # Logic: average correctness on level 3-5 questions
        logic_df = student_df[student_df['question_level'].isin([3,4,5])]
        logic = logic_df['correct'].mean() if len(logic_df) > 0 else 0

        # Strategy: average correctness on is_new_type==1 questions (or fallback to logic)
        strat_df = student_df[student_df['is_new_type'] == 1]
        if len(strat_df) > 0:
            strategy = strat_df['correct'].mean()
        else:
            strategy = logic

        # Expression: ratio of correct questions done in one attempt
        total_correct = student_df[student_df['correct'] == 1].shape[0]
        one_attempt_correct = student_df[(student_df['correct'] == 1) & (student_df['attempts'] == 1)].shape[0]
        expression = one_attempt_correct / total_correct if total_correct > 0 else 0

        # SelfControl: compare actual time vs expected time per level
        # (Assume higher levels allow more time. These are example thresholds.)
        expected_time = {1: 60, 2: 90, 3: 120, 4: 150, 5: 180}  # seconds
        time_scores = []
        for _, row in student_df.iterrows():
            level = row['question_level']
            actual = row['time_spent_sec']
            exp = expected_time.get(level, 120)
            # Ratio capped at 1 (taking longer than expected lowers score)
            ratio = min(exp / actual, 1.0) if actual > 0 else 0
            time_scores.append(ratio)
        self_control = np.mean(time_scores) if time_scores else 0

        # Emotion: penalize long streaks of incorrect answers
        # We compute 1 - (max_streak_of_incorrect / total_questions)
        longest_streak = 0
        current_streak = 0
        for correct in student_df['correct']:
            if correct == 0:
                current_streak += 1
                longest_streak = max(longest_streak, current_streak)
            else:
                current_streak = 0
        n = len(student_df)
        emotion = 1 - longest_streak/n if n > 0 else 1

        # 5. Display scores in a table
        scores = {
            "Knowledge": [knowledge],
            "Logic": [logic],
            "Strategy": [strategy],
            "Expression": [expression],
            "SelfControl": [self_control],
            "Emotion": [emotion],
        }
        scores_df = pd.DataFrame(scores, index=[selected_student])
        scores_df.loc['Overall'] = scores_df.mean()  # overall average of the six scores
        scores_df = scores_df.transpose().reset_index()
        scores_df.columns = ["Dimension", "Score"]

        st.subheader("Capability Scores")
        st.table(scores_df)

        # 6. Radar chart of the six dimensions:contentReference[oaicite:1]{index=1}
        fig_df = pd.DataFrame({
            'score': [knowledge, logic, strategy, expression, self_control, emotion],
            'dimension': ["Knowledge", "Logic", "Strategy", "Expression", "SelfControl", "Emotion"]
        })
        fig = px.line_polar(fig_df, r='score', theta='dimension', line_close=True)
        fig.update_traces(fill='toself')
        # Limit radial axis to [0,1] since scores are ratios
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,1])), showlegend=False)
        st.plotly_chart(fig)

# 7. Simple AI Chat Interface
st.markdown("---")
st.header("AI Tutor Chat")
user_msg = st.chat_input("Type a message...")
if user_msg:
    st.chat_message("user").write(user_msg)
    # If API key is set, use OpenAI ChatCompletion; otherwise, fallback rules
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        openai.api_key = api_key  # set the API key from environment:contentReference[oaicite:2]{index=2}
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": user_msg}],
                temperature=0.7
            )
            answer = response.choices[0].message.content
        except Exception as e:
            answer = f"Error calling OpenAI API: {e}"
    else:
        # Rule-based fallback: simple tips based on keywords
        txt = user_msg.lower()
        if "logic" in txt:
            answer = "Tip: Break down problems into smaller steps and use diagrams to visualize the logic."
        elif "knowledge" in txt:
            answer = "Tip: Review fundamental concepts and practice basic problems to strengthen your knowledge base."
        elif "strategy" in txt:
            answer = "Tip: For new or unusual problem types, try to identify similarities with what you know and tackle them step by step."
        elif "expression" in txt or "attempt" in txt:
            answer = "Tip: Try to solve problems carefully on the first attempt to save time for later problems."
        elif "selfcontrol" in txt or "time" in txt:
            answer = "Tip: Keep an eye on the clock and practice pacing yourself to avoid spending too long on any one problem."
        elif "emotion" in txt or "stress" in txt:
            answer = "Tip: Take a deep breath and stay calm. If you get stuck, move on and come back later with a clear mind."
        else:
            answer = "Sorry, I can only give general study tips on Logic, Knowledge, Strategy, Expression, SelfControl, or Emotion."
    st.chat_message("assistant").write(answer)

