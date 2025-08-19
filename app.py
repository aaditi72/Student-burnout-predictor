# app.py 

# --- 1. Import Necessary Libraries ---
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- 2. Load Model Artifacts ---
try:
    model = joblib.load("burnout_xgb_model.pkl")
    scaler = joblib.load("scaler.pkl")
    mapping = joblib.load("risk_mapping.pkl")
    inverse_mapping = {v: k for k, v in mapping.items()}
except FileNotFoundError:
    st.error("Model files not found! Please ensure all .pkl files are in the same directory.")
    st.stop()

# --- 3. Define the Feedback and Advice Logic ---
# This remains our simple rule-based system for actionable feedback.
advice_knowledge_base = {
    "SleepHours": "Action: Aim for a consistent 7-8 hours of sleep per night.",
    "MissedDeadlines": "Action: Use a digital calendar with reminders for due dates.",
    "GPA": "Action: Consider forming a study group or visiting office hours.",
    "Attendance": "Action: Attend all classes or review lecture recordings promptly.",
    "Assignments": "Action: Break down large assignments into smaller, manageable tasks.",
    "MoodScore": "Action: Practice mindfulness or engage in a hobby to manage stress.",
    "SleepQuality": "Action: Improve sleep quality by creating a relaxing bedtime routine.",
    "HealthIssues": "Action: Don't ignore health issues. Visit the campus health center if you're not feeling well.",
    "ProcrastinationIndex": "Action: High procrastination is a key stressor. Use techniques like the Pomodoro method to stay focused."
}

def generate_feedback(student_data):
    reasons, advice = [], []
    if student_data['Attendance'] < 75:
        reasons.append("Low class attendance.")
        advice.append(advice_knowledge_base['Attendance'])
    if student_data['GPA'] < 6.0:
        reasons.append("A low GPA.")
        advice.append(advice_knowledge_base['GPA'])
    if student_data['MissedDeadlines'] > 3:
        reasons.append("A high number of missed deadlines.")
        advice.append(advice_knowledge_base['MissedDeadlines'])
    if student_data['SleepHours'] < 6:
        reasons.append("Insufficient sleep.")
        advice.append(advice_knowledge_base['SleepHours'])
    if student_data['ProcrastinationIndex'] > 0.7:
        reasons.append("A high procrastination index.")
        advice.append(advice_knowledge_base['ProcrastinationIndex'])
    if student_data['MoodScore'] <= 2:
        reasons.append("A consistently low mood score.")
        advice.append(advice_knowledge_base['MoodScore'])

    if not reasons:
        reasons.append("No major risk factors detected based on our rules.")
        advice.append("You are managing your workload well. Keep up the great work!")
        
    return reasons, advice

# --- 4. Build the Streamlit User Interface with Tabs ---
st.set_page_config(page_title="Student Burnout Predictor", layout="wide")

st.title("🎓 Student Burnout Risk Predictor")
st.write("This tool uses a machine learning model to predict a student's burnout risk. Fill in the details across the tabs.")

# Create the tabs for better organization
tab1, tab2, tab3 = st.tabs(["**Academic Factors**", "**Personal Health**", "**Behavioral Traits**"])

# --- Tab 1: Academic Factors ---
with tab1:
    st.header("Academic Performance")
    col1, col2 = st.columns(2)
    with col1:
        gpa = st.slider("GPA (out of 10.0)", 0.0, 10.0, 7.5, 0.1)
        attendance = st.slider("Class Attendance (%)", 0, 100, 85)
    with col2:
        assignments = st.slider("Assignments Completed (%)", 0, 100, 80)
        missed_deadlines = st.slider("Missed Deadlines (per semester)", 0, 10, 2)

# --- Tab 2: Personal Health ---
with tab2:
    st.header("Health and Wellness")
    col1, col2 = st.columns(2)
    with col1:
        sleep_hours = st.slider("Average Sleep Hours (per night)", 0.0, 12.0, 7.0, 0.5)
        sleep_quality = st.slider("Sleep Quality (1=Poor, 5=Excellent)", 1, 5, 3)
    with col2:
        mood_score = st.slider("Average Mood Score (1=Low, 5=High)", 1, 5, 4)
        health_issues = st.radio("Any recurring health issues? (e.g., headaches, fatigue)", (0, 1), format_func=lambda x: "Yes" if x == 1 else "No")

# --- Tab 3: Behavioral Traits ---
with tab3:
    st.header("Habits and Activities")
    col1, col2 = st.columns(2)
    with col1:
        extra_curricular = st.slider("Extra-Curricular Activities (count)", 0, 10, 3)
        self_study_hours = st.slider("Self-Study Hours (per day)", 0.0, 10.0, 3.0, 0.5)
    with col2:
        screen_time = st.slider("Screen Time (hours per day)", 0.0, 15.0, 8.0, 0.5)
        procrastination_index = st.slider("Procrastination Index (0=Low, 1=High)", 0.0, 1.0, 0.4, 0.1)

# Collect all 12 features into the dictionary
student_data = {
    'Attendance': attendance,
    'Assignments': assignments,
    'GPA': gpa,
    'MissedDeadlines': missed_deadlines,
    'ExtraCurricular': extra_curricular,
    'SleepHours': sleep_hours,
    'ScreenTime': screen_time,
    'SelfStudyHours': self_study_hours,
    'MoodScore': mood_score,
    'SleepQuality': sleep_quality,
    'ProcrastinationIndex': procrastination_index,
    'HealthIssues': health_issues
}

st.write("---")

# --- 5. Prediction Logic and Display ---
if st.button("Analyze Burnout Risk", use_container_width=True, type="primary"):
    # Convert input data to a DataFrame in the correct feature order
    student_df = pd.DataFrame([student_data])
    feature_order = scaler.feature_names_in_
    student_df = student_df[feature_order]
    
    # Scale the features
    student_scaled = scaler.transform(student_df)

    # Make prediction
    prediction_encoded = model.predict(student_scaled)
    prediction_label = inverse_mapping[prediction_encoded[0]]
    prediction_proba = model.predict_proba(student_scaled)

    # Generate feedback
    reasons, advice = generate_feedback(student_data)

    # --- Display Results in a Report Format ---
    st.header("Prediction Report")
    
    # Use columns for a dashboard-like feel
    report_col1, report_col2 = st.columns(2)

    with report_col1:
        st.subheader("Overall Risk Assessment")
        if prediction_label == 'High':
            st.error(f"**Predicted Burnout Risk: HIGH**")
        elif prediction_label == 'Medium':
            st.warning(f"**Predicted Burnout Risk: MEDIUM**")
        else:
            st.success(f"**Predicted Burnout Risk: LOW**")
        
        # Display probabilities as a bar chart
        st.write("**Prediction Confidence:**")
        proba_df = pd.DataFrame({
            'Risk Level': ['Low', 'Medium', 'High'],
            'Probability': prediction_proba[0]
        })
        st.bar_chart(proba_df.set_index('Risk Level'))

    with report_col2:
        st.subheader("Key Input Metrics")
        st.metric(label="GPA", value=f"{gpa:.1f} / 10")
        st.metric(label="Sleep Hours", value=f"{sleep_hours:.1f} / night")
        st.metric(label="Missed Deadlines", value=f"{missed_deadlines} / semester")
        st.metric(label="Attendance", value=f"{attendance}%")

    st.write("---")
    
    st.subheader("💡 Key Factors & Personalized Recommendations")
    feedback_col1, feedback_col2 = st.columns(2)
    
    with feedback_col1:
        st.write("**Factors Increasing Risk:**")
        for r in reasons:
            st.write(f"🔹 {r}")
            
    with feedback_col2:
        st.write("**Personalized Advice:**")
        for a in advice:
            st.write(f"🔸 {a}")

st.write("---")
st.info("**Disclaimer:** This is a predictive tool based on a simulated dataset. It is not a substitute for professional medical or psychological advice. If you are feeling overwhelmed, please consult a healthcare professional.")

