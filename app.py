
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ---------------- PAGE SETTINGS ----------------
st.set_page_config(
    page_title="Mental Health Score Prediction",
    page_icon="🧠",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.stApp {
    background-color: #eaf4f9;
}

.main .block-container {
    max-width: 1600px;
    padding: 1rem 2rem 2rem 2rem;
}

h1 {
    text-align: center;
    color: #234779;
    font-size: 34px !important;
    font-weight: 750;
}

h2 {
    color: #234779;
}

div[data-testid="stVerticalBlockBorderWrapper"] {
    background: white;
    border: 1px solid #d6e3f5;
    border-radius: 20px;
    padding: 18px;
    box-shadow: 0 5px 15px rgba(30,60,90,0.06);
}

div.stButton > button {
    min-height: 48px;
    border-radius: 12px;
    font-weight: 700;
}

div.stButton > button[kind="primary"] {
    background-color: #3657df;
    color: white;
    border: none;
}

label {
    color: #263e5d !important;
    font-weight: 600 !important;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
MODEL_PATH = Path(__file__).parent / "Mental_Health_Model.pkl"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

try:
    model = load_model()
except Exception as e:
    st.error("Could not load Mental_Health_Model.pkl.")
    st.write("Check that the model file is in the same folder as app.py.")
    st.exception(e)
    st.stop()

# ---------------- DEFAULT VALUES ----------------
DEFAULTS = {
    "age": 21,
    "gender": "Female",
    "academic_level": "Undergraduate",
    "country": "USA",
    "platform": "TikTok",
    "purpose": "Entertainment",
    "study_hours": 4.0,
    "social_hours": 4.0,
    "unlocks": 130,
    "activity": 1.0,
    "sleep": 7.0,
    "stress": "Medium"
}

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value

def clear_all():
    for key, value in DEFAULTS.items():
        st.session_state[key] = value
    st.session_state.pop("prediction_result", None)
    st.session_state.pop("prediction_error", None)

# ---------------- TITLE ----------------
st.markdown(
    "<h1>🧠 Mental Health Score Prediction</h1>",
    unsafe_allow_html=True
)

# ---------------- STUDENT PROFILE ----------------
with st.container(border=True):
    st.markdown("## 👤 Student Profile")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.number_input(
            "Age", min_value=1, max_value=120,
            step=1, key="age"
        )

    with c2:
        st.selectbox(
            "Gender",
            ["Female", "Male", "Other"],
            key="gender"
        )

    with c3:
        st.selectbox(
            "Academic Level",
            ["Undergraduate", "Graduate", "High School"],
            key="academic_level"
        )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.selectbox(
            "Country / Region",
            ["USA", "India", "Canada", "Australia", "UK", "Other"],
            key="country"
        )

    with c2:
        st.selectbox(
            "Most Used Platform",
            [
                "TikTok", "Instagram", "Facebook", "YouTube",
                "WhatsApp", "LinkedIn", "Twitter", "Snapchat", "Other"
            ],
            key="platform"
        )

    with c3:
        st.selectbox(
            "Main Purpose",
            ["Entertainment", "Education", "Networking", "News", "Other"],
            key="purpose"
        )

# ---------------- DAILY HABITS ----------------
with st.container(border=True):
    st.markdown("## 🧮 Daily Habits")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.number_input(
            "Study Hours per Day",
            min_value=0.0, max_value=24.0,
            step=0.5, key="study_hours"
        )

    with c2:
        st.number_input(
            "Social Media Hours per Day",
            min_value=0.0, max_value=24.0,
            step=0.5, key="social_hours"
        )

    with c3:
        st.number_input(
            "Daily Phone Unlocks",
            min_value=0, max_value=10000,
            step=1, key="unlocks"
        )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.number_input(
            "Physical Activity Hours per Day",
            min_value=0.0, max_value=24.0,
            step=0.5, key="activity"
        )

    with c2:
        st.number_input(
            "Sleep Hours per Night",
            min_value=0.0, max_value=24.0,
            step=0.5, key="sleep"
        )

    with c3:
        st.selectbox(
            "Stress Level",
            ["Low", "Medium", "High", "Very High"],
            key="stress"
        )

    b1, b2, _ = st.columns([1, 1, 4])

    with b1:
        predict = st.button(
            "Predict Score",
            type="primary",
            use_container_width=True
        )

    with b2:
        st.button(
            "Clear All Values",
            on_click=clear_all,
            use_container_width=True
        )

# ---------------- PREDICTION ----------------
if predict:
    # These column names must match the training dataset.
    input_data = {
        "Study_Hours": float(st.session_state["study_hours"]),
        "Age": int(st.session_state["age"]),
        "Avg_Daily_Usage_Hours": float(st.session_state["social_hours"]),
        "Daily_Unlocks": int(st.session_state["unlocks"]),
        "Physical_Activity_Hours": float(st.session_state["activity"]),
        "Sleep_Hours_Per_Night": float(st.session_state["sleep"]),
        "Stress_Level": st.session_state["stress"],
        "Gender": st.session_state["gender"],
        "Academic_Level": st.session_state["academic_level"],
        "Most_Used_Platform": st.session_state["platform"],
        "Purpose_Of_Use": st.session_state["purpose"],
        "Grouped_country": st.session_state["country"]
    }

    feature_order = [
        "Study_Hours",
        "Age",
        "Avg_Daily_Usage_Hours",
        "Daily_Unlocks",
        "Physical_Activity_Hours",
        "Sleep_Hours_Per_Night",
        "Stress_Level",
        "Gender",
        "Academic_Level",
        "Most_Used_Platform",
        "Purpose_Of_Use",
        "Grouped_country"
    ]

    try:
        input_df = pd.DataFrame(
            [[input_data[col] for col in feature_order]],
            columns=feature_order
        )

        result = float(model.predict(input_df)[0])

        st.session_state["prediction_result"] = round(result, 2)
        st.session_state.pop("prediction_error", None)

    except Exception as e:
        st.session_state.pop("prediction_result", None)
        st.session_state["prediction_error"] = str(e)

# ---------------- SHOW RESULT ----------------
if "prediction_result" in st.session_state:
    with st.container(border=True):
        st.markdown("## 📊 Prediction Result")
        st.metric(
            "Predicted Mental Health Score",
            st.session_state["prediction_result"]
        )
        st.caption(
            "This is a machine-learning estimate, not a medical diagnosis."
        )

if "prediction_error" in st.session_state:
    st.error("Prediction failed. Check the model's expected input features.")
    st.code(st.session_state["prediction_error"])
