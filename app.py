import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title='Energy Production Level Prediction', layout='centered')

st.markdown("""
<style>
.main {
    padding-top: 1rem;
}

.pred-card {
    background-color: #262730;
    padding: 1rem;
    border-radius: 12px;
    margin-bottom: 1rem;
}

.metric-card {
    text-align:center;
    padding: 0.5rem;
    border-radius: 10px;
    background-color: rgba(70,130,180,0.15);
}

.big-title {
    text-align:center;
    font-size:2.3rem;
    font-weight:700;
}

.subtitle {
    text-align:center;
    color:gray;
    margin-bottom:2rem;
}

.stButton > button {
    height: 3.2em;
    font-size: 18px;
    font-weight: bold;
    border-radius: 12px;
}

.stMetric {
    border-radius: 12px;
    padding: 10px;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return joblib.load("energy_level_pipeline.pkl")

@st.cache_resource
def load_metadata():
    return joblib.load("model_metadata.pkl")

model = load_model()
metadata = load_metadata()

label_map = {
    0:'Low',
    1:'Medium',
    2:'High'
}

source_values = metadata['source_values']

min_date = pd.to_datetime(metadata['min_date']).date()
max_date = pd.to_datetime(metadata['max_date']).date()

def season_from_month(month: int) -> str:
    if month in [12, 1, 2]:
        return 'Winter'
    if month in [3, 4, 5]:
        return 'Spring'
    if month in [6, 7, 8]:
        return 'Summer'
    return 'Fall'

def create_features(date, start_hour, source):

    date = pd.to_datetime(date)

    features = pd.DataFrame({
        'Hour_Sin': [
            np.sin(2*np.pi*start_hour/24)
        ],

        'Hour_Cos': [
            np.cos(2*np.pi*start_hour/24)
        ],

        'Day_Of_Year_Sin': [
            np.sin(2*np.pi*date.dayofyear/365)
        ],

        'Day_Of_Year_Cos': [
            np.cos(2*np.pi*date.dayofyear/365)
        ],

        'Month_Number': [
            date.month
        ],

        'Day_Of_Week': [
            date.dayofweek
        ],

        'Season': [
            season_from_month(date.month)
        ],

        'Source': [
            source
        ]
    })

    return features

st.markdown("""
<div class='big-title'>
⚡ Energy Production Level Prediction
</div>

<div class='subtitle'>
Renewable Energy Production Classification using Machine Learning
</div>
""", unsafe_allow_html=True)

colA, colB, colC = st.columns(3)

colA.metric(
    "Accuracy",
    f"{metadata['accuracy']:.1%}"
)

colB.metric(
    "Macro F1",
    f"{metadata['macro_f1']:.1%}"
)

colC.metric(
    "Energy Sources",
    len(metadata['source_values'])
)

with st.sidebar:
    st.header("ℹ️ About This App")

    st.write(
        """
        This application predicts renewable energy production levels
        based on temporal information and energy source type.

        Users can select:
        - Energy Source
        - Date
        - Start Hour

        The model then classifies the expected production level into:
        - Low
        - Medium
        - High
        """
    )

    st.caption(
        """
        Educational machine learning project using
        chronological validation on renewable energy data.
        """
    )

    st.markdown("---")

    st.header('Model Info')

    st.write(f"Model: {metadata['model_name']}")
    st.write(f"Test Accuracy: {metadata['accuracy']:.2%}")
    st.write(f"Macro F1: {metadata['macro_f1']:.2%}")
    st.write(f"Validation: {metadata['chronological_split']}")

    st.caption(
        "Scores obtained using chronological split "
        "(train 2021–2024, test 2025)."
    )

st.subheader('User Input')
source = st.selectbox('Energy Source', source_values)
selected_date = st.date_input('Date', value=max_date, min_value=min_date, max_value=max_date)
start_hour = st.slider('Start Hour', min_value=0, max_value=23, value=12, step=1)

selected_timestamp = pd.to_datetime(selected_date)
day_name = selected_timestamp.day_name()
month_name = selected_timestamp.month_name()
day_of_year = selected_timestamp.dayofyear
season = season_from_month(selected_timestamp.month)

with st.expander("📊 View Derived Features"):
    col1, col2, col3 = st.columns(3)

    col1.metric('Day',day_name)
    col2.metric('Month',month_name)
    col3.metric('Season',season)

    st.markdown("### Temporal Feature Summary")

    c1, c2 = st.columns(2)
    c1.info(f"🎯 Day of Year: {day_of_year}")
    c2.info(f"⏰ Start Hour: {start_hour}")

input_df = create_features(
    selected_date,
    start_hour,
    source
)

predict_btn = st.button(
    "⚡ Predict Energy Production Level",
    use_container_width=True
)

if predict_btn:
    with st.spinner("Running prediction..."):
        pred = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0]

    classes = model.classes_
    pred_label = label_map[int(pred)]
    confidence = float(np.max(proba))

    st.markdown("---")
    st.subheader("Prediction Result")

    if pred_label == "High":
        emoji = "🟢"
    elif pred_label == "Medium":
        emoji = "🟡"
    else:
        emoji = "🔴"

    st.markdown(
        f"""
        ## {emoji} Predicted Production Level
        # {pred_label}
        """
    )

    st.metric(
        "Confidence",
        f"{confidence:.2%}"
    )

    st.progress(confidence)

    prob_df = pd.DataFrame({
        'Level': [label_map[int(cls)] for cls in classes],
        'Probability': proba
    })

    st.markdown('### Class Probability')
    chart_df = prob_df.set_index("Level")
    st.bar_chart(
        chart_df,
        use_container_width=True
    )

    if confidence > 0.8:
        st.success(
            "Model shows strong confidence for this prediction."
        )
    elif confidence > 0.6:
        st.info(
            "Model confidence is moderate."
        )
    else:
        st.warning(
            "Prediction confidence is relatively low."
        )

with st.sidebar:
    st.markdown("---")
    st.markdown("""
    ### Dataset Coverage

    Available:
    - Solar
    - Wind

    Prediction Task:
    - Hourly Production Level Classification

    Target Classes:
    - Low
    - Medium
    - High
    """)

    st.caption(
    """
    Built with Streamlit • Scikit-Learn • Logistic Regression

    Pipeline includes:
    - Temporal Feature Engineering
    - Numerical Feature Scaling
    - Categorical Feature Encoding
    - Logistic Regression Classification
    """
    )
