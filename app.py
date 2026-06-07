import time
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import os
import gdown
from energy_pipeline import EnergyFeatureEngineer, EnergyPreprocessor

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
</style>
""", unsafe_allow_html=True)

MODEL_FILE = "energy_level_pipeline.pkl"

@st.cache_resource
def load_artifacts():
    if not os.path.exists(MODEL_FILE):
        file_id = "GOOGLE_DRIVE_FILE_ID"
        url = f"https://drive.google.com/uc?id=1Ati7qCvOA9sKA4J1luEM9BPtfRPk_Qvq"
        gdown.download(url, MODEL_FILE, quiet=False)

    model = joblib.load(MODEL_FILE)
    metadata = joblib.load('model_metadata.pkl')
    return model, metadata

model, metadata = load_artifacts()

colA, colB, colC = st.columns(3)

colA.metric(
    "Accuracy",
    f"{metadata['random_split_accuracy']:.1%}"
)

colB.metric(
    "Macro F1",
    f"{metadata['random_split_macro_f1']:.1%}"
)

colC.metric(
    "Energy Sources",
    len(metadata['source_values'])
)

label_map = metadata['label_map']
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

st.markdown("""
<div class='big-title'>
⚡ Energy Production Level Prediction
</div>

<div class='subtitle'>
Renewable Energy Production Classification using Machine Learning
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header('Model Info')
    st.write(f"Model: {metadata['model_name']}")
    st.write(f"Random split accuracy: {metadata['random_split_accuracy']:.2%}")
    st.write(f"Random split macro F1: {metadata['random_split_macro_f1']:.2%}")
    st.write(f"Chronological accuracy: {metadata['chronological_accuracy']:.2%}")
    st.caption('Chronological score adalah catatan limitation untuk data time-series.')

st.subheader('User Input')
source = st.selectbox('Energy Source', source_values)
selected_date = st.date_input('Date', value=max_date, min_value=min_date, max_value=max_date)
start_hour = st.slider('Start Hour', min_value=0, max_value=23, value=12, step=1)

selected_timestamp = pd.to_datetime(selected_date)
end_hour = (start_hour + 1) % 24
day_name = selected_timestamp.day_name()
month_name = selected_timestamp.month_name()
day_of_year = selected_timestamp.dayofyear
season = season_from_month(selected_timestamp.month)
is_weekend = int(day_name in ['Saturday', 'Sunday'])
is_daytime = int(6 <= start_hour <= 18)

st.markdown('### Derived Features')
col1, col2, col3 = st.columns(3)
col1.metric('End Hour', end_hour)
col2.metric('Duration', '1 hour')
col3.metric('Season', season)

st.markdown("### Feature Summary")

c1, c2, c3 = st.columns(3)

c1.info(f"📅 Day: {day_name}")
c2.info(f"🗓 Month: {month_name}")
c3.info(f"🌞 Daytime: {'Yes' if is_daytime else 'No'}")

c4, c5 = st.columns(2)

c4.info(f"🎯 Day of Year: {day_of_year}")
c5.info(f"🏖 Weekend: {'Yes' if is_weekend else 'No'}")

input_df = pd.DataFrame({
    'Date': [selected_date.strftime('%Y-%m-%d')],
    'Start_Hour': [start_hour],
    'Source': [source]
})

predict_btn = st.button(
    "⚡ Predict Energy Production Level",
    use_container_width=True
)

if predict_btn:
    with st.spinner("Running prediction..."):
        start = time.perf_counter()
        pred = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0]
        latency_ms = (time.perf_counter() - start) * 1000

    classes = model.classes_
    pred_label = label_map[int(pred)]
    confidence = float(np.max(proba))

    st.markdown("---")
    st.subheader("Prediction Result")

    st.success(f'Prediksi Level: **{pred_label}**')
    st.metric('Confidence Score', f'{confidence:.2%}')
    st.progress(confidence)
    st.metric('Prediction Latency', f'{latency_ms:.2f} ms')

    prob_df = pd.DataFrame({
        'Level': [label_map[int(cls)] for cls in classes],
        'Probability': proba
    })

    st.markdown('### Class Probability')
    st.dataframe(
        prob_df.assign(Probability=lambda d: d['Probability'].map(lambda x: f'{x:.2%}')),
        use_container_width=True
    )
    chart_df = prob_df.set_index("Level")

    st.bar_chart(
        chart_df,
        use_container_width=True
    )

    if latency_ms <= 100:
        st.info('Latency memenuhi target < 100 ms pada prediksi ini.')
    else:
        st.warning('Latency prediksi ini > 100 ms. Coba jalankan ulang setelah cache model aktif.')

st.markdown("---")

st.caption(
"""
Built with Streamlit • Scikit-Learn • Random Forest

Pipeline includes:
- Feature Engineering
- Missing Value Imputation
- One-Hot Encoding
- Random Forest Classification
"""
)
