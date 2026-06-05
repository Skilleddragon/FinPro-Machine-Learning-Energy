import time
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import os
import gdown
from energy_pipeline import EnergyFeatureEngineer, EnergyPreprocessor

st.set_page_config(page_title='Energy Production Level Prediction', layout='centered')

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

st.title('⚡ Energy Production Level Prediction')
st.write('Prediksi level produksi energi terbarukan: **Rendah**, **Sedang**, atau **Tinggi**.')

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

st.write({
    'Day': day_name,
    'Month': month_name,
    'Day of Year': int(day_of_year),
    'Weekend': bool(is_weekend),
    'Daytime': bool(is_daytime)
})

input_df = pd.DataFrame({
    'Date': [selected_date.strftime('%Y-%m-%d')],
    'Start_Hour': [start_hour],
    'Source': [source]
})

if st.button('Predict'):
    start = time.perf_counter()
    pred = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0]
    latency_ms = (time.perf_counter() - start) * 1000

    classes = model.classes_
    pred_label = label_map[int(pred)]
    confidence = float(np.max(proba))

    st.success(f'Prediksi Level: **{pred_label}**')
    st.metric('Confidence Score', f'{confidence:.2%}')
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
    st.bar_chart(prob_df.set_index('Level'))

    if latency_ms <= 100:
        st.info('Latency memenuhi target < 100 ms pada prediksi ini.')
    else:
        st.warning('Latency prediksi ini > 100 ms. Coba jalankan ulang setelah cache model aktif.')

st.caption('Aplikasi memakai full pipeline: feature engineering, OneHotEncoder, dan Random Forest.')
