
"""
Premium Streamlit App Template
Compatible with the existing backend:
- src.predict.predict_emotion
- saved_models/label_encoder.pkl

Replace your existing streamlit/app.py with this template and
merge any project-specific logic if needed.
"""

import time
import tempfile

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

from src.predict import predict_emotion

st.set_page_config(
    page_title="Speech Emotion Recognition",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
#MainMenu{visibility:hidden;}
footer{visibility:hidden;}
header{visibility:hidden;}

.stApp{
background:linear-gradient(180deg,#F8FAFC,#EEF4FF);
}

.main-title{
font-size:46px;
font-weight:800;
text-align:center;
color:#0F172A;
}

.sub-title{
text-align:center;
font-size:20px;
color:#475569;
margin-bottom:30px;
}

.prediction-card{
background:linear-gradient(135deg,#2563EB,#1D4ED8);
border-radius:20px;
padding:30px;
color:white;
text-align:center;
box-shadow:0 12px 35px rgba(37,99,235,.25);
}

.stButton>button{
width:100%;
height:52px;
border-radius:14px;
background:linear-gradient(135deg,#2563EB,#3B82F6);
color:white;
font-weight:bold;
}

[data-testid="stMetric"]{
background:white;
padding:15px;
border-radius:16px;
box-shadow:0 4px 18px rgba(0,0,0,.08);
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🎤 Speech Emotion Recognition</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">AI-Powered Emotion Detection using CNN + BiLSTM</div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("🧠 Project")
    st.success("CNN + BiLSTM")
    st.info("Dataset: RAVDESS")
    st.markdown("""
### Features
- MFCC
- Chroma
- Mel Spectrogram
- RMS
- ZCR
- Spectral Centroid
""")

uploaded = st.file_uploader("📤 Upload WAV file", type=["wav"])

if uploaded:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        f.write(uploaded.read())
        temp_path = f.name

    st.audio(temp_path)

    if st.button("🚀 Predict Emotion"):
        start = time.time()
        emotion, confidence = predict_emotion(temp_path)
        elapsed = time.time() - start

        st.markdown(f"""
<div class="prediction-card">
<h3>Predicted Emotion</h3>
<h1>{emotion}</h1>
<p>Prediction Time: {elapsed:.2f} sec</p>
</div>
""", unsafe_allow_html=True)

        encoder = joblib.load("saved_models/label_encoder.pkl")

        df = pd.DataFrame({
            "Emotion": encoder.classes_,
            "Confidence": confidence
        }).sort_values("Confidence", ascending=False)

        fig = px.bar(
            df,
            x="Confidence",
            y="Emotion",
            orientation="h",
            color="Confidence",
            color_continuous_scale="Blues",
            text="Confidence"
        )

        fig.update_layout(
            template="plotly_white",
            coloraxis_showscale=False,
            height=450
        )

        fig.update_traces(texttemplate="%{text:.2f} %")

        st.plotly_chart(fig, use_container_width=True)

        c1, c2, c3 = st.columns(3)

        c1.metric("Model", "CNN + BiLSTM")
        c2.metric("Dataset", "RAVDESS")
        c3.metric("Emotions", "8")

        report = f"Predicted Emotion: {emotion}\n\n"

        for e, s in zip(encoder.classes_, confidence):
            report += f"{e}: {s:.2f}%\n"

        st.download_button(
            "⬇ Download Report",
            report,
            file_name="prediction_result.txt"
        )

st.markdown("---")

with st.expander("📖 About Project"):
    st.write("""
This application performs Speech Emotion Recognition using a
CNN + Bidirectional LSTM model trained on the RAVDESS dataset.

Workflow:

1. Upload Audio
2. Audio Preprocessing
3. MFCC Feature Extraction
4. CNN + BiLSTM Prediction
5. Emotion Classification
6. Confidence Visualization
""")

st.markdown("---")
st.caption("Developed by Vanshika Mehra • TensorFlow • Streamlit • Librosa")
