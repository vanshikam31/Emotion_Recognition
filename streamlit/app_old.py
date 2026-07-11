import tempfile
import joblib
import pandas as pd
import plotly.express as px
import streamlit as st
import time

from src.predict import predict_emotion

# ---------------------------------------------------
# Page Configuration
# ---------------------------------------------------

st.set_page_config(
    page_title="Emotion Recognition",
    page_icon="🎤",
    layout="wide"
)

# ---------------------------------------------------
# Custom CSS
# ---------------------------------------------------

st.markdown("""
<style>

/* Main App */
.stApp{
    background-color:#F5F7FA;
}

/* Title */
.main-title{
    font-size:46px;
    font-weight:800;
    color:#1E3A8A;
    text-align:center;
    margin-bottom:10px;
}

/* Subtitle */
.sub-title{
    font-size:18px;
    color:#555;
    text-align:center;
    margin-bottom:30px;
}

/* Prediction Card */
.prediction-card{
    background:linear-gradient(135deg,#2563EB,#1D4ED8);
    color:white;
    padding:30px;
    border-radius:18px;
    text-align:center;
    font-size:34px;
    font-weight:bold;
    box-shadow:0px 10px 30px rgba(0,0,0,0.2);
}

/* Footer */
.footer{
    text-align:center;
    color:#666;
    margin-top:40px;
    font-size:14px;
}

/* Sidebar */
section[data-testid="stSidebar"]{
    background:#FFFFFF;
}

/* Buttons */
.stButton>button{
    width:100%;
    border-radius:12px;
    height:50px;
    font-size:18px;
    font-weight:600;
    background:#2563EB;
    color:white;
}

.stButton>button:hover{
    background:#1D4ED8;
}

/* Metrics */
[data-testid="stMetric"]{
    background:skyblue;
    padding:15px;
    border-radius:12px;
    box-shadow:0px 2px 10px rgba(0,0,0,0.08);
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# Header
# ---------------------------------------------------

st.markdown(
    '<p class="main-title">🎤 Emotion Recognition from Speech</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="sub-title">AI Powered Speech Emotion Recognition using CNN + BiLSTM</p>',
    unsafe_allow_html=True
)

st.write(
    "Upload a speech (.wav) file and predict the emotion using our trained deep learning model."
)

# ---------------------------------------------------
# Sidebar
# ---------------------------------------------------

st.sidebar.title("📌 Project Information")

st.sidebar.info("""
### Model
CNN + BiLSTM

---

### Dataset
RAVDESS

---

### Audio Features
- MFCC
- Chroma
- Mel Spectrogram
- RMS Energy
- Zero Crossing Rate
- Spectral Centroid

---

### Emotions
- Angry
- Calm
- Disgust
- Fear
- Happy
- Neutral
- Sad
- Surprise
""")

# ---------------------------------------------------
# Load Label Encoder
# ---------------------------------------------------

encoder = joblib.load("saved_models/label_encoder.pkl")

# ---------------------------------------------------
# File Upload
# ---------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload WAV File",
    type=["wav"]
)

st.info(
    "Supported format: **.wav** | Sample Rate: **22050 Hz**"
)

# ---------------------------------------------------
# Prediction
# ---------------------------------------------------

if uploaded_file is not None:

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:

        temp_audio.write(uploaded_file.read())

        temp_path = temp_audio.name

    st.audio(temp_path)

    st.subheader("🎵 Uploaded Audio")

    st.success("Audio uploaded successfully.")

    if st.button("🎯 Predict Emotion"):

        with st.spinner("Predicting Emotion..."):

            start = time.time()

            try:
                emotion, confidence = predict_emotion(temp_path)

                end = time.time()

                prediction_time = end - start

            except Exception as e:
                st.error(f"Prediction Failed: {e}")
                st.stop()

        # ---------------- Prediction Card ----------------

        st.markdown(
            f"""
            <div class="prediction-card">

            Predicted Emotion

            <br><br>

            {emotion}

            </div>
            """,
            unsafe_allow_html=True
        )

        # ---------------- Metrics ----------------

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Predicted Emotion",
                emotion
            )

        with col2:
            st.metric(
                "Confidence",
                f"{max(confidence):.2f}%"
            )

        st.caption(
            f"Prediction completed in {prediction_time:.2f} seconds."
)

        # ---------------- Confidence Chart ----------------

        st.subheader("📊 Confidence Distribution")

        df = pd.DataFrame({

            "Emotion": encoder.classes_,
            "Confidence": confidence

        })

        df = df.sort_values(
            by="Confidence",
            ascending=False
        )

        fig = px.bar(

            df,

            x="Confidence",

            y="Emotion",

            orientation="h",

            text="Confidence",

            color="Confidence",

            color_continuous_scale="Blues"

        )

        fig.update_traces(

            texttemplate="%{text:.2f}",

            textposition="outside"

        )

        fig.update_layout(

            xaxis_title="Confidence (%)",

            yaxis_title="Emotion"

        )

        st.plotly_chart(

            fig,

            use_container_width=True

        )

st.markdown("---")

st.subheader("📌 Model Information")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Model", "CNN + BiLSTM")

with col2:
    st.metric("Dataset", "RAVDESS")

with col3:
    st.metric("Emotions", "8")

st.markdown("---")

st.subheader("📌 Model Information")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Model", "CNN + BiLSTM")

with col2:
    st.metric("Dataset", "RAVDESS")

with col3:
    st.metric("Emotions", "8")

with st.expander("📖 About This Project"):

    st.write("""
This application predicts human emotions from speech using a CNN + BiLSTM Deep Learning model.

Dataset:
RAVDESS

Audio Features:
- MFCC
- Chroma
- Mel Spectrogram
- RMS Energy
- Zero Crossing Rate
- Spectral Centroid

Frameworks:
TensorFlow
Librosa
Streamlit
Scikit-learn
""")
# ---------------------------------------------------
# Footer
# ---------------------------------------------------

st.markdown("---")

st.markdown(
"""
<div class="footer">

Developed using TensorFlow • Streamlit • Librosa

</div>
""",
unsafe_allow_html=True
)