import streamlit as st
import pandas as pd
import plotly.express as px

from src.predict import predict_emotion

st.set_page_config(
    page_title="Emotion Recognition",
    page_icon="🎤",
    layout="wide"
)

st.title("🎤 Emotion Recognition from Speech")

st.write(
    "Upload a speech (.wav) file and predict the emotion using our CNN + BiLSTM model."
)

st.sidebar.title("Model Information")

st.sidebar.markdown("""
### Model

- CNN + BiLSTM

### Dataset

- RAVDESS

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

uploaded_file = st.file_uploader(
    "Upload WAV File",
    type=["wav"]
)

if uploaded_file:

    st.audio(uploaded_file)

import tempfile

if uploaded_file is not None:

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:

        temp_audio.write(uploaded_file.read())

        temp_path = temp_audio.name

    st.audio(temp_path)


if uploaded_file is not None:

    if st.button("Predict Emotion"):

        emotion, confidence = predict_emotion(temp_path)
        
        st.success(f"Predicted Emotion: {emotion}")

import joblib

encoder = joblib.load(
    "saved_models/label_encoder.pkl"
)

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

    title="Confidence Scores"

)

fig.update_traces(

    texttemplate="%{text:.2f}",

    textposition="outside"

)

st.plotly_chart(

    fig,

    use_container_width=True

)