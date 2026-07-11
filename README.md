# 🎤 Emotion Recognition from Speech using Deep Learning

An end-to-end Deep Learning application that recognizes human emotions from speech audio using a **CNN + Bidirectional LSTM (BiLSTM)** model. The project extracts speech features from audio recordings, predicts one of eight human emotions, and provides an interactive Streamlit web application for real-time inference.

---

## 🚀 Project Highlights

- 🎙️ Speech Emotion Recognition using Deep Learning
- 🧠 CNN + BiLSTM Architecture
- 🎵 Audio Feature Extraction using Librosa
- 📊 Confidence Score Visualization
- 🌐 Interactive Streamlit Web Application
- 💾 Saved Model for Inference
- 📈 Model Evaluation using Multiple Metrics
- 📥 Download Prediction Results
- 🛠 Modular & Production-Oriented Code Structure

---

## 📖 Project Overview

Speech Emotion Recognition (SER) is a machine learning task that identifies the emotional state of a speaker from audio signals.

This project uses the **RAVDESS dataset** and a **CNN + Bidirectional LSTM** architecture to classify speech into eight different emotions.

The application allows users to upload a speech recording (.wav file), processes the audio, extracts meaningful features, predicts the emotion, and displays confidence scores through a user-friendly Streamlit interface.

---

## 😊 Supported Emotions

- Angry
- Calm
- Disgust
- Fear
- Happy
- Neutral
- Sad
- Surprise

---

# 🧠 Model Architecture

```
Speech Audio
      │
      ▼
Audio Preprocessing
      │
      ▼
MFCC Feature Extraction
      │
      ▼
CNN Layers
      │
      ▼
Batch Normalization
      │
      ▼
Max Pooling
      │
      ▼
Dropout
      │
      ▼
Bidirectional LSTM
      │
      ▼
Dense Layers
      │
      ▼
Softmax Classifier
      │
      ▼
Emotion Prediction
```

---

# ⚙️ Working Methodology

The application follows the complete Speech Emotion Recognition pipeline.

### Step 1 — Audio Input

The user uploads a speech recording through the Streamlit web application.

---

### Step 2 — Audio Preprocessing

The uploaded audio is preprocessed using Librosa.

Processing includes:

- Audio Loading
- Normalization
- Fixed Sampling Rate
- Padding / Trimming

---

### Step 3 — Feature Extraction

The following audio features are extracted.

- MFCC
- Chroma Features
- Mel Spectrogram
- Zero Crossing Rate (ZCR)
- RMS Energy
- Spectral Centroid

MFCC features are used as the primary model input.

---

### Step 4 — Model Prediction

The extracted features are passed into the trained CNN + BiLSTM model.

The model predicts one of the supported emotions and calculates confidence scores for all emotion classes.

---

### Step 5 — Result Visualization

The Streamlit application displays

- Predicted Emotion
- Confidence Distribution
- Model Information
- Downloadable Prediction Report

---

# 📊 Model Performance

| Metric | Score |
|---------|-------|
| Accuracy | **55.21%** |
| Precision | **59.15%** |
| Recall | **55.21%** |
| F1-Score | **54.54%** |

The model was evaluated using the **RAVDESS Speech Emotion Dataset**.

---

# 📂 Dataset

**Dataset Used**

RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)

The dataset contains professionally recorded emotional speech samples from multiple actors.

---

# 🛠 Technology Stack

| Category | Technology |
|-----------|------------|
| Programming Language | Python |
| Deep Learning | TensorFlow / Keras |
| Audio Processing | Librosa |
| Machine Learning | Scikit-learn |
| Data Processing | NumPy, Pandas |
| Visualization | Matplotlib, Plotly |
| Web Framework | Streamlit |

---

# 📁 Project Structure

```text
Emotion_Recognition/
│
├── dataset/
│   ├── raw/
│   └── features/
│
├── saved_models/
│
├── images/
│
├── logs/
│
├── src/
│   ├── config.py
│   ├── preprocessing.py
│   ├── feature_extraction.py
│   ├── augmentation.py
│   ├── prepare_dataset.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│
├── streamlit/
│   └── app.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

# ⚙️ Installation

Clone the repository

```bash
git clone <repository-url>
```

Move into the project directory

```bash
cd Emotion_Recognition
```

Create a virtual environment

```bash
python -m venv venv
```

Activate the virtual environment

### Windows

```bash
.\venv\Scripts\activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Application

Launch the Streamlit application.

```bash
python -m streamlit run streamlit/app.py
```

---

# 🧠 Training the Model

```bash
python src/train.py
```

---

# 📈 Evaluating the Model

```bash
python test_evaluate.py
```

---

# 🎯 Predict Emotion

```bash
python test_prediction.py
```

---

# 📸 Application Preview

The application provides

- Audio Upload
- Audio Playback
- Emotion Prediction
- Confidence Distribution Chart
- Model Information
- Download Prediction Results

> Screenshots of the application will be added here.

---

# 🚀 Future Improvements

- Improve model accuracy using larger datasets
- Support additional speech emotion datasets
- Real-time microphone prediction
- Multilingual speech emotion recognition
- Model optimization for mobile deployment
- Hugging Face Spaces deployment
- Streamlit Community Cloud deployment
- ONNX / TensorFlow Lite model conversion

---

# 🙏 Acknowledgements

- RAVDESS Dataset
- TensorFlow
- Librosa
- Streamlit
- Scikit-learn

---

# 👩‍💻 Author

**Vanshika Mehra**

Bachelor of Technology (Artificial Intelligence & Machine Learning)

Government Mahila Engineering College, Ajmer

---

# 📜 License

This project is developed for educational, research, and portfolio purposes.