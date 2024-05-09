import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
import librosa
import os

# Load your trained model
model = load_model('cnn.h5')

# Emotion mapping dictionary
emotions = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

# Define the function to extract features from the audio file
def extract_feature(data, sr, mfcc=True, chroma=True, mel=True):
    """
    Extract features from audio files into a numpy array

    Parameters
    ----------
    data : np.ndarray, audio time series
    sr : number > 0, sampling rate
    mfcc : boolean, Mel Frequency Cepstral Coefficient, represents the short-term power spectrum of a sound
    chroma : boolean, pertains to the 12 different pitch classes
    mel : boolean, Mel Spectrogram Frequency
    """
    if chroma:
        stft = np.abs(librosa.stft(data))
    result = np.array([])

    if mfcc:
        mfccs = np.mean(librosa.feature.mfcc(y=data, sr=sr, n_mfcc=40).T, axis=0)
        result = np.hstack((result, mfccs))

    if chroma:
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T, axis=0)
        result = np.hstack((result, chroma))

    if mel:
        mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sr).T, axis=0)
        result = np.hstack((result, mel))

    return result

# Define the function to predict emotion
def predict_emotion(features):
    prediction = model.predict(np.expand_dims(features, axis=0))
    predicted_class = np.argmax(prediction, axis=1)[0]
    emotion_label = emotions.get(str(predicted_class + 1).zfill(2), 'Unknown')
    return emotion_label

# Streamlit app
def main():
    st.title("Speech Emotion Recognition")

    # Upload audio file
    audio_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])

    if audio_file is not None:
        # Load the audio data
        audio_data, sr = librosa.load(audio_file, sr=None)

        # Extract features from the audio data
        features = extract_feature(audio_data, sr)

        # Make a prediction
        emotion = predict_emotion(features)

        html_str = f"""
            <div style="background-color: #F4F6F6; padding: 20px; border-radius: 10px; box-shadow: 0 4px 8px 0 rgba(0, 0, 0, 0.2), 0 6px 20px 0 rgba(0, 0, 0, 0.19);">
                <h1 style="color: #2E8B57; text-align: center; font-family: 'Georgia', serif;">The predicted emotion for the audio is:</h1>
                <h2 style="color: #2E8B57; text-align: center; font-family: 'Georgia', serif; font-size: 48px;">{emotion}</h2>
            </div>
        """
        st.markdown(html_str, unsafe_allow_html=True)

if __name__ == "__main__":
    main()