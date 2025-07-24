import streamlit as st
import os
import pickle
import numpy as np
import speech_recognition as sr
from textblob import TextBlob
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# Constants
UPLOAD_FOLDER = 'uploads'
MAX_SEQUENCE_LENGTH = 100
TOKENIZER_FILE = 'tokenizer.pickle'

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the tokenizer
try:
    with open(TOKENIZER_FILE, 'rb') as handle:
        tokenizer = pickle.load(handle)
except FileNotFoundError:
    st.error("Tokenizer file not found. Make sure 'tokenizer.pickle' exists in the working directory.")
    tokenizer = None

# Function to transcribe audio
def transcribe_audio(audio_file):
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(audio_file) as source:
            audio_data = recognizer.record(source)
        return recognizer.recognize_google(audio_data)
    except sr.RequestError as e:
        st.error(f"Speech recognition request failed: {e}")
    except sr.UnknownValueError:
        st.error("Could not understand audio")
    except Exception as ex:
        st.error(f"An error occurred during audio processing: {ex}")

# Preprocess the text into padded sequences
def preprocess_text(text):
    if tokenizer is None:
        return None
    sequence = tokenizer.texts_to_sequences([text])
    return pad_sequences(sequence, maxlen=MAX_SEQUENCE_LENGTH)

# Analyze sentiment using TextBlob (you can extend this with a custom model)
def analyze_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0:
        return "Positive 😊"
    elif polarity < 0:
        return "Negative 😞"
    else:
        return "Neutral 😐"

# Main function to run the Streamlit app
def main():
    st.title('🎙️ Audio Sentiment Analysis App')

    uploaded_file = st.file_uploader("Upload a WAV audio file", type=["wav"])

    if uploaded_file is not None:
        audio_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
        
        # Save the uploaded audio
        with open(audio_path, 'wb') as f:
            f.write(uploaded_file.getvalue())

        # Transcribe the uploaded audio
        transcribed_text = transcribe_audio(audio_path)

        if transcribed_text:
            st.subheader("🔤 Transcribed Text")
            st.write(transcribed_text)

            # Sentiment Analysis
            sentiment = analyze_sentiment(transcribed_text)
            st.markdown(f"<p style='font-size:24px; color:gold;'>Sentiment: {sentiment}</p>", unsafe_allow_html=True)

        # Play the uploaded audio
        st.audio(open(audio_path, 'rb'), format='audio/wav')

if __name__ == '__main__':
    main()