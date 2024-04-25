import os
import pandas as pd
import streamlit as st
from Speaking_Silence.utils.common import decodeImage # Assuming this is the correct import path
from Speaking_Silence.pipeline.prediction import PredictionPipeline
from googletrans import Translator
from gtts import gTTS

st.set_page_config(
    layout="centered",
)

class ClientApp:
    def __init__(self):
        # Initialize with a placeholder or default filename
        self.filename = "video.mp4"
        self.classifier = PredictionPipeline(self.filename)

    def update_filename(self, new_filename):
        self.filename = new_filename
        self.classifier.update_filename(self.filename) 

    def predict_gesture(self, image_data):
        # Predict gesture and return result
        result = self.classifier.predict(image_data)
        return result

clApp = ClientApp()
translator = Translator()

st.title("Speaking Silence")

col1, col2 = st.columns(2)

uploaded_file = st.sidebar.file_uploader("Upload File: ")

with col1:
    if st.button("Predict gesture"):
        with st.spinner("Adding data..."):
            if uploaded_file is not None:
                if uploaded_file.type.startswith('video'):
                    # If video file is uploaded
                    with open(uploaded_file.name, "wb") as f:
                        f.write(uploaded_file.getvalue())
                    clApp.update_filename(uploaded_file.name)
                    result = clApp.classifier.predict()
                    st.write("Predicted Gesture:", result)
                    # Translation
                    dest_language = st.selectbox("Select Destination Language:", ["fr", "es", "de"])  # Dropdown menu for selecting destination language
                    translation = translator.translate(result, dest=dest_language)
                    st.write("Translated Result:", translation.text)
                    
                    # Text-to-speech
                    tts = gTTS(text=translation.text, lang=dest_language)
                    tts.save("translated_result.mp3")
                    st.audio("translated_result.mp3", format='audio/mp3', start_time=0)
                else:
                    # If image file is uploaded
                    image_data = decodeImage(uploaded_file.getvalue(), uploaded_file.name)
                    result = clApp.predict_gesture(image_data)
                    st.write("Predicted Gesture:", result)
                    
                    # Translation
                    dest_language = st.selectbox("Select Destination Language:", ["fr", "es", "de"])  # Dropdown menu for selecting destination language
                    translation = translator.translate(result, dest=dest_language)
                    st.write("Translated Result:", translation.text)
                    
                    # Text-to-speech
                    tts = gTTS(text=translation.text, lang=dest_language)
                    tts.save("translated_result.mp3")
                    st.audio("translated_result.mp3", format='audio/mp3', start_time=0)

            else:
                st.write("Please upload a file.")
    else:
        st.write("Please upload a file.")
