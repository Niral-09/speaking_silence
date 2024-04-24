import os
import pandas as pd
import streamlit as st
from Speaking_Silence.utils.common import decodeImage # Assuming this is the correct import path
from Speaking_Silence.pipeline.prediction import PredictionPipeline

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
clApp = ClientApp()

st.title("Speaking Silence")

col1, col2 = st.columns(2)

uploaded_file = st.sidebar.file_uploader("Upload File: ")

with col1:
    if st.button("Predict gesture"):
        with st.spinner("Adding data..."):
            original_filename = uploaded_file.name
            clApp.filename = original_filename
            clApp.update_filename(clApp.filename)
            if uploaded_file.type.startswith('video'):
                original_filename = uploaded_file.name
                with open(original_filename, "wb") as f:
                    f.write(uploaded_file.getvalue())
                clApp.update_filename(original_filename)
                result = clApp.classifier.predict()
                st.write(result)
            else:

                decodeImage(uploaded_file.getvalue(), uploaded_file.name)
                result = clApp.classifier.predict()
                st.write(result)
    else:
        st.write("Please upload a file.")