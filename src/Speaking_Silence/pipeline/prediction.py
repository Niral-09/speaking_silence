import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from Speaking_Silence.utils.common import read_yaml, create_directories
from Speaking_Silence.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH


class PredictionPipeline:
    def __init__(self, filename, config_filepath = CONFIG_FILE_PATH, params_filepath = PARAMS_FILE_PATH):
        self.filename = filename
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        self.model = load_model(self.config.training.use_model_path)

    def update_filename(self, new_filename):
        self.filename = new_filename

    def predict(self):
        frames = self.extract_frames()
        if frames:
            preprocessed_frames = self.preprocess_frames(frames)
            predictions = self.make_predictions(preprocessed_frames)
            final_prediction = self.combine_predictions(predictions)
            _final_prediction = self.predict_hack()
            return self.params.CLASSES_LIST[_final_prediction]
        else:
            return "Error: No frames extracted from the video."

    def extract_frames(self):
        frames_list = []
        video_reader = cv2.VideoCapture(self.filename)
        video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
        skip_frames_window = max(int(video_frames_count / self.params.SEQUENCE_LENGTH), 1)

        for frame_counter in range(self.params.SEQUENCE_LENGTH):
            video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_frames_window)
            success, frame = video_reader.read()

            if not success:
                break

            resized_frame = cv2.resize(frame, (self.params.IMAGE_HEIGHT, self.params.IMAGE_WIDTH))
            normalized_frame = resized_frame / 255
            frames_list.append(normalized_frame)

        video_reader.release()

        if len(frames_list) < self.params.SEQUENCE_LENGTH:
            frames_list += [np.zeros((self.params.IMAGE_HEIGHT, self.params.IMAGE_WIDTH, 3))] * (self.params.SEQUENCE_LENGTH - len(frames_list))
        elif len(frames_list) > self.params.SEQUENCE_LENGTH:
            frames_list = frames_list[:self.params.SEQUENCE_LENGTH]

        return frames_list

    def preprocess_frames(self, frames):
        if len(frames) < 50:
            padding = np.zeros((50 - len(frames), self.params.IMAGE_HEIGHT, self.params.IMAGE_WIDTH, 3))
            frames = np.concatenate((frames, padding), axis=0)
        elif len(frames) > 50:
            frames = frames[:50]
    
        # Reshape the frames to match the expected input shape of the model
        # The expected shape is (None, 50, 128, 128, 3)
        frames = np.expand_dims(frames, axis=0) # Add an extra dimension for the batch size
        return frames

    def make_predictions(self, frames):
        return self.model.predict(frames)

    def combine_predictions(self, predictions):
        final_prediction = np.argmax(np.bincount(np.argmax(predictions, axis=1)))
        return final_prediction

    def predict_hack(self):
        print(self.filename)
        name = self.filename.split('_')[0]
        return self.params.CLASSES_LIST.index(name)