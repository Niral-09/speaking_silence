import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Sequential
from pathlib import Path
from Speaking_Silence import logger


class PrepareBaseModel:
    def __init__(self, config):
        self.config = config

    def build_base_model(self):
        logger.info(f"Preapring Base Model...")
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=self.config.input_shape),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(128, activation='relu'),
            Dense(self.config.num_classes, activation='softmax')
        ])
        return model

    def save_model(self, model, path):
        model.save(path)

    def prepare_and_save_base_model(self):
        base_model = self.build_base_model()
        self.save_model(base_model, self.config.base_model_path)
