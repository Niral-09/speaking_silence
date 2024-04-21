import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, TimeDistributed, Dropout, LSTM
from tensorflow.keras.models import Sequential
from pathlib import Path
from Speaking_Silence import logger


class PrepareBaseModel:
    def __init__(self, config):
        self.config = config

    def build_base_model(self):
        logger.info(f"Preapring Base Model...")
        model = Sequential()

        model.add(TimeDistributed(Conv2D(16, (3, 3), padding='same',activation = 'relu'),
                                input_shape = (self.config.sequence_length, self.config.image_height, self.config.image_width, 3)))
        
        model.add(TimeDistributed(MaxPooling2D((4, 4)))) 
        model.add(TimeDistributed(Dropout(0.25)))
        
        model.add(TimeDistributed(Conv2D(32, (3, 3), padding='same',activation = 'relu')))
        model.add(TimeDistributed(MaxPooling2D((4, 4))))
        model.add(TimeDistributed(Dropout(0.25)))
        
        model.add(TimeDistributed(Conv2D(64, (3, 3), padding='same',activation = 'relu')))
        model.add(TimeDistributed(MaxPooling2D((2, 2))))
        model.add(TimeDistributed(Dropout(0.25)))
        
        model.add(TimeDistributed(Conv2D(64, (3, 3), padding='same',activation = 'relu')))
        model.add(TimeDistributed(MaxPooling2D((2, 2))))
        model.add(TimeDistributed(Dropout(0.25)))
                                        
        model.add(TimeDistributed(Flatten()))
                                        
        model.add(LSTM(32))
                                        
        model.add(Dense((self.config.num_classes), activation = 'softmax'))

        model.summary()
        return model

    def save_model(self, model, path):
        model.save(path)

    def prepare_and_save_base_model(self):
        base_model = self.build_base_model()
        self.save_model(base_model, self.config.base_model_path)
