import os
import numpy as np
import cv2
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import plot_model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from pathlib import Path
from Speaking_Silence import logger
from Speaking_Silence.entity.config_entity import TrainingConfig




class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config

    
    def get_base_model(self):
        self.model = tf.keras.models.load_model(
            self.config.base_model_path
        )
    
    def print_model_summary(self):
        self.model.summary()

    def plot_metric(self, model_training_history, metric_name_1, metric_name_2, plot_name):

        metric_value_1 = model_training_history.history[metric_name_1]
        metric_value_2 = model_training_history.history[metric_name_2]
        
        epochs = range(len(metric_value_1))

        plt.plot(epochs, metric_value_1, 'blue', label = metric_name_1)
        plt.plot(epochs, metric_value_2, 'red', label = metric_name_2)

        plt.title(str(plot_name))

        plt.legend()
        plt.savefig(f"artifacts/training/{plot_name}.png")
        plt.close()

    def frames_extraction(self, video_path):
        frames_list = []
        video_reader = cv2.VideoCapture(video_path)
        video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
        skip_frames_window = max(int(video_frames_count/self.config.sequence_length), 1)

        for frame_counter in range(self.config.sequence_length):
            video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_frames_window)
            success, frame = video_reader.read() 

            if not success:
                break

            resized_frame = cv2.resize(frame, (self.config.image_height, self.config.image_width))            
            normalized_frame = resized_frame / 255            
            frames_list.append(normalized_frame)
        
        video_reader.release()

        return frames_list

    def create_dataset(self):
        features = []
        labels = []
        video_files_paths = []
        
        for class_index, class_name in enumerate(self.config.classes_list):            
            print(f'Extracting Data of Class: {class_name}')            
            files_list = os.listdir(os.path.join(self.config.training_data_path, class_name))
            
            for file_name in files_list:                
                video_file_path = os.path.join(self.config.training_data_path, class_name, file_name)
                frames = self.frames_extraction(video_file_path)

                if len(frames) == self.config.sequence_length:
                    features.append(frames)
                    labels.append(class_index)
                    video_files_paths.append(video_file_path)

        features = np.asarray(features)
        labels = np.array(labels)  
        
        return features, labels, video_files_paths

    def train_model(self):
        seed_constant = 27
        features, labels, video_files_paths = self.create_dataset()
        one_hot_encoded_labels = to_categorical(labels)
        features_train, features_test, labels_train, labels_test = train_test_split(features, one_hot_encoded_labels,
                                                                            test_size = 0.25, shuffle = True,
                                                                            random_state = seed_constant)
        np.save('artifacts/training/features_test.npy', features_test)
        np.save('artifacts/training/labels_test.npy', labels_test)
        LRCN_model = self.model
        plot_model(LRCN_model, to_file = 'artifacts/training/LRCN_model_structure_plot.png', show_shapes = True, show_layer_names = True)
        LRCN_model.compile(loss = 'categorical_crossentropy', optimizer = 'Adam', metrics = ["accuracy"])

        LRCN_model_training_history = LRCN_model.fit(x = features_train, 
                                                    y = labels_train, 
                                                    epochs = self.config.params_epochs,
                                                    batch_size = self.config.params_batch_size,
                                                    validation_split = 0.2)
        model_evaluation_history = LRCN_model.evaluate(features_test, labels_test)
        model_evaluation_loss, model_evaluation_accuracy = model_evaluation_history
        
        self.plot_metric(LRCN_model_training_history, 'loss', 'val_loss', 'Total Loss vs Total Validation Loss')
        self.plot_metric(LRCN_model_training_history, 'accuracy', 'val_accuracy', 'Total Accuracy vs Total Validation Accuracy')

        LRCN_model.save(self.config.trained_model_path)
