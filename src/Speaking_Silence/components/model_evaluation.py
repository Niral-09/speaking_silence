import tensorflow as tf
from pathlib import Path
import mlflow
import mlflow.keras
from urllib.parse import urlparse
import keras
keras_version = keras.__version__
from Speaking_Silence.utils.common import save_json
from Speaking_Silence.entity.config_entity import EvaluationConfig
import numpy as np


class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.features_test = np.load('artifacts/training/features_test.npy')
        self.labels_test = np.load('artifacts/training/labels_test.npy')

    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        return tf.keras.models.load_model(path)

    def validate(self):
        model_evaluation_history = self.model.evaluate(self.features_test, self.labels_test)
        model_evaluation_loss, model_evaluation_accuracy = model_evaluation_history
        return model_evaluation_loss, model_evaluation_accuracy

    def evaluation(self):
        self.model = self.load_model(self.config.path_of_model)
        self.score = self.validate()
        self.save_score()

    def save_score(self):
        scores = {"loss": self.score[0], "accuracy": self.score[1]}
        save_json(path=Path("scores.json"), data=scores)

    def log_into_mlflow(self):
        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        keras_version = keras.__version__
        print(keras_version)
        with mlflow.start_run():
            mlflow.log_params(self.config.all_params)
            mlflow.log_metrics(
                {"loss": self.score[0], "accuracy": self.score[1]}
            )
            if tracking_url_type_store != "file":
                mlflow.keras.log_model(self.model, "model",  registered_model_name="speaking_silence_model")
            else:
                mlflow.keras.log_model(self.model, "model")