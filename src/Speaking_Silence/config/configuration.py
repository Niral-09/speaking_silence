from Speaking_Silence.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from Speaking_Silence.utils.common import read_yaml, create_directories
from Speaking_Silence.entity.config_entity import DataIngestionConfig, PrepareBaseModelConfig, TrainingConfig, EvaluationConfig
import os
from pathlib import Path


class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])


    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion


        data_ingestion_config = DataIngestionConfig(
            json_file=config.json_file,
            artifact_folder=config.artifact_folder,
            user_agent=config.user_agent,
            db_name=config.db_name,
            db_host=config.db_host,
        )

        return data_ingestion_config

    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        config = self.config.prepare_base_model
        
        create_directories([config.root_dir])

        prepare_base_model_config = PrepareBaseModelConfig(
            root_dir=Path(config.root_dir),
            base_model_path=Path(config.base_model_path),
            image_height=self.params.IMAGE_HEIGHT,
            image_width=self.params.IMAGE_WIDTH,
            sequence_length=self.params.SEQUENCE_LENGTH,
            learning_rate=self.params.LEARNING_RATE,
            include_top=self.params.INCLUDE_TOP,
            num_classes=self.params.CLASSES
        )

        return prepare_base_model_config

    def get_training_config(self) -> TrainingConfig:
        training = self.config.training
        prepare_base_model = self.config.prepare_base_model
        params = self.params
        training_data = (self.config.prepare_base_model.dataset_dir)

        training_config = TrainingConfig(
            root_dir=Path(training.root_dir),
            base_model_path=Path(prepare_base_model.base_model_path),
            trained_model_path=Path(training.trained_model_path),
            training_data_path=Path(training_data),
            params_epochs=params.EPOCHS,
            params_batch_size=params.BATCH_SIZE,
            params_is_augmentation=params.AUGMENTATION,
            image_height=self.params.IMAGE_HEIGHT,
            image_width=self.params.IMAGE_WIDTH,
            sequence_length=self.params.SEQUENCE_LENGTH,
            classes_list=self.params.CLASSES_LIST,
        )

        return training_config

    def get_evaluation_config(self) -> EvaluationConfig:
        eval_config = EvaluationConfig(
            path_of_model=self.config.training.trained_model_path,
            mlflow_uri=os.environ["MLFLOW_TRACKING_URI"],
            all_params=self.params,
        ) 
        return eval_config