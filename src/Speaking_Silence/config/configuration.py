from Speaking_Silence.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from Speaking_Silence.utils.common import read_yaml, create_directories
from Speaking_Silence.entity.config_entity import DataIngestionConfig, PrepareBaseModelConfig
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
            updated_base_model_path=Path(config.updated_base_model_path),
            input_shape=self.params.INPUT_SHAPE,
            learning_rate=self.params.LEARNING_RATE,
            include_top=self.params.INCLUDE_TOP,
            num_classes=self.params.CLASSES
        )

        return prepare_base_model_config