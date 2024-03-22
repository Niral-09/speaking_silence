from Speaking_Silence.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from Speaking_Silence.utils.common import read_yaml, create_directories
from Speaking_Silence.entity.config_entity import DataIngestionConfig
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