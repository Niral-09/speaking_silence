from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    json_file: Path  # Path to the JSON file containing data
    artifact_folder: Path  # Root directory where artifacts will be stored
    user_agent: str  # User-Agent string for HTTP requests
    db_name: str  # MongoDB database name
    db_host: str # MongoDB host URL


@dataclass(frozen=True)
class PrepareBaseModelConfig:
    root_dir: Path  # Root directory of the project
    base_model_path: Path  # Path to save the base model
    updated_base_model_path: Path  # Path to save the updated base model

    input_shape: tuple  # Input shape of the model (e.g., (height, width, channels))
    num_classes: int  # Number of gesture classes
    learning_rate: float  # Learning rate for model training
    include_top: bool  # Whether to include the top layers (True for fine-tuning)