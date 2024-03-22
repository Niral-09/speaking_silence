from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    json_file: Path  # Path to the JSON file containing data
    artifact_folder: Path  # Root directory where artifacts will be stored
    user_agent: str  # User-Agent string for HTTP requests
    db_name: str  # MongoDB database name
    db_host: str # MongoDB host URL