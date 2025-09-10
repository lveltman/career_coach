from pathlib import Path
from pydantic_settings import BaseSettings

ROOT_DIR = Path(__file__).parent

class DefaultPipelineSettings(BaseSettings):
    # API
    API_TOKEN: str
    MODEL_URL: str
    MODEL_NAME: str
    FOLDER_ID: str
    MODEL_TEMP: float

    MAX_HISTORY: int

    class Config:
        env_file = ROOT_DIR / ".env"
        env_file_encoding = "utf-8"


config = DefaultPipelineSettings()
