import os
from typing import List
from dotenv import load_dotenv

load_dotenv()

class Settings:
    PROJECT_NAME: str = "Aluminium Wheel Quality Intelligence"
    API_V1_STR: str = "/api"
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    SECRET_KEY: str = os.getenv("SECRET_KEY", "dev_secret_key_insecure_default")
    API_BASE_URL: str = os.getenv("API_BASE_URL", "http://localhost:8000")

    DATABASE_URL: str = os.getenv(
        "DATABASE_URL",
        "postgresql://quality_user:quality_secret@localhost:5432/quality_db"
    )

    MODEL_PATH: str = os.getenv(
        "MODEL_PATH",
        "ai/defect_detection/models/"
    )

    CORS_ORIGINS: List[str] = [
        origin.strip()
        for origin in os.getenv(
            "CORS_ORIGINS",
            "http://localhost:5173,http://localhost:3000"
        ).split(",")
        if origin.strip()
    ]


settings = Settings()