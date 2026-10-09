import os
from typing import List

from dotenv import load_dotenv

# Load environment variables from the project's .env file.
load_dotenv()


class Settings:
    PROJECT_NAME: str = "Aluminium Wheel Quality Intelligence"
    API_V1_STR: str = "/api"

    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")

    # Use a strong secret from the environment in production.
    SECRET_KEY: str = os.getenv(
        "SECRET_KEY",
        "dev_secret_key_insecure_default",
    )

    API_BASE_URL: str = os.getenv(
        "API_BASE_URL",
        "http://localhost:8000",
    )

    # Retained for compatibility with the existing project configuration.
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL",
        "postgresql://quality_user:quality_secret@localhost:5432/quality_db",
    )

    MODEL_PATH: str = os.getenv(
        "MODEL_PATH",
        "ai/defect_detection/models/",
    )

    # Allowed frontend origins for FastAPI CORS.
    # Override these defaults using CORS_ORIGINS in .env.
    CORS_ORIGINS: List[str] = [
        origin.strip()
        for origin in os.getenv(
            "CORS_ORIGINS",
            (
                "http://localhost:5174,"
                "http://127.0.0.1:5174,"
                "http://localhost:5173,"
                "http://127.0.0.1:5173,"
                "http://localhost:3000,"
                "http://127.0.0.1:3000"
            ),
        ).split(",")
        if origin.strip()
    ]


settings = Settings()