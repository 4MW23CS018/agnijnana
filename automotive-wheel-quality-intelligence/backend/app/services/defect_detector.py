from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel


class Detection(BaseModel):
    defect_type: str
    location: List[float]
    confidence: float


class DetectionResult(BaseModel):
    model_loaded: bool
    model_version: Optional[str] = None
    detections: List[Detection] = []


class DefectDetector:
    """
    Boundary between the FastAPI backend and the defect-detection model.

    The actual trained model will be plugged into this class once model
    weights are available.
    """

    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        self.model = None
        self.model_version = None

        self._load_model()

    def _load_model(self) -> None:
        """
        Load the trained detection model if the weights exist.

        We deliberately do not create fake predictions when the model
        is unavailable.
        """

        if not self.model_path.exists():
            return

        # Model integration will be added once Vijeath provides
        # the final trained weights and inference configuration.
        #
        # Example future implementation:
        #
        # from ultralytics import YOLO
        # self.model = YOLO(str(self.model_path))
        # self.model_version = self.model_path.name

    @property
    def is_ready(self) -> bool:
        return self.model is not None

    def predict(self, image_path: str) -> DetectionResult:
        """
        Run defect detection on an image.

        Raises:
            RuntimeError: if the trained model is not available.
        """

        if not self.is_ready:
            raise RuntimeError(
                "Defect detection model is not loaded. "
                "Please provide the trained model weights."
            )

        # Actual model inference will be implemented here.
        #
        # Example:
        # results = self.model(image_path)
        #
        # detections = ...
        #
        # return DetectionResult(
        #     model_loaded=True,
        #     model_version=self.model_version,
        #     detections=detections,
        # )

        raise NotImplementedError(
            "Model inference has not been implemented yet."
        )