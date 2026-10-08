import io
import os
import time
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CNN_MODEL_PATH = (
    PROJECT_ROOT / "ai" / "defect_detection" / "models" / "rim_cnn_v1.pth"
)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class RimCNNPrediction(BaseModel):
    """Prediction output contract for ConvNeXt-Tiny Rim CNN Classifier."""

    model: str = "rim_cnn_v1"
    class_id: int = Field(..., ge=0, le=7, description="Predicted class index (0-7)")
    defect_type: str = Field(..., description="Predicted rim defect category name")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Model confidence score [0.0, 1.0]")
    inference_ms: float = Field(..., ge=0.0, description="Inference execution latency in milliseconds")
    device: str = Field(..., description="Device used for inference ('cpu' or 'cuda')")


class RimCNNClassifier:
    """
    ConvNeXt-Tiny CNN Classifier for Aluminium Rim Defect Classification.

    Supports 8-class rim defect classification:
    [0: bent_rim, 1: blow_hole, 2: crack, 3: incomplete_welding,
     4: paint_damage, 5: porosity, 6: scratch, 7: scuff]
    """

    def __init__(self, model_path: Optional[Union[str, Path]] = None):
        self.model_path = Path(model_path) if model_path else DEFAULT_CNN_MODEL_PATH
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.class_names: List[str] = []
        self.num_classes: int = 0
        self.model_version: str = "rim_cnn_v1"

        self.transform = T.Compose(
            [
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

        self._load_model()

    def _load_model(self) -> None:
        """
        Load ConvNeXt-Tiny model weights from checkpoint. Handles both standard .pth
        single file checkpoints and unzipped directory packages.
        """
        if not self.model_path.exists():
            return

        try:
            if self.model_path.is_dir():
                base_dir = (
                    self.model_path / "convnext_tiny_8class_best"
                    if (self.model_path / "convnext_tiny_8class_best").exists()
                    else self.model_path
                )
                buf = io.BytesIO()
                with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zf:
                    for root, _, files in os.walk(base_dir):
                        for f in files:
                            full_path = os.path.join(root, f)
                            rel_path = os.path.relpath(full_path, base_dir)
                            zf.write(full_path, arcname=os.path.join("archive", rel_path))
                buf.seek(0)
                checkpoint = torch.load(buf, weights_only=False, map_location=self.device)
            else:
                checkpoint = torch.load(
                    self.model_path, weights_only=False, map_location=self.device
                )

            self.class_names = checkpoint.get(
                "class_names",
                [
                    "bent_rim",
                    "blow_hole",
                    "crack",
                    "incomplete_welding",
                    "paint_damage",
                    "porosity",
                    "scratch",
                    "scuff",
                ],
            )
            self.num_classes = checkpoint.get("num_classes", len(self.class_names))

            # Remap state dict keys if classifier was saved as classifier.2.1.weight
            msd = {}
            raw_sd = checkpoint.get("model_state_dict", checkpoint)
            for k, v in raw_sd.items():
                new_k = k.replace("classifier.2.1.", "classifier.2.")
                msd[new_k] = v

            model = models.convnext_tiny(num_classes=self.num_classes)
            model.load_state_dict(msd)
            model.to(self.device)
            model.eval()

            self.model = model
        except Exception as err:
            # Keep model as None if loading fails
            self.model = None

    @property
    def is_ready(self) -> bool:
        return self.model is not None

    def predict(self, image_input: Union[str, Path, Image.Image]) -> RimCNNPrediction:
        """
        Run defect classification on a wheel image or PIL Image object.

        Raises:
            RuntimeError: if the CNN model is not loaded.
            ValueError: if image cannot be loaded.
        """
        if not self.is_ready:
            raise RuntimeError("Rim CNN model is not available")

        if isinstance(image_input, (str, Path)):
            img_path = Path(image_input)
            if not img_path.is_absolute():
                img_path = PROJECT_ROOT / img_path

            if not img_path.exists():
                raise ValueError(f"Image path does not exist: {image_input}")

            try:
                pil_img = Image.open(img_path).convert("RGB")
            except Exception as e:
                raise ValueError(f"Invalid image file: {str(e)}")
        elif isinstance(image_input, Image.Image):
            pil_img = image_input.convert("RGB")
        else:
            raise ValueError("Unsupported image input type")

        tensor_img = self.transform(pil_img).unsqueeze(0).to(self.device)

        start_time = time.time()
        with torch.inference_mode():
            logits = self.model(tensor_img)
            probabilities = torch.softmax(logits, dim=1)

        latency_ms = (time.time() - start_time) * 1000.0

        confidence_tensor, pred_idx_tensor = torch.max(probabilities, dim=1)
        pred_idx = pred_idx_tensor.item()
        confidence = round(float(confidence_tensor.item()), 4)

        defect_name = (
            self.class_names[pred_idx]
            if pred_idx < len(self.class_names)
            else f"unknown_{pred_idx}"
        )

        return RimCNNPrediction(
            model=self.model_version,
            class_id=pred_idx,
            defect_type=defect_name,
            confidence=confidence,
            inference_ms=round(latency_ms, 2),
            device=str(self.device.type),
        )


# Singleton instance
rim_cnn_classifier = RimCNNClassifier()
