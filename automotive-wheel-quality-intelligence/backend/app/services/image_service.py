import io
from pathlib import Path
from uuid import uuid4

from fastapi import HTTPException, UploadFile
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[3]
UPLOAD_DIR = PROJECT_ROOT / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


async def save_wheel_image(file: UploadFile) -> str:
    """
    Validate and save an uploaded wheel image. Verify binary header using PIL.

    Returns the project-relative path of the saved image.
    """

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Image filename is missing",
        )

    extension = Path(file.filename).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported image format: {extension}",
        )

    contents = await file.read()

    if not contents:
        raise HTTPException(
            status_code=400,
            detail="Uploaded image is empty",
        )

    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="Image exceeds the 10 MB limit",
        )

    # Verify that contents is a valid RGB-convertible image
    try:
        img = Image.open(io.BytesIO(contents))
        img.verify()
        # Re-open after verify() to test format/conversion
        img_check = Image.open(io.BytesIO(contents))
        img_check.convert("RGB")
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is not a valid image or is corrupted",
        )

    filename = f"{uuid4().hex}{extension}"
    output_path = UPLOAD_DIR / filename

    output_path.write_bytes(contents)

    return str(output_path.relative_to(PROJECT_ROOT))