import asyncio
from fastapi import HTTPException, status
from app.services.rim_cnn_classifier import rim_cnn_classifier, RimCNNPrediction


def run_rim_cnn_inference(image_path: str) -> RimCNNPrediction:
    """
    Execute Rim CNN classification inference on an image path.
    """
    if not rim_cnn_classifier.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Rim CNN model is not available",
        )

    try:
        return rim_cnn_classifier.predict(image_path)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal inference failure",
        )


async def run_rim_cnn_inference_async(image_path: str) -> RimCNNPrediction:
    """
    Asynchronously offload Rim CNN inference to a worker thread to prevent blocking main FastAPI loop.
    """
    return await asyncio.to_thread(run_rim_cnn_inference, image_path)