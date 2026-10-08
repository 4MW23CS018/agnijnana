from fastapi import APIRouter, status
from app.schemas.quality import WheelAIOutputContract

router = APIRouter()


@router.post(
    "/inspect",
    response_model=WheelAIOutputContract,
    status_code=status.HTTP_200_OK,
    summary="Inspect aluminium wheel for defects (Placeholder)",
)
def inspect_wheel():
    """
    Submit an aluminium wheel/rim image and metadata for defect inspection.
    TODO — DECISION REQUIRED: Connect to Vijeath's defect detection inference pipeline.
    """
    return {
        "wheel_id": "PENDING-WHEEL-INSPECTION",
        "batch_id": "PENDING",
        "machine_id": "PENDING",
        "defect_type": "none",
        "location": "none",
        "severity": "Low",
        "defect_confidence": 0.0,
        "root_cause": "Pending model integration",
        "root_cause_confidence": 0.0,
        "future_risk": 0.0,
        "recommended_action": "Pending model integration",
        "affected_batches": [],
    }


@router.get(
    "/{wheel_id}",
    response_model=WheelAIOutputContract,
    summary="Get inspection result for a wheel (Placeholder)",
)
def get_inspection(wheel_id: str):
    """
    Retrieve the inspection report for a specific aluminium wheel.
    TODO — DECISION REQUIRED: Integrate with database inspection records.
    """
    return {
        "wheel_id": wheel_id,
        "batch_id": "PENDING",
        "machine_id": "PENDING",
        "defect_type": "none",
        "location": "none",
        "severity": "Low",
        "defect_confidence": 0.0,
        "root_cause": "Pending model integration",
        "root_cause_confidence": 0.0,
        "future_risk": 0.0,
        "recommended_action": "Pending model integration",
        "affected_batches": [],
    }
