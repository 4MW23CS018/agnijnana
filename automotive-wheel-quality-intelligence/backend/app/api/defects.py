from typing import List
from fastapi import APIRouter, Query
from app.schemas.quality import WheelAIOutputContract

router = APIRouter()


@router.get(
    "/",
    response_model=List[WheelAIOutputContract],
    summary="List wheel defects (Placeholder)",
)
def list_defects(
    batch_id: str = Query(None, description="Filter by casting batch"),
    machine_id: str = Query(None, description="Filter by machine station"),
    severity: str = Query(None, description="Filter by severity ('Low' | 'Critical')"),
):
    """
    Retrieve detected defects on aluminium wheels/rims.
    TODO — DECISION REQUIRED: Connect to PostgreSQL defects table.
    """
    return []
