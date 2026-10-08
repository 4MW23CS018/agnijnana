from fastapi import APIRouter

router = APIRouter()


@router.get("/", summary="List active quality alerts (Placeholder)")
def get_alerts():
    """
    TODO — DECISION REQUIRED: Integrate with database alerts table.
    """
    return []
