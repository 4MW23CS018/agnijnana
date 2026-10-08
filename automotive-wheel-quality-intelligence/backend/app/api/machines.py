from fastapi import APIRouter

router = APIRouter()


@router.get("/", summary="List casting/machining equipment (Placeholder)")
def get_machines():
    """
    TODO — DECISION REQUIRED: Integrate with database machine records.
    """
    return []
