from fastapi import APIRouter

router = APIRouter()


@router.get("/", summary="List casting batches (Placeholder)")
def get_batches():
    """
    TODO — DECISION REQUIRED: Integrate with database casting batch records.
    """
    return []
