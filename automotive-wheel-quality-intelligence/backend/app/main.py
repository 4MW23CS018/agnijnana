from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.core.config import settings
from app.schemas.quality import HealthResponse
from app.api import (
    inspection,
    defects,
    root_cause,
    risk,
    recommendations,
    alerts,
    batches,
    machines,
)

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Industrial Quality Intelligence System REST API for Aluminium Alloy Wheels — Singularity 2026",
    version="0.1.0",
)

# Mount static uploads directory for serving uploaded wheel images
PROJECT_ROOT = Path(__file__).resolve().parents[2]
UPLOADS_PATH = PROJECT_ROOT / "data" / "uploads"
UPLOADS_PATH.mkdir(parents=True, exist_ok=True)
app.mount("/data/uploads", StaticFiles(directory=str(UPLOADS_PATH)), name="uploads")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Service Health Check",
)
def health_check():
    """Verify backend API readiness and operational health."""
    return HealthResponse(
        status="ok",
        service="Aluminium Wheel Quality Intelligence Backend",
        version="0.1.0",
    )


# Include API routers
app.include_router(inspection.router, prefix="/api/inspection", tags=["Wheel Inspection"])
app.include_router(defects.router, prefix="/api/defects", tags=["Defects"])
app.include_router(root_cause.router, prefix="/api/root-cause", tags=["Root Cause"])
app.include_router(risk.router, prefix="/api/risk", tags=["Risk Prediction"])
app.include_router(recommendations.router, prefix="/api/recommendations", tags=["Recommendations"])
app.include_router(alerts.router, prefix="/api/alerts", tags=["Alerts"])
app.include_router(batches.router, prefix="/api/batches", tags=["Casting Batches"])
app.include_router(machines.router, prefix="/api/machines", tags=["Casting Machines"])
