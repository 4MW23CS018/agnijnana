from typing import List, Optional, Union
from pydantic import BaseModel, Field


class WheelAIOutputContract(BaseModel):
    """
    Common Wheel AI Output Contract.
    Provisional — fields will be finalized after Vijeath verifies the physical dataset.
    """
    wheel_id: str = Field(..., description="Unique wheel/rim tracking ID")
    batch_id: str = Field(..., description="Casting batch or lot number")
    machine_id: str = Field(..., description="Die-casting machine or CNC station ID")
    defect_type: str = Field(..., description="Defect category from dataset (do NOT hard-code)")
    location: Union[str, List[float]] = Field(..., description="Bounding box [x, y, w, h] or spatial reference on wheel")
    severity: str = Field(..., description="'Low' or 'Critical'")
    defect_confidence: float = Field(0.0, ge=0.0, le=1.0, description="Detection model confidence")
    root_cause: str = Field(..., description="Identified manufacturing root cause")
    root_cause_confidence: float = Field(0.0, ge=0.0, le=1.0, description="Root cause model confidence")
    future_risk: float = Field(0.0, ge=0.0, le=1.0, description="Predicted probability of subsequent defects")
    recommended_action: str = Field(..., description="Recommended corrective action")
    affected_batches: List[str] = Field(default_factory=list, description="List of casting batches at risk")


class DefectQuery(BaseModel):
    batch_id: Optional[str] = None
    machine_id: Optional[str] = None
    severity: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    service: str = "Aluminium Wheel Quality Intelligence Backend"
    version: str = "0.1.0"
