"""replace the docstring."""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Sex(str, Enum):
    """replace the docstring."""

    MALE = "M"
    FEMALE = "F"


class ClinicalData(BaseModel):
    """replace the docstring."""

    model_config = ConfigDict(extra="forbid")

    age: int = Field(..., ge=0, le=120)
    sex: Sex
    weight: int
    height: int
    blood_pressure: Optional[str] = None
    stage: str
    smoking_history: Optional[bool] = None
    bmi: Optional[float] = Field(None, ge=10, le=60)
    comorbidities: list[str] = Field(default_factory=list)

    @field_validator("stage")
    @classmethod
    def validate_stage_format(cls, v: str) -> str:
        """replace the docstring."""
        return v.upper()
