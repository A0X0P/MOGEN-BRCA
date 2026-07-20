"""replace the docstring."""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, field_validator


class ImageModality(str, Enum):
    """replace the docstring."""

    CT = "CT"
    MRI = "MRI"
    MAMMOGRAM = "MAMMOGRAM"


class ImageryData(BaseModel):
    """replace the docstring."""

    model_config = ConfigDict(extra="forbid")

    modality: ImageModality
    file_path: str
    is_3d: bool
    shape: tuple[int, ...]
    voxel_spacing: Optional[tuple[float, float, float]] = None
    series_uid: Optional[str] = None

    @field_validator("shape")
    @classmethod
    def validate_shape_dims(cls, dim: tuple[int, ...]) -> tuple[int, ...]:
        """replace the docstring."""

        if len(dim) not in (2, 3):
            raise ValueError("Shape must be 2D or 3D.")
        return dim
