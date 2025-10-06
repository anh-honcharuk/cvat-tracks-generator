from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class SahiConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SAHI_", env_file=".env", env_file_encoding="utf-8")
    slice_height: int = Field(default=640, ge=64)
    slice_width: int = Field(default=640, ge=64)
    overlap_height_ratio: float = Field(default=0.2, ge=0.0, le=0.9)
    overlap_width_ratio: float = Field(default=0.2, ge=0.0, le=0.9)
    conf: float = Field(default=0.25, ge=0.0, le=1.0)
    iou_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    max_age: int = Field(default=30, ge=0)


sahi_cfg = SahiConfig()
