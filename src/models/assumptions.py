"""Assumption data models."""

from __future__ import annotations

from typing import Dict, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class SegmentAssumptions(BaseModel):
    """Assumptions used to drive cash flow projections for a segment."""

    segment_key: str = Field(..., description="Segment identifier")
    decay_rate: float = Field(..., ge=0.0, le=1.0, description="Annual decay rate")
    wal_years: float = Field(
        ...,
        gt=0,
        le=15,
        description="Weighted average life in years",
    )
    beta_up: float = Field(
        ...,
        ge=0.0,
        le=1.5,
        description="Deposit beta applied when rates rise",
    )
    beta_down: float = Field(
        ...,
        ge=0.0,
        le=1.5,
        description="Repricing beta applied when rates fall",
    )
    notes: Optional[str] = Field(
        None, description="Optional commentary about the assumption set"
    )

    @field_validator("segment_key")
    @classmethod
    def _strip_key(cls, value: str) -> str:
        """Ensure the segment key is non-empty."""
        value = value.strip()
        if not value:
            raise ValueError("segment_key cannot be empty")
        return value

    @model_validator(mode="after")
    def _cross_check(self) -> "SegmentAssumptions":
        """Apply qualitative cross-validations."""
        if self.decay_rate > 0.5 and self.wal_years > 5:
            raise ValueError(
                "High decay rate with very long WAL is inconsistent; "
                "consider lowering WAL or decay rate."
            )
        return self

    def monthly_decay_rate(self) -> float:
        """Convert annual decay to monthly equivalent."""
        return 1 - (1 - self.decay_rate) ** (1 / 12)


class AssumptionSet(BaseModel):
    """Collection of assumptions by segment."""

    segmentation_method: str = Field(
        default="all", description="Segmentation method identifier"
    )
    segments: Dict[str, SegmentAssumptions] = Field(
        default_factory=dict,
        description="Mapping of segment id to assumptions",
    )

    def add(self, assumptions: SegmentAssumptions) -> None:
        """Add or replace a segment assumption."""
        self.segments[assumptions.segment_key] = assumptions

    def get(self, segment_key: str) -> SegmentAssumptions:
        """Return assumptions for the requested segment."""
        if segment_key not in self.segments:
            raise KeyError(f"No assumptions defined for {segment_key!r}")
        return self.segments[segment_key]
