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
    deposit_beta_up: float = Field(
        ...,
        ge=0.0,
        le=1.5,
        description="Deposit beta applied when market rates rise",
    )
    deposit_beta_down: float = Field(
        ...,
        ge=0.0,
        le=1.5,
        description="Deposit beta applied when market rates fall",
    )
    repricing_beta_up: float = Field(
        default=1.0,
        ge=0.0,
        le=2.0,
        description="Repricing beta applied to market shocks when rates rise",
    )
    repricing_beta_down: float = Field(
        default=1.0,
        ge=0.0,
        le=2.0,
        description="Repricing beta applied to market shocks when rates fall",
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

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_fields(cls, values: Dict[str, object]) -> Dict[str, object]:
        """Support legacy beta fields by upgrading them to the new schema."""
        if isinstance(values, dict):
            beta_up = values.get("beta_up")
            beta_down = values.get("beta_down")
            if "deposit_beta_up" not in values and beta_up is not None:
                values["deposit_beta_up"] = beta_up
            if "deposit_beta_down" not in values and beta_down is not None:
                values["deposit_beta_down"] = beta_down
            # Default repricing betas to 1.0 (full pass-through) if not provided.
            values.setdefault("repricing_beta_up", 1.0)
            values.setdefault("repricing_beta_down", 1.0)
        return values

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
