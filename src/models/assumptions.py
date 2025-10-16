"""Assumption data models."""

from __future__ import annotations

from typing import Dict, Optional

from pydantic import BaseModel, Field, PrivateAttr, field_validator, model_validator

from ..core.decay import DecayResolution, resolve_decay_parameters


class SegmentAssumptions(BaseModel):
    """Assumptions used to drive cash flow projections for a segment."""

    segment_key: str = Field(..., description="Segment identifier")
    decay_rate: Optional[float] = Field(
        None, ge=0.0, lt=1.0, description="Annual decay rate input (decimal form)."
    )
    wal_years: Optional[float] = Field(
        None,
        gt=0,
        le=50,
        description="Weighted average life input in years.",
    )
    deposit_beta_up: float = Field(
        ...,
        ge=0.0,
        le=1.5,
        description="Deposit beta applied when market rates rise.",
    )
    deposit_beta_down: float = Field(
        ...,
        ge=0.0,
        le=1.5,
        description="Deposit beta applied when market rates fall.",
    )
    repricing_beta_up: float = Field(
        default=1.0,
        ge=0.0,
        le=2.0,
        description="Repricing beta applied to market shocks when rates rise.",
    )
    repricing_beta_down: float = Field(
        default=1.0,
        ge=0.0,
        le=2.0,
        description="Repricing beta applied to market shocks when rates fall.",
    )
    notes: Optional[str] = Field(
        None, description="Optional commentary about the assumption set."
    )
    decay_priority: str = Field(
        default="auto",
        description=(
            "Which decay input to prioritise when both WAL and decay are supplied. "
            "Options: 'wal', 'decay', or 'auto'."
        ),
    )
    resolved_decay_rate: float = Field(
        0.0,
        ge=0.0,
        lt=1.0,
        description="Resolved annual decay rate used for projections.",
    )
    resolved_monthly_decay_rate: float = Field(
        0.0,
        ge=0.0,
        lt=1.0,
        description="Resolved monthly decay rate used for projections.",
    )
    resolved_wal_years: float = Field(
        0.0,
        ge=0.0,
        description="Resolved weighted-average life in years used for projections.",
    )
    decay_input_mode: str = Field(
        default="unspecified",
        description="Indicates whether WAL, decay, or both inputs were supplied.",
    )
    decay_warning: Optional[str] = Field(
        default=None, description="Warning message when inputs are inconsistent."
    )
    implied_wal_from_decay: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Implied WAL (years) from the provided decay rate.",
    )
    implied_decay_from_wal: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Implied annual decay from the provided WAL.",
    )
    decay_difference_years: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Absolute WAL difference (years) when both inputs were supplied.",
    )
    decay_difference_ratio: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Relative WAL difference versus the WAL input when both inputs were supplied.",
    )

    _input_wal_years: Optional[float] = PrivateAttr(default=None)
    _input_decay_rate: Optional[float] = PrivateAttr(default=None)

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
    def _resolve_decay(self) -> "SegmentAssumptions":
        """Resolve WAL/decay inputs into a consistent set of parameters."""
        wal_input = self.wal_years
        decay_input = self.decay_rate

        resolution: DecayResolution = resolve_decay_parameters(
            wal_input,
            decay_input,
            priority=self.decay_priority,
        )

        if (
            self.decay_priority.lower().strip() == "auto"
            and resolution.inconsistent
            and resolution.warning
            and resolution.priority_used == "wal"
        ):
            raise ValueError(
                f"Inconsistent WAL/decay inputs for segment '{self.segment_key}'. "
                f"{resolution.warning} Specify decay_priority='wal' or decay_priority='decay'."
            )

        self._input_wal_years = wal_input
        self._input_decay_rate = decay_input

        object.__setattr__(self, "resolved_wal_years", resolution.wal_years)
        object.__setattr__(self, "resolved_decay_rate", resolution.annual_decay_rate)
        object.__setattr__(self, "resolved_monthly_decay_rate", resolution.monthly_decay_rate)
        object.__setattr__(self, "decay_input_mode", resolution.mode)
        object.__setattr__(self, "decay_priority", resolution.priority_used)
        object.__setattr__(self, "decay_warning", resolution.warning)
        object.__setattr__(self, "implied_wal_from_decay", resolution.implied_wal_from_decay)
        object.__setattr__(self, "implied_decay_from_wal", resolution.implied_decay_from_wal)
        object.__setattr__(self, "decay_difference_years", resolution.difference_years)
        object.__setattr__(self, "decay_difference_ratio", resolution.difference_ratio)

        object.__setattr__(self, "wal_years", resolution.wal_years)
        object.__setattr__(self, "decay_rate", resolution.annual_decay_rate)

        if (
            self.decay_rate is not None
            and self.decay_rate > 0.5
            and self.wal_years is not None
            and self.wal_years > 5
        ):
            raise ValueError(
                "High decay rate with very long WAL is inconsistent; "
                "consider lowering WAL or decay rate."
            )
        return self

    def monthly_decay_rate(self) -> float:
        """Return the resolved monthly decay rate used for projections."""
        return self.resolved_monthly_decay_rate

    @property
    def input_wal_years(self) -> Optional[float]:
        """Return the WAL provided by the user (if any)."""
        return self._input_wal_years

    @property
    def input_decay_rate(self) -> Optional[float]:
        """Return the decay rate provided by the user (if any)."""
        return self._input_decay_rate


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

