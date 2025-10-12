"""Account level data model definitions."""

from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, field_validator


class AccountRecord(BaseModel):
    """Represents a single non-maturity deposit account entry."""

    account_id: str = Field(..., description="Unique account identifier")
    balance: float = Field(..., gt=0, description="Current balance")
    interest_rate: float = Field(
        ...,
        ge=0,
        le=0.15,
        description="Current interest rate (decimal form, e.g. 0.035)",
    )
    account_type: Optional[str] = Field(
        None, description="Optional product type for segmentation"
    )
    customer_segment: Optional[str] = Field(
        None, description="Optional customer segment for segmentation"
    )
    rate_type: Optional[str] = Field(None, description="Optional rate type metadata")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional columns preserved for downstream reporting",
    )

    @field_validator("account_id", mode="before")
    @classmethod
    def _coerce_account_id(cls, value: Any) -> str:
        """Ensure the account id is a trimmed string."""
        if value is None:
            raise ValueError("account_id cannot be null")
        return str(value).strip()

    @field_validator("account_type", "customer_segment", "rate_type", mode="before")
    @classmethod
    def _normalize_optional_str(cls, value: Optional[Any]) -> Optional[str]:
        """Normalize optional string fields."""
        if value is None:
            return None
        value = str(value).strip()
        return value or None

    @field_validator("balance")
    @classmethod
    def _validate_balance(cls, value: float) -> float:
        """Balance must be positive."""
        if value <= 0:
            raise ValueError("balance must be positive")
        return float(value)

    @field_validator("interest_rate")
    @classmethod
    def _validate_rate(cls, value: float) -> float:
        """Interest rate must be in reasonable range."""
        if not 0 <= value <= 0.15:
            raise ValueError("interest_rate must be between 0 and 0.15")
        return float(value)

    def key(self, segmentation_key: Optional[str] = None) -> str:
        """Return the segment key that this account belongs to."""
        if segmentation_key == "account_type":
            if not self.account_type:
                raise ValueError("Account type segmentation requires account_type value")
            return self.account_type
        if segmentation_key == "customer_segment":
            if not self.customer_segment:
                raise ValueError(
                    "Customer segment segmentation requires customer_segment value"
                )
            return self.customer_segment
        if segmentation_key == "account_type_customer_segment":
            if not self.account_type or not self.customer_segment:
                raise ValueError(
                    "Cross segmentation requires both account_type and customer_segment"
                )
            return f"{self.account_type}::{self.customer_segment}"
        return "ALL"


AccountCollection = Dict[str, AccountRecord]
