"""Input validation utilities."""

from __future__ import annotations

from typing import Dict, Iterable, List

import pandas as pd

from ..models.account import AccountRecord


class ValidationError(Exception):
    """Custom error for validation related issues."""


def validate_field_mapping(field_map: Dict[str, str], dataframe: pd.DataFrame) -> None:
    """Ensure required fields are present in the provided mapping."""
    required_fields = {"account_id", "balance", "interest_rate"}
    missing = required_fields - field_map.keys()
    if missing:
        raise ValidationError(f"Missing required field mapping(s): {', '.join(missing)}")

    renamed_columns = {target.lower(): source for target, source in field_map.items()}
    for target_field in required_fields:
        source_column = renamed_columns[target_field]
        if source_column not in dataframe.columns:
            raise ValidationError(
                f"Field mapping for {target_field!r} references missing column "
                f"{source_column!r}"
            )


def validate_unique_accounts(accounts: Iterable[AccountRecord]) -> None:
    """Ensure account identifiers are unique."""
    seen = set()
    duplicates: List[str] = []
    for account in accounts:
        if account.account_id in seen:
            duplicates.append(account.account_id)
        else:
            seen.add(account.account_id)
    if duplicates:
        raise ValidationError(
            "Duplicate account_id values detected: " + ", ".join(sorted(set(duplicates)))
        )
