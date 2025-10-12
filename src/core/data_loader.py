"""Data ingestion routines for the ALM engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from ..models.account import AccountRecord
from .validator import ValidationError, validate_field_mapping, validate_unique_accounts


@dataclass
class LoadResult:
    """Represents the outcome of a data load operation."""

    accounts: List[AccountRecord]
    dataframe: pd.DataFrame
    field_map: Dict[str, str]
    optional_fields: Sequence[str]


class DataLoader:
    """Load deposit account data and apply user-defined field mappings."""

    def __init__(self, preserve_extra_columns: bool = True) -> None:
        self.preserve_extra_columns = preserve_extra_columns

    def load_accounts(
        self,
        file_path: str,
        field_map: Dict[str, str],
        optional_fields: Optional[Iterable[str]] = None,
        dtypes: Optional[Dict[str, str]] = None,
    ) -> LoadResult:
        """Load account data from CSV applying the user supplied field mapping."""
        dataframe = self._read_csv(file_path, dtypes=dtypes)
        validate_field_mapping(field_map, dataframe)

        optional_fields = tuple(optional_fields or ())
        mapped_df = self._apply_mapping(dataframe, field_map)
        accounts = self._build_accounts(mapped_df, optional_fields)
        validate_unique_accounts(accounts)

        return LoadResult(
            accounts=accounts,
            dataframe=mapped_df,
            field_map=field_map,
            optional_fields=optional_fields,
        )

    def load_accounts_from_dataframe(
        self,
        dataframe: pd.DataFrame,
        field_map: Dict[str, str],
        optional_fields: Optional[Iterable[str]] = None,
    ) -> LoadResult:
        """Create account records from an in-memory dataframe."""
        validate_field_mapping(field_map, dataframe)
        optional_fields = tuple(optional_fields or ())
        mapped_df = self._apply_mapping(dataframe.copy(), field_map)
        accounts = self._build_accounts(mapped_df, optional_fields)
        validate_unique_accounts(accounts)
        return LoadResult(
            accounts=accounts,
            dataframe=mapped_df,
            field_map=field_map,
            optional_fields=optional_fields,
        )

    def _read_csv(
        self, file_path: str, dtypes: Optional[Dict[str, str]] = None
    ) -> pd.DataFrame:
        """Read the CSV file into a dataframe."""
        try:
            return pd.read_csv(file_path, dtype=dtypes)
        except FileNotFoundError as exc:
            raise ValidationError(f"File not found: {file_path}") from exc
        except pd.errors.ParserError as exc:
            raise ValidationError(f"Unable to parse CSV: {exc}") from exc

    def _apply_mapping(
        self, dataframe: pd.DataFrame, field_map: Dict[str, str]
    ) -> pd.DataFrame:
        """Rename dataframe columns according to the canonical schema."""
        rename_map = {source: target for target, source in field_map.items()}
        mapped_df = dataframe.rename(columns=rename_map)
        required_columns = ["account_id", "balance", "interest_rate"]
        for column in required_columns:
            if column not in mapped_df.columns:
                raise ValidationError(
                    f"Column {column!r} missing after applying field mapping"
                )
        return mapped_df

    def _build_accounts(
        self, dataframe: pd.DataFrame, optional_fields: Sequence[str]
    ) -> List[AccountRecord]:
        """Convert dataframe rows into AccountRecord models."""
        accounts: List[AccountRecord] = []
        for _, row in dataframe.iterrows():
            metadata = {}
            if self.preserve_extra_columns:
                metadata = {
                    column: row[column]
                    for column in dataframe.columns
                    if column
                    not in {"account_id", "balance", "interest_rate"}
                    and column not in optional_fields
                }
            account_kwargs = {
                "account_id": row["account_id"],
                "balance": row["balance"],
                "interest_rate": row["interest_rate"],
            }
            if "account_type" in optional_fields and "account_type" in dataframe.columns:
                account_kwargs["account_type"] = row.get("account_type")
            if "customer_segment" in optional_fields and "customer_segment" in dataframe.columns:
                account_kwargs["customer_segment"] = row.get("customer_segment")
            if "rate_type" in optional_fields and "rate_type" in dataframe.columns:
                account_kwargs["rate_type"] = row.get("rate_type")
            if metadata:
                account_kwargs["metadata"] = metadata
            account = AccountRecord(**account_kwargs)
            accounts.append(account)
        return accounts
