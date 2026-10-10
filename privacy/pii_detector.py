"""
PII Detection Engine for AITHOS-FL
Identifies sensitive identifiers, email addresses, phone numbers, and HIPAA-sensitive column headers.
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Any
import pandas as pd


# Default regex pattern for email detection
DEFAULT_EMAIL_PATTERN = r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"

# Configurable regex pattern for standard domestic and international phone numbers
DEFAULT_PHONE_PATTERN = (
    r"(?:\+?\d{1,3}[-.\s]?)?"           # Optional country code
    r"(?:\(?\d{2,4}\)?[-.\s]?)"         # Area code / prefix (e.g. (555), 555-, 020-)
    r"\d{3,4}[-.\s]?\d{3,4}"           # Line numbers
)

# Standard sensitive column identifiers as per HIPAA Safe Harbor & Clinical Specs
DEFAULT_SENSITIVE_COLUMN_NAMES = {
    "patient_name",
    "name",
    "full_name",
    "first_name",
    "last_name",
    "email",
    "email_address",
    "phone",
    "mobile",
    "telephone",
    "phone_number",
    "mobile_number",
    "address",
    "home_address",
    "street_address",
    "patient_id",
    "patientid",
    "medical_record_number",
    "mrn",
}


@dataclass
class PIIScanResult:
    """Encapsulates factual PII scanning statistics for a dataset."""
    total_rows: int = 0
    total_columns: int = 0
    sensitive_columns: list[str] = field(default_factory=list)
    email_matches: int = 0
    phone_matches: int = 0
    findings_by_column: dict[str, dict[str, Any]] = field(default_factory=dict)
    scan_completed: bool = False

    @property
    def total_pii_detections(self) -> int:
        """Total distinct PII pattern matches detected (emails + phones)."""
        return self.email_matches + self.phone_matches

    @property
    def has_pii(self) -> bool:
        """Indicates whether any sensitive column or regex pattern was detected."""
        return (
            len(self.sensitive_columns) > 0
            or self.email_matches > 0
            or self.phone_matches > 0
        )


class PIIDetector:
    """Configurable PII detector for tabular clinical datasets."""

    def __init__(
        self,
        email_pattern: str | None = None,
        phone_pattern: str | None = None,
        sensitive_column_names: set[str] | list[str] | None = None,
    ):
        self.email_pattern_str = email_pattern or DEFAULT_EMAIL_PATTERN
        self.phone_pattern_str = phone_pattern or DEFAULT_PHONE_PATTERN
        
        self.email_regex = re.compile(self.email_pattern_str, re.IGNORECASE)
        self.phone_regex = re.compile(self.phone_pattern_str)
        
        if sensitive_column_names is None:
            self.sensitive_column_names = {s.lower() for s in DEFAULT_SENSITIVE_COLUMN_NAMES}
        else:
            self.sensitive_column_names = {s.strip().lower() for s in sensitive_column_names}

    def normalize_column_name(self, col_name: str) -> str:
        """Normalize column name for robust comparison (e.g., 'Patient Name' -> 'patient_name')."""
        cleaned = re.sub(r"[^\w\s]", "", str(col_name).strip().lower())
        return re.sub(r"\s+", "_", cleaned)

    def is_sensitive_column(self, col_name: str) -> bool:
        """Checks if a column header matches known sensitive attributes."""
        raw_lower = str(col_name).strip().lower()
        if raw_lower in self.sensitive_column_names:
            return True
        norm = self.normalize_column_name(col_name)
        if norm in self.sensitive_column_names:
            return True
        # Check partial/prefix matches for common composites (e.g., patient_id_num)
        for pattern in self.sensitive_column_names:
            if pattern == norm or (len(pattern) > 3 and pattern in norm.split("_")):
                return True
        return False

    def scan_dataframe(self, df: pd.DataFrame) -> PIIScanResult:
        """
        Scans all rows and columns of a DataFrame for sensitive headers and PII patterns.
        Computes exact counts without fabricating any metrics.
        """
        if df is None:
            return PIIScanResult(scan_completed=False)

        total_rows = int(len(df))
        total_cols = int(len(df.columns))

        sensitive_cols: list[str] = []
        for col in df.columns:
            if self.is_sensitive_column(str(col)):
                sensitive_cols.append(str(col))

        total_email_matches = 0
        total_phone_matches = 0
        findings_by_column: dict[str, dict[str, Any]] = {}

        # Scan text-convertible columns
        for col in df.columns:
            col_str = str(col)
            is_sensitive_col = col_str in sensitive_cols
            col_series = df[col]

            col_email_count = 0
            col_phone_count = 0

            # Scan values that are strings or objects
            for val in col_series.dropna():
                val_text = str(val).strip()
                if not val_text:
                    continue

                # Scan emails
                email_matches = self.email_regex.findall(val_text)
                if email_matches:
                    col_email_count += len(email_matches)

                # Scan phones
                phone_matches = self.phone_regex.findall(val_text)
                if phone_matches:
                    # Filter out short numeric false positives (e.g. simple 2 or 3 digit integers)
                    valid_phones = [p for p in phone_matches if sum(c.isdigit() for c in p) >= 7]
                    col_phone_count += len(valid_phones)

            total_email_matches += col_email_count
            total_phone_matches += col_phone_count

            findings_by_column[col_str] = {
                "is_sensitive_column": is_sensitive_col,
                "email_count": col_email_count,
                "phone_count": col_phone_count,
            }

        return PIIScanResult(
            total_rows=total_rows,
            total_columns=total_cols,
            sensitive_columns=sensitive_cols,
            email_matches=total_email_matches,
            phone_matches=total_phone_matches,
            findings_by_column=findings_by_column,
            scan_completed=True,
        )
