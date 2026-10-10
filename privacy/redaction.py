"""
Redaction and Sanitization Engine for AITHOS-FL
Generates a sanitized copy of datasets by masking detected emails, phone numbers,
and redacting sensitive column attributes without modifying the original data.
"""

from __future__ import annotations
from typing import Any
import pandas as pd
from privacy.pii_detector import PIIDetector, PIIScanResult


def redact_dataframe(
    df: pd.DataFrame,
    detector: PIIDetector | None = None,
    scan_result: PIIScanResult | None = None,
    redact_token: str = "[REDACTED]",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Produces a sanitized copy of the input DataFrame.
    
    Guarantees:
    - Never mutates the source DataFrame (creates an explicit deep copy).
    - Entire sensitive columns (e.g., patient_name, medical_record_number) have their
      non-null values replaced with `[REDACTED]`.
    - Text fields in non-sensitive columns have detected email addresses and phone
      numbers masked inline with `[REDACTED]`.
    
    Returns:
        (sanitized_df, summary_dict)
    """
    if df is None:
        return pd.DataFrame(), {"error": "DataFrame is None"}

    # Strict defensive deep copy to ensure the original upload is NEVER overwritten
    sanitized_df = df.copy(deep=True)

    if detector is None:
        detector = PIIDetector()

    if scan_result is None:
        scan_result = detector.scan_dataframe(df)

    sensitive_cols = set(scan_result.sensitive_columns)
    total_cells_redacted = 0
    total_inline_masks = 0

    for col in sanitized_df.columns:
        col_str = str(col)

        if col_str in sensitive_cols:
            # Sensitive column: replace non-null values with [REDACTED]
            mask = sanitized_df[col].notna()
            non_null_count = int(mask.sum())
            sanitized_df[col] = sanitized_df[col].astype(object)
            sanitized_df.loc[mask, col] = redact_token
            total_cells_redacted += non_null_count
        else:
            # Check for inline text PII (emails or phone numbers in unstructured text columns)
            col_findings = scan_result.findings_by_column.get(col_str, {})
            has_inline_pii = (
                col_findings.get("email_count", 0) > 0
                or col_findings.get("phone_count", 0) > 0
            )

            if has_inline_pii:
                sanitized_df[col] = sanitized_df[col].astype(object)

                def mask_text_value(val: Any) -> Any:
                    nonlocal total_inline_masks
                    if pd.isna(val):
                        return val
                    text = str(val)
                    original_text = text

                    # Mask emails
                    if detector.email_regex.search(text):
                        text = detector.email_regex.sub(redact_token, text)

                    # Mask phones (filtering numbers with sufficient digits)
                    def phone_replacer(match: Any) -> str:
                        matched_str = match.group(0)
                        if sum(c.isdigit() for c in matched_str) >= 7:
                            return redact_token
                        return matched_str

                    text = detector.phone_regex.sub(phone_replacer, text)

                    if text != original_text:
                        total_inline_masks += 1
                    return text

                sanitized_df[col] = sanitized_df[col].apply(mask_text_value)

    summary = {
        "sensitive_columns_redacted": list(sensitive_cols),
        "sensitive_column_cells_redacted": total_cells_redacted,
        "inline_text_masks": total_inline_masks,
        "total_redactions": total_cells_redacted + total_inline_masks,
    }

    return sanitized_df, summary
