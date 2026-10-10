"""
Unit tests for Redaction module (privacy/redaction.py).
Tests non-destructive masking, sensitive column redaction, and summary calculations.
"""

import pandas as pd
from privacy.pii_detector import PIIDetector
from privacy.redaction import redact_dataframe


def test_redaction_does_not_mutate_original_dataframe():
    df = pd.DataFrame({
        "patient_name": ["Alice Smith", "Bob Jones"],
        "medical_record_number": ["MRN-101", "MRN-102"],
        "vitals": [98.6, 99.1],
    })
    original_copy = df.copy(deep=True)

    sanitized_df, summary = redact_dataframe(df)

    # Verify original df was not modified
    pd.testing.assert_frame_equal(df, original_copy)

    # Verify sanitized copy has redacted sensitive columns
    assert all(sanitized_df["patient_name"] == "[REDACTED]")
    assert all(sanitized_df["medical_record_number"] == "[REDACTED]")
    assert list(sanitized_df["vitals"]) == [98.6, 99.1]
    assert summary["sensitive_column_cells_redacted"] == 4


def test_inline_text_pii_masking():
    df = pd.DataFrame({
        "clinical_notes": [
            "Contact Dr. Lee at lee@demo-health.org for records.",
            "Call emergency clinic at 555-432-8765 immediately.",
            "Regular follow-up in two weeks.",
        ],
        "dosage_mg": [50, 100, 25],
    })

    sanitized_df, summary = redact_dataframe(df)

    # Ensure sensitive items inside clinical_notes are replaced with [REDACTED]
    assert "lee@demo-health.org" not in sanitized_df.loc[0, "clinical_notes"]
    assert "[REDACTED]" in sanitized_df.loc[0, "clinical_notes"]

    assert "555-432-8765" not in sanitized_df.loc[1, "clinical_notes"]
    assert "[REDACTED]" in sanitized_df.loc[1, "clinical_notes"]

    # Clean text remains unchanged
    assert sanitized_df.loc[2, "clinical_notes"] == "Regular follow-up in two weeks."
    # Numeric column remains unchanged
    assert list(sanitized_df["dosage_mg"]) == [50, 100, 25]
    assert summary["inline_text_masks"] >= 2
