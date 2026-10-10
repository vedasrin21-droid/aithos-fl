"""
Unit tests for PII Detection module (privacy/pii_detector.py).
Tests regex matching, sensitive column detection, and factual count accuracy.
"""

import pandas as pd
import pytest
from privacy.pii_detector import PIIDetector, PIIScanResult


def test_detect_emails_and_phones():
    detector = PIIDetector()
    
    df = pd.DataFrame({
        "notes": [
            "Contact me at alice@hospital.org or test.user+fl@health.co.uk",
            "Call emergency number 555-123-4567 or (800) 555-0199",
            "No PII here, just normal clinical metrics",
        ],
        "metric_val": [10.5, 20.3, 30.1],
    })

    result = detector.scan_dataframe(df)

    assert result.scan_completed is True
    assert result.total_rows == 3
    assert result.total_columns == 2
    assert result.email_matches == 2
    assert result.phone_matches >= 2
    assert result.has_pii is True


def test_detect_sensitive_column_names():
    detector = PIIDetector()

    df = pd.DataFrame({
        "patient_id": [101, 102],
        "Patient Name": ["Jane Doe", "John Roe"],
        "medical_record_number": ["MRN-1", "MRN-2"],
        "email": ["jane@demo.test", "john@demo.test"],
        "phone": ["(555) 123-4567", "555-987-6543"],
        "address": ["123 Main St", "456 Elm St"],
        "blood_pressure": ["120/80", "130/85"],
    })

    result = detector.scan_dataframe(df)

    assert result.scan_completed is True
    assert "patient_id" in result.sensitive_columns
    assert "Patient Name" in result.sensitive_columns
    assert "medical_record_number" in result.sensitive_columns
    assert "email" in result.sensitive_columns
    assert "phone" in result.sensitive_columns
    assert "address" in result.sensitive_columns
    assert "blood_pressure" not in result.sensitive_columns
    assert len(result.sensitive_columns) == 6


def test_clean_dataset_has_zero_pii():
    detector = PIIDetector()

    df = pd.DataFrame({
        "feature_1": [1.2, 3.4, 5.6],
        "feature_2": [10, 20, 30],
        "cohort": ["A", "B", "A"],
    })

    result = detector.scan_dataframe(df)

    assert result.scan_completed is True
    assert result.total_rows == 3
    assert len(result.sensitive_columns) == 0
    assert result.email_matches == 0
    assert result.phone_matches == 0
    assert result.has_pii is False
