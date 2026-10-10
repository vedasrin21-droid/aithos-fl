"""
Unit tests for access control and hospital isolation (privacy/dataset_store.py).
Verifies that:
- Hospital A data is kept separate from Hospital B data.
- Normal users cannot upload or view hospital-private datasets.
- Cross-hospital access is strictly forbidden.
- Malformed and invalid files are rejected.
"""

import pytest
import pandas as pd
from privacy.dataset_store import HospitalDatasetStore
from privacy.synthetic_data import generate_synthetic_csv_bytes


@pytest.fixture
def store():
    return HospitalDatasetStore()


@pytest.fixture
def hospital_a_user():
    return {
        "email": "hospital.a@demo.test",
        "role": "Hospital Node Administrator",
        "hospital_id": "HOSP-A",
    }


@pytest.fixture
def hospital_b_user():
    return {
        "email": "hospital.b@demo.test",
        "role": "Hospital Node Administrator",
        "hospital_id": "HOSP-B",
    }


@pytest.fixture
def normal_user():
    return {
        "email": "user@demo.test",
        "role": "Clinical Research Scientist",
        "hospital_id": "Unassigned (Independent / External)",
    }


def test_normal_user_cannot_upload_dataset(store, normal_user):
    filename, csv_bytes = generate_synthetic_csv_bytes("HOSP-A")
    with pytest.raises(PermissionError) as exc_info:
        store.ingest_dataset(normal_user, filename, csv_bytes)
    assert "not authorized to upload" in str(exc_info.value)


def test_normal_user_cannot_view_datasets(store, normal_user):
    with pytest.raises(PermissionError) as exc_info:
        store.get_datasets_for_user(normal_user)
    assert "not authorized to view" in str(exc_info.value)


def test_hospital_a_and_b_data_separation(store, hospital_a_user, hospital_b_user):
    filename_a, bytes_a = generate_synthetic_csv_bytes("HOSP-A")
    filename_b, bytes_b = generate_synthetic_csv_bytes("HOSP-B")

    ds_a = store.ingest_dataset(hospital_a_user, filename_a, bytes_a)
    ds_b = store.ingest_dataset(hospital_b_user, filename_b, bytes_b)

    # Hospital A should only see dataset A
    list_a = store.get_datasets_for_user(hospital_a_user)
    assert len(list_a) == 1
    assert list_a[0].id == ds_a.id
    assert list_a[0].hospital_id == "HOSP-A"

    # Hospital B should only see dataset B
    list_b = store.get_datasets_for_user(hospital_b_user)
    assert len(list_b) == 1
    assert list_b[0].id == ds_b.id
    assert list_b[0].hospital_id == "HOSP-B"

    # Cross-hospital access attempt by Hospital B on Hospital A's dataset ID
    with pytest.raises(PermissionError) as exc_info:
        store.get_dataset_by_id(hospital_b_user, ds_a.id)
    assert "Cross-Hospital Access Violation" in str(exc_info.value)


def test_file_validation_rejects_non_csv(store, hospital_a_user):
    with pytest.raises(ValueError) as exc:
        store.ingest_dataset(hospital_a_user, "data.xlsx", b"some bytes")
    assert "Only .csv files are supported" in str(exc.value)


def test_file_validation_rejects_empty_file(store, hospital_a_user):
    with pytest.raises(ValueError) as exc:
        store.ingest_dataset(hospital_a_user, "empty.csv", b"")
    assert "empty (0 bytes)" in str(exc.value)


def test_file_validation_rejects_malformed_csv(store, hospital_a_user):
    # Malformed data with no rows
    with pytest.raises(ValueError) as exc:
        store.ingest_dataset(hospital_a_user, "bad.csv", b"col1,col2,col3\n")
    assert "zero data rows" in str(exc.value)
