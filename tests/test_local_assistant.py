"""
Unit tests for Local AI Assistant (local_ai/assistant.py).
Verifies:
- Hospital A queries against its own uploaded dataset.
- Hospital B queries against its own uploaded dataset.
- Cross-hospital access prevention (Hospital B cannot retrieve Hospital A records).
- Normal User access denial.
- Missing patient IDs and unsupported query handling without hallucination.
"""

import pytest
from privacy.dataset_store import HospitalDatasetStore
from privacy.synthetic_data import generate_synthetic_csv_bytes
from local_ai.assistant import LocalAIAssistant, LOCAL_STATUS_LABEL


@pytest.fixture
def store():
    store_obj = HospitalDatasetStore()
    
    # Ingest Hospital A demo dataset
    hosp_a_user = {
        "email": "hospital.a@demo.test",
        "role": "Hospital Node Administrator",
        "hospital_id": "HOSP-A",
        "name": "Hospital A Medical Center",
    }
    fn_a, bytes_a = generate_synthetic_csv_bytes("HOSP-A")
    store_obj.ingest_dataset(hosp_a_user, fn_a, bytes_a)

    # Ingest Hospital B demo dataset
    hosp_b_user = {
        "email": "hospital.b@demo.test",
        "role": "Hospital Node Administrator",
        "hospital_id": "HOSP-B",
        "name": "Hospital B Regional Health",
    }
    fn_b, bytes_b = generate_synthetic_csv_bytes("HOSP-B")
    store_obj.ingest_dataset(hosp_b_user, fn_b, bytes_b)

    return store_obj


@pytest.fixture
def assistant(store):
    return LocalAIAssistant(store)


@pytest.fixture
def hospital_a_user():
    return {
        "email": "hospital.a@demo.test",
        "role": "Hospital Node Administrator",
        "hospital_id": "HOSP-A",
        "name": "Hospital A Medical Center",
    }


@pytest.fixture
def hospital_b_user():
    return {
        "email": "hospital.b@demo.test",
        "role": "Hospital Node Administrator",
        "hospital_id": "HOSP-B",
        "name": "Hospital B Regional Health",
    }


@pytest.fixture
def normal_user():
    return {
        "email": "user@demo.test",
        "role": "Clinical Research Scientist",
        "hospital_id": "Unassigned (Independent / External)",
        "name": "Standard Research User",
    }


def test_hospital_a_patient_count(assistant, hospital_a_user):
    resp = assistant.answer_query(hospital_a_user, "How many patients are in my uploaded dataset?")
    assert resp.success is True
    assert "5 unique patients" in resp.answer
    assert resp.status_label == LOCAL_STATUS_LABEL


def test_hospital_a_top_department(assistant, hospital_a_user):
    resp = assistant.answer_query(hospital_a_user, "Which department has the most records?")
    assert resp.success is True
    assert "Cardiology" in resp.answer
    assert "3 records" in resp.answer


def test_hospital_a_patient_email(assistant, hospital_a_user):
    resp = assistant.answer_query(hospital_a_user, "What email address is recorded for patient P001?")
    assert resp.success is True
    assert "eleanor.vance@synthetic-health.org" in resp.answer
    assert "P001" in resp.answer


def test_hospital_a_patient_phone(assistant, hospital_a_user):
    resp = assistant.answer_query(hospital_a_user, "Show the phone number for patient P002.")
    assert resp.success is True
    assert "+1-555-432-8765" in resp.answer
    assert "P002" in resp.answer


def test_hospital_a_appointment_policy(assistant, hospital_a_user):
    resp = assistant.answer_query(hospital_a_user, "What is the appointment rescheduling policy for patient P001?")
    assert resp.success is True
    assert "24 hours" in resp.answer
    assert "Hospital A" in resp.answer


def test_hospital_b_own_records(assistant, hospital_b_user):
    # Hospital B queries its own records
    resp_count = assistant.answer_query(hospital_b_user, "How many patients are in my uploaded dataset?")
    assert resp_count.success is True
    assert "4 unique patients" in resp_count.answer

    resp_dept = assistant.answer_query(hospital_b_user, "Which department has the most records?")
    assert resp_dept.success is True
    assert "Oncology" in resp_dept.answer

    resp_p101 = assistant.answer_query(hospital_b_user, "What email address is recorded for patient P101?")
    assert resp_p101.success is True
    assert "devon.alexander@regional-care.test" in resp_p101.answer


def test_cross_hospital_isolation(assistant, hospital_b_user, hospital_a_user):
    # Hospital B asks for patient P001 (which belongs to Hospital A)
    resp = assistant.answer_query(hospital_b_user, "What email address is recorded for patient P001?")
    assert resp.success is False
    assert "Patient **P001** was not found in **HOSP-B**'s uploaded dataset" in resp.answer
    assert "eleanor.vance" not in resp.answer

    # Hospital A asks for patient P101 (which belongs to Hospital B)
    resp_a = assistant.answer_query(hospital_a_user, "What email address is recorded for patient P101?")
    assert resp_a.success is False
    assert "Patient **P101** was not found in **HOSP-A**'s uploaded dataset" in resp_a.answer
    assert "devon.alexander" not in resp_a.answer


def test_normal_user_access_denied(assistant, normal_user):
    # Backend access check must reject normal users with PermissionError
    with pytest.raises(PermissionError) as exc_info:
        assistant.answer_query(normal_user, "How many patients are in my uploaded dataset?")
    assert "not authorized to query private hospital datasets" in str(exc_info.value)


def test_missing_patient_id(assistant, hospital_a_user):
    resp = assistant.answer_query(hospital_a_user, "Show the phone number for patient P999.")
    assert resp.success is False
    assert "Patient **P999** was not found" in resp.answer


def test_unsupported_question(assistant, hospital_a_user):
    resp = assistant.answer_query(hospital_a_user, "What is the stock market forecast for tomorrow?")
    assert resp.success is False
    assert "could not find an exact match" in resp.answer
    assert "Supported query examples" in resp.answer


def test_cross_hospital_dataset_id_raises_permission_error(assistant, store, hospital_a_user, hospital_b_user):
    # Hospital A dataset ID
    hosp_a_datasets = store.get_datasets_for_user(hospital_a_user)
    assert len(hosp_a_datasets) >= 1
    hosp_a_id = hosp_a_datasets[0].id

    # Hospital B attempting to pass Hospital A's dataset ID directly must raise PermissionError
    with pytest.raises(PermissionError) as exc_info:
        assistant.answer_query(
            hospital_b_user,
            "How many patients are in my uploaded dataset?",
            dataset_id=hosp_a_id,
        )
    assert "Cross-Hospital Access Violation" in str(exc_info.value)


def test_query_patient_by_name(assistant, hospital_a_user, hospital_b_user):
    # Hospital A queries by patient name (Eleanor Vance)
    resp_a = assistant.answer_query(hospital_a_user, "What email address is recorded for Eleanor Vance?")
    assert resp_a.success is True
    assert "eleanor.vance@synthetic-health.org" in resp_a.answer

    # Hospital B queries for Eleanor Vance: must not find or expose Eleanor Vance's records
    resp_b = assistant.answer_query(hospital_b_user, "What email address is recorded for Eleanor Vance?")
    assert resp_b.success is False
    assert "eleanor.vance" not in resp_b.answer

