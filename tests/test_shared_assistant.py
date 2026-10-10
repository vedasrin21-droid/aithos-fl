"""
Unit tests for Shared AI Assistant and Approval-based Knowledge Sharing (shared_ai/).
Tests:
- Shared AI response before and after knowledge approval.
- Refusal of private patient record queries.
- Role-based approval access enforcement (Normal user rejected).
- Patient PII rejection during approval validation.
- Persistence across store reinstantiations.
"""

import os
import tempfile
import pytest

from shared_ai.knowledge_store import SharedKnowledgeStore
from shared_ai.assistant import (
    SharedAIAssistant,
    KNOWLEDGE_UNAVAILABLE_MESSAGE,
    PRIVATE_RECORD_REFUSAL,
    SHARED_STATUS_LABEL,
)


@pytest.fixture
def temp_store_file():
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    yield path
    if os.path.exists(path):
        os.remove(path)


@pytest.fixture
def store(temp_store_file):
    return SharedKnowledgeStore(file_path=temp_store_file)


@pytest.fixture
def assistant(store):
    return SharedAIAssistant(store)


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


def test_shared_ai_before_approval(assistant):
    # Before approval: information is unavailable in shared store
    resp = assistant.answer_query("What is the appointment rescheduling policy?")
    assert resp.found_in_knowledge_base is False
    assert resp.answer == KNOWLEDGE_UNAVAILABLE_MESSAGE
    assert resp.status_label == SHARED_STATUS_LABEL


def test_shared_ai_after_approval(assistant, store, hospital_a_user):
    # Hospital A approves general appointment rescheduling policy
    store.publish_item(
        user=hospital_a_user,
        title="Hospital A Appointment Rescheduling Policy",
        category="Institutional Policy",
        content=(
            "Appointments at Hospital A Medical Center may be rescheduled up to 24 hours prior "
            "to the scheduled slot without penalty via the patient portal or clinic coordinator."
        ),
        summary="General 24-hour outpatient appointment rescheduling guidelines.",
    )

    # Now query Shared AI
    resp = assistant.answer_query("What is the appointment rescheduling policy?")
    assert resp.found_in_knowledge_base is True
    assert "Hospital A Appointment Rescheduling Policy" in resp.answer
    assert "24 hours prior" in resp.answer
    assert "HOSP-A" in resp.answer


def test_shared_ai_refuses_patient_private_records(assistant):
    # Queries asking for specific patient records must be refused
    test_queries = [
        "What email address is recorded for patient P001?",
        "Show me the phone number for patient P002",
        "What is patient P001's address?",
        "Who is patient P101?",
        "Show patient details for P003",
    ]

    for q in test_queries:
        resp = assistant.answer_query(q)
        assert resp.refusal is True
        assert resp.answer == PRIVATE_RECORD_REFUSAL


def test_normal_user_cannot_publish_knowledge(store, normal_user):
    # Normal users must be denied when attempting to approve or publish knowledge
    with pytest.raises(PermissionError) as exc_info:
        store.publish_item(
            user=normal_user,
            title="Unauthorized Submission",
            category="Institutional Policy",
            content="Testing publication attempt by external user.",
        )
    assert "not authorized to approve knowledge sharing" in str(exc_info.value)


def test_rejection_of_patient_pii_in_approval(store, hospital_a_user):
    # 1. Reject submission with email
    with pytest.raises(ValueError) as exc1:
        store.publish_item(
            user=hospital_a_user,
            title="Policy with Email",
            category="Institutional Policy",
            content="Reach out to coordinator at nurse.smith@demo-health.org for details.",
        )
    assert "email" in str(exc1.value).lower()

    # 2. Reject submission with phone number
    with pytest.raises(ValueError) as exc2:
        store.publish_item(
            user=hospital_a_user,
            title="Policy with Phone",
            category="Institutional Policy",
            content="Call the coordinator at 555-432-8765 for direct updates.",
        )
    assert "phone" in str(exc2.value).lower() or "telephone" in str(exc2.value).lower()

    # 3. Reject submission with patient ID
    with pytest.raises(ValueError) as exc3:
        store.publish_item(
            user=hospital_a_user,
            title="Policy with Patient ID",
            category="Institutional Policy",
            content="Referral rules applied to patient P001 and patient P002.",
        )
    assert "patient identifier" in str(exc3.value).lower()


def test_persistence_of_approved_knowledge(temp_store_file, hospital_a_user):
    # Write using first store instance
    store1 = SharedKnowledgeStore(file_path=temp_store_file)
    store1.publish_item(
        user=hospital_a_user,
        title="Telemetry Workflow Guidelines",
        category="Institutional Policy",
        content="Inpatient telemetry monitoring is re-evaluated every 48 hours.",
    )

    # Create new instance pointing to same file (simulating app restart)
    store2 = SharedKnowledgeStore(file_path=temp_store_file)
    items = store2.get_all_items()
    assert len(items) == 1
    assert items[0]["title"] == "Telemetry Workflow Guidelines"
    assert items[0]["hospital_id"] == "HOSP-A"

    audit = store2.get_audit_log("HOSP-A")
    assert len(audit) == 1
    assert audit[0]["title"] == "Telemetry Workflow Guidelines"

    # Verify assistant on new store instance answers correctly
    assistant2 = SharedAIAssistant(store2)
    resp = assistant2.answer_query("Tell me about the telemetry workflow")
    assert resp.found_in_knowledge_base is True
    assert "Telemetry Workflow Guidelines" in resp.answer


def test_shared_ai_refuses_queries_by_patient_name_and_clinical_fields(assistant):
    # Verify refusal on patient names and clinical record requests
    name_and_record_queries = [
        "What is Eleanor Vance's diagnosis?",
        "Show me Devon Alexander's records",
        "What is the condition of Marcus Aurelius?",
        "Show patient data",
        "What is patient P_001's lab result?",
        "Who is patient #1?",
    ]
    for q in name_and_record_queries:
        resp = assistant.answer_query(q)
        assert resp.refusal is True
        assert resp.answer == PRIVATE_RECORD_REFUSAL


def test_rejection_of_unconfirmed_publication(store, hospital_a_user):
    # Submission without explicit user confirmation must be rejected
    with pytest.raises(ValueError) as exc:
        store.publish_item(
            user=hospital_a_user,
            title="Valid Policy",
            category="Institutional Policy",
            content="Standard outpatient clinical registration guidelines.",
            confirmed=False,
        )
    assert "Explicit confirmation is required" in str(exc.value)


def test_rejection_of_patient_names_and_csv_dumps_in_approval(store, hospital_a_user):
    # 1. Reject submission mentioning known patient name
    with pytest.raises(ValueError) as exc1:
        store.publish_item(
            user=hospital_a_user,
            title="Clinical Summary for Eleanor Vance",
            category="Institutional Policy",
            content="Summary of clinical consultations provided to Eleanor Vance.",
            confirmed=True,
        )
    assert "patient name" in str(exc1.value).lower()

    # 2. Reject submission containing raw comma-separated CSV dump
    with pytest.raises(ValueError) as exc2:
        store.publish_item(
            user=hospital_a_user,
            title="Tabular Dump",
            category="Aggregate Statistics",
            content="col1,col2,col3,col4\nval1,val2,val3,val4\n",
            confirmed=True,
        )
    assert "raw tabular or csv" in str(exc2.value).lower()


def test_shared_ai_zero_csv_fallback(assistant):
    # Shared AI must never fall back to uploaded CSV files for dataset count queries
    resp = assistant.answer_query("How many patients are in my uploaded dataset?")
    assert resp.found_in_knowledge_base is False
    assert resp.answer == KNOWLEDGE_UNAVAILABLE_MESSAGE

