"""
Synthetic Healthcare Data Generator for AITHOS-FL
Generates non-real synthetic clinical records for testing PII detection, redaction, and local query answering.
No real patient data is used.
"""

import io
import pandas as pd

# Hospital A Synthetic Dataset
# Configured with patient IDs (P001, P002, etc.) and department allocations
SYNTHETIC_HOSPITAL_A_ROWS = [
    {
        "patient_id": "P001",
        "patient_name": "Eleanor Vance",
        "medical_record_number": "MRN-A-88219",
        "email": "eleanor.vance@synthetic-health.org",
        "phone": "(555) 234-8901",
        "department": "Cardiology",
        "address": "742 Evergreen Terrace, Sector 4",
        "age": 52,
        "blood_pressure": "128/82",
        "cholesterol": 210,
        "diagnosis_code": "I10.0",
        "clinical_notes": "Patient reported mild fatigue. In case of emergency notify caregiver at caregiver.vance@demo-family.net or call 555-998-1122.",
    },
    {
        "patient_id": "P002",
        "patient_name": "Marcus Aurelius Thorne",
        "medical_record_number": "MRN-A-88220",
        "email": "m.thorne@synthetic-provider.net",
        "phone": "+1-555-432-8765",
        "department": "Cardiology",
        "address": "128 Pine Hollow Road, District 2",
        "age": 64,
        "blood_pressure": "142/88",
        "cholesterol": 235,
        "diagnosis_code": "E11.9",
        "clinical_notes": "HbA1c levels stable. Follow-up lab results sent to physician Dr. Clark at clark.clinic@hospital-alpha.test.",
    },
    {
        "patient_id": "P003",
        "patient_name": "Amina Fatima Patel",
        "medical_record_number": "MRN-A-88221",
        "email": "amina.patel@clinical-demo.org",
        "phone": "555-876-1234",
        "department": "Neurology",
        "address": "456 Maple Blvd, Suite 101",
        "age": 38,
        "blood_pressure": "118/76",
        "cholesterol": 185,
        "diagnosis_code": "J45.0",
        "clinical_notes": "Mild exercise-induced asthma. Inhaler dosage renewed. Direct line: 555-314-1592.",
    },
    {
        "patient_id": "P004",
        "patient_name": "Carlos Gomez",
        "medical_record_number": "MRN-A-88222",
        "email": "cgomez@synthetic-mail.com",
        "phone": "(555) 678-9012",
        "department": "Cardiology",
        "address": "890 Cedar Lane, Apt 3B",
        "age": 47,
        "blood_pressure": "130/84",
        "cholesterol": 205,
        "diagnosis_code": "K21.9",
        "clinical_notes": "Acid reflux management. Dietary recommendations provided.",
    },
    {
        "patient_id": "P005",
        "patient_name": "Sarah Jenkins",
        "medical_record_number": "MRN-A-88223",
        "email": "sarah.j@synthetic-health.org",
        "phone": "+1-555-789-0123",
        "department": "Pediatrics",
        "address": "312 Birch Court, Unit 12",
        "age": 29,
        "blood_pressure": "112/72",
        "cholesterol": 170,
        "diagnosis_code": "R51.0",
        "clinical_notes": "Tension headache evaluation. Sleep hygiene protocol advised. Contact desk: support@synthetic-care.org.",
    },
]

# Hospital B Synthetic Dataset
SYNTHETIC_HOSPITAL_B_ROWS = [
    {
        "patient_id": "P101",
        "patient_name": "Devon Alexander",
        "medical_record_number": "MRN-B-44101",
        "email": "devon.alexander@regional-care.test",
        "phone": "(555) 765-4321",
        "department": "Oncology",
        "address": "1001 Highland Ridge, Bay Area",
        "age": 59,
        "blood_pressure": "136/86",
        "cholesterol": 224,
        "diagnosis_code": "I25.1",
        "clinical_notes": "Cardiology consultation completed. Reach attending nurse at triage.b@regional-health.test or 555-121-3434.",
    },
    {
        "patient_id": "P102",
        "patient_name": "Elena Rostova",
        "medical_record_number": "MRN-B-44102",
        "email": "e.rostova@synthetic-clinic.org",
        "phone": "+1-555-890-1234",
        "department": "Oncology",
        "address": "220 Riverview Park, Building C",
        "age": 41,
        "blood_pressure": "122/78",
        "cholesterol": 192,
        "diagnosis_code": "E03.9",
        "clinical_notes": "Hypothyroidism monitoring. Thyroid panel scheduled next quarter.",
    },
    {
        "patient_id": "P103",
        "patient_name": "Liam O'Connor",
        "medical_record_number": "MRN-B-44103",
        "email": "liam.oconnor@synthetic-net.org",
        "phone": "555-654-9870",
        "department": "Orthopedics",
        "address": "55 Ocean Crest Drive",
        "age": 33,
        "blood_pressure": "116/74",
        "cholesterol": 178,
        "diagnosis_code": "M54.5",
        "clinical_notes": "Low back pain physical therapy referral made. Contact PT center at pt-rehab@hospital-b.test.",
    },
    {
        "patient_id": "P104",
        "patient_name": "Zoe Chen",
        "medical_record_number": "MRN-B-44104",
        "email": "z.chen@regional-health.test",
        "phone": "(555) 321-9876",
        "department": "Emergency",
        "address": "908 Sunset Boulevard",
        "age": 48,
        "blood_pressure": "124/80",
        "cholesterol": 190,
        "diagnosis_code": "S93.4",
        "clinical_notes": "Ankle sprain managed with RICE protocol. Scheduled for follow-up if pain persists.",
    },
]


def generate_synthetic_dataset(hospital_id: str = "HOSP-A") -> pd.DataFrame:
    """Returns a synthetic pandas DataFrame tailored for the hospital node."""
    rows = SYNTHETIC_HOSPITAL_A_ROWS if hospital_id == "HOSP-A" else SYNTHETIC_HOSPITAL_B_ROWS
    return pd.DataFrame(rows)


def generate_synthetic_csv_bytes(hospital_id: str = "HOSP-A") -> tuple[str, bytes]:
    """Generates synthetic CSV bytes and a suggested filename."""
    df = generate_synthetic_dataset(hospital_id)
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    csv_bytes = buf.getvalue().encode("utf-8")
    filename = f"synthetic_clinical_{hospital_id.lower().replace('-', '_')}.csv"
    return filename, csv_bytes
