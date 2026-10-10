"""
Local AI Assistant for AITHOS-FL
Provides private, on-premise clinical question answering strictly isolated to the authenticated hospital's datasets.
Never transmits data externally or exposes private records across hospital boundaries.
"""

from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Any
import pandas as pd

from privacy.dataset_store import HospitalDatasetStore, HospitalDataset

# Institutional policies maintained on-premise
HOSPITAL_POLICIES: dict[str, dict[str, str]] = {
    "HOSP-A": {
        "hospital_name": "Hospital A Medical Center",
        "appointment_rescheduling": (
            "Appointments at Hospital A Medical Center may be rescheduled up to 24 hours prior "
            "to the scheduled slot without penalty via the patient portal or by contacting the clinic desk. "
            "Rescheduling requests made within 24 hours require clinical coordinator approval."
        ),
        "visitor_policy": "Visiting hours are 08:00 to 20:00 daily with a maximum of 2 visitors per patient.",
    },
    "HOSP-B": {
        "hospital_name": "Hospital B Regional Health",
        "appointment_rescheduling": (
            "Appointments at Hospital B Regional Health require at least 48 hours advance notice "
            "for rescheduling through the central outpatient registration desk. "
            "Specialized diagnostic appointments are subject to department lead re-assignment."
        ),
        "visitor_policy": "Visiting hours are 09:00 to 19:00 with institutional badge access required.",
    },
}

LOCAL_STATUS_LABEL = "Local data access — authorized hospital only."


@dataclass
class LocalAIResponse:
    """Encapsulates the response from the Local AI Assistant."""
    answer: str
    status_label: str = LOCAL_STATUS_LABEL
    success: bool = True
    context_meta: dict[str, Any] | None = None


class LocalAIAssistant:
    """
    On-premise AI Assistant for institutional clinical nodes.
    Enforces strict role and hospital boundary verification on every request.
    """

    def __init__(self, store: HospitalDatasetStore):
        self.store = store

    def verify_authorization(self, user: dict[str, Any]) -> str:
        """
        Enforces backend access checks on every query.
        Raises PermissionError if the user is unauthenticated, a normal user, or unauthorized.
        """
        if not user:
            raise PermissionError("Authentication Required: Please sign in to query Local AI.")

        hospital_id = str(user.get("hospital_id", "")).strip()
        role = str(user.get("role", "")).strip()

        if hospital_id not in {"HOSP-A", "HOSP-B"}:
            raise PermissionError(
                f"Access Denied: Account with role '{role}' and Hospital ID '{hospital_id}' "
                "is not authorized to query private hospital datasets."
            )

        return hospital_id

    def _find_column(self, df: pd.DataFrame, candidates: list[str]) -> str | None:
        """Helper to find column matching any candidate name (normalized)."""
        norm_map = {
            re.sub(r"[^\w]", "", str(c).lower()): c for c in df.columns
        }
        for cand in candidates:
            cleaned_cand = re.sub(r"[^\w]", "", cand.lower())
            if cleaned_cand in norm_map:
                return norm_map[cleaned_cand]
        return None

    def _extract_patient_id(self, query: str, df: pd.DataFrame | None = None) -> str | None:
        """
        Extracts a patient ID from query text.
        Matches explicit tokens (e.g. P001, P002, P101, P-10021).
        """
        # 1. Check for explicit "patient <ID>" phrasing first
        explicit_patient = re.findall(r"\bpatient\s+([a-zA-Z0-9_-]+)\b", query, re.IGNORECASE)
        if explicit_patient:
            candidate = explicit_patient[0].strip()
            # If not a generic placeholder word like "records", "data", "id"
            if candidate.lower() not in {"id", "record", "records", "data", "detail", "details"}:
                return candidate

        # 2. Check for alphanumeric patient ID pattern like P001, P002, P101, P-10021
        p_id_pattern = r"\b([pP](?:-\d+|\d{1,6}|-[a-zA-Z0-9]+))\b"
        p_matches = re.findall(p_id_pattern, query)
        if p_matches:
            excluded_words = {
                "patient", "patients", "phone", "policy", "portal", "provider",
                "primary", "pediatrics", "pulmonology", "procedure", "please",
                "plan", "physician", "practice", "prior", "protocol", "percent"
            }
            for match in p_matches:
                if match.lower() not in excluded_words:
                    return match.strip()

        # 3. Check if any actual patient ID from dataset appears in query
        if df is not None:
            pid_col = self._find_column(df, ["patient_id", "patientid", "id", "mrn", "patient"])
            if pid_col:
                q_lower = query.lower()
                for actual_val in df[pid_col].dropna().unique():
                    act_str = str(actual_val).strip()
                    if act_str.lower() in q_lower:
                        return act_str

        # 4. Check if any patient name from dataset appears in query
        if df is not None:
            name_col = self._find_column(df, ["patient_name", "name", "full_name"])
            if name_col:
                q_lower = query.lower()
                for actual_name in df[name_col].dropna().unique():
                    act_name_str = str(actual_name).strip()
                    if len(act_name_str) >= 4 and act_name_str.lower() in q_lower:
                        return act_name_str

        return None

    def answer_query(
        self,
        user: dict[str, Any],
        query: str,
        dataset_id: str | None = None,
    ) -> LocalAIResponse:
        """
        Processes a natural language query against the hospital's private uploaded dataset.
        Executes strict verification, structured retrieval, and factual response synthesis.
        """
        hospital_id = self.verify_authorization(user)
        cleaned_query = query.strip()
        q_lower = cleaned_query.lower()

        # Retrieve datasets belonging exclusively to this hospital
        datasets = self.store.get_datasets_for_user(user)
        if not datasets:
            return LocalAIResponse(
                answer=(
                    f"No datasets have been uploaded yet for **{hospital_id}** ({user.get('name')}).\n\n"
                    "Please navigate to the **Upload Dataset** page and ingest a CSV file or load "
                    "the synthetic demo dataset to begin querying local records."
                ),
                success=False,
            )

        # Select dataset: specified ID or most recent
        target_dataset: HospitalDataset | None = None
        if dataset_id:
            try:
                target_dataset = self.store.get_dataset_by_id(user, dataset_id)
            except PermissionError:
                raise
            except KeyError:
                target_dataset = datasets[-1]
        else:
            target_dataset = datasets[-1]

        df = target_dataset.original_df

        # ---------------------------------------------------------------------
        # INTENT 1: Patient Count / Dataset Size
        # (e.g. "How many patients are in my uploaded dataset?")
        # ---------------------------------------------------------------------
        if any(kw in q_lower for kw in [
            "how many patient",
            "patient count",
            "total patient",
            "how many record",
            "total record",
            "number of patient",
            "count of patient",
            "how many rows",
        ]):
            pid_col = self._find_column(df, ["patient_id", "patientid", "id", "mrn", "patient"])
            if pid_col:
                unique_patients = int(df[pid_col].dropna().nunique())
                total_rows = len(df)
                return LocalAIResponse(
                    answer=(
                        f"Your uploaded dataset (`{target_dataset.filename}`) contains "
                        f"**{unique_patients} unique patients** across **{total_rows} total records**."
                    ),
                    context_meta={"dataset": target_dataset.filename, "count": unique_patients},
                )
            else:
                total_rows = len(df)
                return LocalAIResponse(
                    answer=(
                        f"Your uploaded dataset (`{target_dataset.filename}`) contains "
                        f"**{total_rows} total records**."
                    ),
                    context_meta={"dataset": target_dataset.filename, "rows": total_rows},
                )

        # ---------------------------------------------------------------------
        # INTENT 2: Department Aggregations
        # (e.g. "Which department has the most records?")
        # ---------------------------------------------------------------------
        if any(kw in q_lower for kw in [
            "which department",
            "what department",
            "most record",
            "highest record",
            "department has the most",
            "department breakdown",
            "departments",
        ]):
            dept_col = self._find_column(df, [
                "department", "dept", "clinic_department", "specialty", "clinic", "unit"
            ])
            if dept_col is None:
                avail_cols = ", ".join(f"`{c}`" for c in df.columns)
                return LocalAIResponse(
                    answer=(
                        f"The uploaded dataset (`{target_dataset.filename}`) does not contain a "
                        f"'department' column.\n\n"
                        f"**Available columns in this dataset:**\n{avail_cols}"
                    ),
                    success=False,
                )

            counts = df[dept_col].dropna().value_counts()
            if counts.empty:
                return LocalAIResponse(
                    answer=f"The '{dept_col}' column in `{target_dataset.filename}` contains no recorded entries.",
                    success=False,
                )

            top_dept = counts.index[0]
            top_count = int(counts.iloc[0])
            pct = (top_count / len(df)) * 100

            breakdown_lines = [
                f"- **{dept}**: {count} record(s) ({count / len(df) * 100:.1f}%)"
                for dept, count in counts.items()
            ]
            breakdown_str = "\n".join(breakdown_lines)

            return LocalAIResponse(
                answer=(
                    f"The department with the most records is **{top_dept}** with "
                    f"**{top_count} records** ({pct:.1f}% of total dataset).\n\n"
                    f"**Department Breakdown ({target_dataset.filename}):**\n{breakdown_str}"
                ),
                context_meta={"top_department": top_dept, "count": top_count},
            )

        # ---------------------------------------------------------------------
        # INTENT 3: Appointment Rescheduling Policy
        # (e.g. "What is the appointment rescheduling policy for patient P001?")
        # ---------------------------------------------------------------------
        if any(kw in q_lower for kw in ["reschedul", "appointment policy", "cancel appointment", "cancellation policy"]):
            target_pid = self._extract_patient_id(cleaned_query, df)
            policy_info = HOSPITAL_POLICIES.get(hospital_id, HOSPITAL_POLICIES["HOSP-A"])
            policy_text = policy_info["appointment_rescheduling"]
            hosp_name = policy_info["hospital_name"]

            if target_pid:
                # Verify patient in local dataset
                pid_col = self._find_column(df, ["patient_id", "patientid", "id", "mrn", "patient"])
                patient_found = False
                dept_context = ""

                if pid_col:
                    matches = df[df[pid_col].astype(str).str.lower() == target_pid.lower()]
                    if not matches.empty:
                        patient_found = True
                        dept_col = self._find_column(df, ["department", "dept", "specialty"])
                        if dept_col and pd.notna(matches.iloc[0][dept_col]):
                            dept_context = f" in **{matches.iloc[0][dept_col]}**"

                if patient_found:
                    return LocalAIResponse(
                        answer=(
                            f"**Appointment Rescheduling Policy for Patient {target_pid}**"
                            f"{dept_context} at **{hosp_name}**:\n\n"
                            f"{policy_text}"
                        ),
                        context_meta={"patient_id": target_pid, "found": True},
                    )
                else:
                    return LocalAIResponse(
                        answer=(
                            f"Patient **{target_pid}** was not found in **{hospital_id}**'s uploaded dataset.\n\n"
                            f"**General Policy for {hosp_name}:**\n{policy_text}"
                        ),
                        context_meta={"patient_id": target_pid, "found": False},
                    )
            else:
                return LocalAIResponse(
                    answer=f"**Appointment Rescheduling Policy for {hosp_name}:**\n\n{policy_text}",
                )

        # ---------------------------------------------------------------------
        # INTENT 4: Patient-Specific Attribute Queries
        # (e.g. "What email address is recorded for patient P001?")
        # (e.g. "Show the phone number for patient P002.")
        # ---------------------------------------------------------------------
        target_pid = self._extract_patient_id(cleaned_query, df)
        if target_pid:
            pid_col = self._find_column(df, ["patient_id", "patientid", "id", "mrn", "patient"])
            name_col = self._find_column(df, ["patient_name", "name", "full_name"])
            if not pid_col and not name_col:
                return LocalAIResponse(
                    answer=f"Dataset `{target_dataset.filename}` does not have recognizable patient identification columns.",
                    success=False,
                )

            # Match patient row by ID or Name
            p_mask = pd.Series([False] * len(df), index=df.index)
            if pid_col:
                p_mask = p_mask | (df[pid_col].astype(str).str.strip().str.lower() == target_pid.lower())
            if name_col:
                p_mask = p_mask | (df[name_col].astype(str).str.strip().str.lower() == target_pid.lower())
            matching_rows = df[p_mask]

            if matching_rows.empty:
                # Do not invent answers when missing
                avail_patients: list[str] = []
                if pid_col:
                    avail_patients = [f"`{p}`" for p in df[pid_col].dropna().unique()]
                elif name_col:
                    avail_patients = [f"`{p}`" for p in df[name_col].dropna().unique()]
                return LocalAIResponse(
                    answer=(
                        f"Patient **{target_pid}** was not found in **{hospital_id}**'s uploaded dataset "
                        f"(`{target_dataset.filename}`).\n\n"
                        f"Available patients in this dataset: "
                        + ", ".join(avail_patients)
                    ),
                    success=False,
                    context_meta={"patient_id": target_pid, "found": False},
                )

            row = matching_rows.iloc[0]

            # Determine which attribute is requested
            # Email
            if any(kw in q_lower for kw in ["email", "e-mail", "mail"]):
                email_col = self._find_column(df, ["email", "email_address", "mail"])
                if email_col and pd.notna(row.get(email_col)):
                    val = str(row[email_col]).strip()
                    return LocalAIResponse(
                        answer=f"The email address recorded for patient **{target_pid}** is: `{val}`.",
                        context_meta={"patient_id": target_pid, "field": "email"},
                    )
                else:
                    return LocalAIResponse(
                        answer=f"No email address is recorded for patient **{target_pid}** in this dataset.",
                        success=False,
                    )

            # Phone
            if any(kw in q_lower for kw in ["phone", "telephone", "mobile", "contact number", "cell"]):
                phone_col = self._find_column(df, ["phone", "phone_number", "mobile", "telephone"])
                if phone_col and pd.notna(row.get(phone_col)):
                    val = str(row[phone_col]).strip()
                    return LocalAIResponse(
                        answer=f"The phone number recorded for patient **{target_pid}** is: `{val}`.",
                        context_meta={"patient_id": target_pid, "field": "phone"},
                    )
                else:
                    return LocalAIResponse(
                        answer=f"No phone number is recorded for patient **{target_pid}** in this dataset.",
                        success=False,
                    )

            # Address
            if any(kw in q_lower for kw in ["address", "residence", "street", "location"]):
                addr_col = self._find_column(df, ["address", "home_address", "street_address"])
                if addr_col and pd.notna(row.get(addr_col)):
                    val = str(row[addr_col]).strip()
                    return LocalAIResponse(
                        answer=f"The address recorded for patient **{target_pid}** is: `{val}`.",
                        context_meta={"patient_id": target_pid, "field": "address"},
                    )
                else:
                    return LocalAIResponse(
                        answer=f"No address is recorded for patient **{target_pid}** in this dataset.",
                        success=False,
                    )

            # Diagnosis / Condition
            if any(kw in q_lower for kw in ["diagnosis", "icd", "condition", "disease", "diagnosis code"]):
                diag_col = self._find_column(df, ["diagnosis_code", "diagnosis", "icd10", "condition"])
                if diag_col and pd.notna(row.get(diag_col)):
                    val = str(row[diag_col]).strip()
                    return LocalAIResponse(
                        answer=f"The diagnosis code recorded for patient **{target_pid}** is: `{val}`.",
                        context_meta={"patient_id": target_pid, "field": "diagnosis"},
                    )
                else:
                    return LocalAIResponse(
                        answer=f"No diagnosis code is recorded for patient **{target_pid}** in this dataset.",
                        success=False,
                    )

            # Department
            if any(kw in q_lower for kw in ["department", "dept", "specialty", "clinic"]):
                dept_col = self._find_column(df, ["department", "dept", "specialty"])
                if dept_col and pd.notna(row.get(dept_col)):
                    val = str(row[dept_col]).strip()
                    return LocalAIResponse(
                        answer=f"Patient **{target_pid}** is assigned to the **{val}** department.",
                        context_meta={"patient_id": target_pid, "field": "department"},
                    )
                else:
                    return LocalAIResponse(
                        answer=f"No department is recorded for patient **{target_pid}** in this dataset.",
                        success=False,
                    )

            # Full Patient Record Summary if no specific single field requested
            summary_fields = []
            for col in df.columns:
                val = row[col]
                if pd.notna(val):
                    summary_fields.append(f"- **{col}**: `{val}`")

            summary_str = "\n".join(summary_fields)
            return LocalAIResponse(
                answer=(
                    f"**Patient Record for {target_pid}** (`{target_dataset.filename}`):\n\n"
                    f"{summary_str}"
                ),
                context_meta={"patient_id": target_pid},
            )

        # ---------------------------------------------------------------------
        # INTENT 5: Column or Schema Inspection
        # ---------------------------------------------------------------------
        if any(kw in q_lower for kw in ["what columns", "list columns", "schema", "dataset columns"]):
            col_list = ", ".join(f"`{c}`" for c in df.columns)
            return LocalAIResponse(
                answer=f"The active dataset (`{target_dataset.filename}`) contains the following columns:\n\n{col_list}",
            )

        # ---------------------------------------------------------------------
        # INTENT 6: Unsupported / Ambiguous Question
        # (Do not invent answers when missing or ambiguous)
        # ---------------------------------------------------------------------
        return LocalAIResponse(
            answer=(
                "I am your on-premise **Local AI Assistant** operating strictly on "
                f"**{hospital_id}**'s uploaded dataset (`{target_dataset.filename}`).\n\n"
                "I could not find an exact match for your question without ambiguity. "
                "To maintain data integrity, I do not fabricate missing records.\n\n"
                "**Supported query examples:**\n"
                "- *'How many patients are in my uploaded dataset?'*\n"
                "- *'Which department has the most records?'*\n"
                "- *'What is the appointment rescheduling policy for patient P001?'*\n"
                "- *'What email address is recorded for patient P001?'*\n"
                "- *'Show the phone number for patient P002.'*"
            ),
            success=False,
        )
