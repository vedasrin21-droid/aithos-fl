"""
Hospital Dataset Store and Access Controller for AITHOS-FL
Enforces strict institutional boundary separation, role-based access control,
file validation, and non-destructive dataset preservation.
"""

from __future__ import annotations
import io
import time
import uuid
from dataclasses import dataclass, field
from typing import Any
import pandas as pd

from privacy.pii_detector import PIIDetector, PIIScanResult
from privacy.redaction import redact_dataframe

# Authorized institutional hospital nodes
AUTHORIZED_HOSPITAL_IDS = {"HOSP-A", "HOSP-B"}

# Maximum file size allowed for local node ingestion (25 MB)
MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024


@dataclass
class HospitalDataset:
    """Represents an ingested clinical dataset associated with an institutional node."""
    id: str
    filename: str
    hospital_id: str
    uploaded_at: str
    row_count: int
    column_count: int
    column_names: list[str]
    size_bytes: int
    original_df: pd.DataFrame
    scan_result: PIIScanResult
    redacted_df: pd.DataFrame
    redaction_summary: dict[str, Any] = field(default_factory=dict)


class HospitalDatasetStore:
    """
    In-memory isolated dataset repository partitioned by hospital_id.
    Ensures complete isolation: Hospital A cannot see or access Hospital B's data,
    and unassigned/normal users cannot upload or view datasets.
    """

    def __init__(self):
        # Partitioned store: hospital_id -> dict of dataset_id -> HospitalDataset
        self._partitions: dict[str, dict[str, HospitalDataset]] = {
            "HOSP-A": {},
            "HOSP-B": {},
        }
        self.detector = PIIDetector()

    def validate_user_access(self, user: dict[str, Any], action: str = "access") -> str:
        """
        Validates whether the user is permitted to interact with hospital datasets.
        Raises PermissionError if unauthorized.
        """
        if not user:
            raise PermissionError("Unauthenticated: No user session active.")

        hospital_id = str(user.get("hospital_id", "")).strip()
        role = str(user.get("role", "")).strip()

        if hospital_id not in AUTHORIZED_HOSPITAL_IDS:
            raise PermissionError(
                f"Access Denied: Account with role '{role}' and Hospital ID '{hospital_id}' "
                f"is not authorized to {action} hospital-private datasets."
            )
        return hospital_id

    def validate_and_parse_csv(
        self,
        file_bytes: bytes,
        filename: str,
    ) -> pd.DataFrame:
        """
        Validates CSV file size, extension, and content structure.
        Raises ValueError for invalid, empty, or malformed inputs.
        """
        if not filename.lower().endswith(".csv"):
            raise ValueError(f"Invalid file type for '{filename}'. Only .csv files are supported.")

        if len(file_bytes) == 0:
            raise ValueError("The uploaded CSV file is empty (0 bytes).")

        if len(file_bytes) > MAX_FILE_SIZE_BYTES:
            mb_limit = MAX_FILE_SIZE_BYTES // (1024 * 1024)
            raise ValueError(f"File size exceeds maximum threshold of {mb_limit} MB.")

        # Attempt decoding with standard encodings
        decoded_text = None
        for encoding in ["utf-8", "latin-1", "cp1252"]:
            try:
                decoded_text = file_bytes.decode(encoding)
                break
            except UnicodeDecodeError:
                continue

        if decoded_text is None:
            raise ValueError("Malformed CSV: Unable to decode file with UTF-8 or Latin-1 encodings.")

        # Parse CSV with pandas
        try:
            df = pd.read_csv(io.StringIO(decoded_text))
        except (pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
            raise ValueError(f"Malformed CSV content: {str(exc)}") from exc
        except Exception as exc:
            raise ValueError(f"Failed to parse CSV: {str(exc)}") from exc

        if df.empty and len(df.columns) == 0:
            raise ValueError("Malformed CSV: File contains no header or tabular rows.")

        if len(df) == 0:
            raise ValueError("The uploaded CSV file contains headers but zero data rows.")

        return df

    def ingest_dataset(
        self,
        user: dict[str, Any],
        filename: str,
        file_bytes: bytes,
    ) -> HospitalDataset:
        """
        Validates permissions, parses CSV, executes PII detection and redaction,
        and saves the dataset inside the owning hospital partition.
        """
        hospital_id = self.validate_user_access(user, action="upload")
        df = self.validate_and_parse_csv(file_bytes, filename)

        # Run PII scanner on original dataset
        scan_result = self.detector.scan_dataframe(df)

        # Generate a distinct sanitized copy without touching the original
        redacted_df, summary = redact_dataframe(
            df=df,
            detector=self.detector,
            scan_result=scan_result,
        )

        dataset_id = str(uuid.uuid4())[:8]
        timestamp_str = time.strftime("%Y-%m-%d %H:%M:%S")

        dataset = HospitalDataset(
            id=dataset_id,
            filename=filename,
            hospital_id=hospital_id,
            uploaded_at=timestamp_str,
            row_count=len(df),
            column_count=len(df.columns),
            column_names=[str(c) for c in df.columns],
            size_bytes=len(file_bytes),
            original_df=df,
            scan_result=scan_result,
            redacted_df=redacted_df,
            redaction_summary=summary,
        )

        self._partitions[hospital_id][dataset_id] = dataset
        return dataset

    def get_datasets_for_user(self, user: dict[str, Any]) -> list[HospitalDataset]:
        """Returns all datasets owned exclusively by the user's hospital node."""
        hospital_id = self.validate_user_access(user, action="view")
        return list(self._partitions.get(hospital_id, {}).values())

    def get_dataset_by_id(
        self,
        user: dict[str, Any],
        dataset_id: str,
    ) -> HospitalDataset:
        """
        Retrieves a dataset by ID, strictly enforcing hospital partition boundary.
        Prevents cross-hospital access.
        """
        hospital_id = self.validate_user_access(user, action="access")
        partition = self._partitions.get(hospital_id, {})
        if dataset_id not in partition:
            # Check if dataset exists in another partition to detect cross-hospital violations
            for other_hosp_id, other_part in self._partitions.items():
                if other_hosp_id != hospital_id and dataset_id in other_part:
                    raise PermissionError(
                        f"Cross-Hospital Access Violation: Dataset '{dataset_id}' belongs to "
                        f"{other_hosp_id} and cannot be accessed by {hospital_id}."
                    )
            raise KeyError(f"Dataset '{dataset_id}' not found in {hospital_id} repository.")
        return partition[dataset_id]

    def clear_all(self):
        """Clears all dataset partitions (useful for test resets)."""
        for hosp in self._partitions:
            self._partitions[hosp].clear()
