"""
Shared AI Assistant for AITHOS-FL
Provides controlled retrieval-based question answering strictly from approved shared knowledge.
Never accesses raw hospital CSV files, refuses private patient requests, and returns standard
fallback when information has not been approved.
"""

from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Any

from shared_ai.knowledge_store import SharedKnowledgeStore

SHARED_STATUS_LABEL = "Shared knowledge only — no direct access to hospital-private records."
PRIVATE_RECORD_REFUSAL = (
    "Private patient records are unavailable. The Shared AI Assistant only accesses approved "
    "institutional knowledge and cannot access or disclose patient-specific records."
)
KNOWLEDGE_UNAVAILABLE_MESSAGE = "I don't have that information in the shared knowledge base."


@dataclass
class SharedAIResponse:
    """Encapsulates responses from the Shared AI Assistant."""
    answer: str
    status_label: str = SHARED_STATUS_LABEL
    found_in_knowledge_base: bool = False
    refusal: bool = False
    matched_item: dict[str, Any] | None = None


class SharedAIAssistant:
    """
    Consortium-wide Shared AI Assistant.
    Retrieves answers strictly from the approved SharedKnowledgeStore.
    """

    def __init__(self, knowledge_store: SharedKnowledgeStore):
        self.store = knowledge_store

    def is_private_record_query(self, query: str) -> bool:
        """
        Detects if query is attempting to elicit private patient records,
        direct patient contact information, or individual identifiers.
        """
        q_lower = query.lower()

        # 1. Queries with patient IDs (e.g., P001, P-001, P_001, P101, patient #001)
        pid_matches = re.findall(r"\b[pP](?:[-_]\d+|\d{1,6}|[-_][a-zA-Z0-9]+)\b", query)
        excluded = {"policy", "policies", "procedure", "practice", "patient", "patients", "percent", "protocol"}
        if any(m.lower() not in excluded for m in pid_matches):
            return True

        if re.search(r"\bpatient\s+#?\d+\b", q_lower):
            return True

        # 2. Queries asking about individual patients or patient-level records
        patient_phrases = [
            "patient's email", "patient email", "patient's phone", "patient phone",
            "patient's name", "patient name", "patient's address", "patient address",
            "patient's diagnosis", "patient diagnosis", "patient's condition", "patient condition",
            "patient's lab", "patient lab", "patient's test", "patient test",
            "patient's chart", "patient chart", "patient's vitals", "patient vitals",
            "patient's doctor", "patient doctor", "patient's physician", "patient physician",
            "medical record number", "patient mrn", "patient record", "patient records",
            "patient details", "patient history", "patient info", "patient information",
            "patient data", "individual patient", "who is patient", "contact for patient",
            "patient-specific", "individual record", "individual records",
        ]
        if any(phrase in q_lower for phrase in patient_phrases):
            return True

        # 3. Direct requests for patient personal attributes
        if any(term in q_lower for term in ["mrn-", "patient id", "patient_id", "patient-id"]):
            return True

        # 4. Queries referencing specific known patient names
        known_patient_names = [
            "eleanor vance", "marcus aurelius", "marcus thorne", "amina patel", "amina fatima",
            "carlos gomez", "sarah jenkins", "devon alexander", "elena rostova",
            "liam o'connor", "zoe chen"
        ]
        if any(name in q_lower for name in known_patient_names):
            return True

        return False

    def answer_query(self, query: str, user: dict[str, Any] | None = None) -> SharedAIResponse:
        """
        Processes a query against the approved shared knowledge base.
        """
        cleaned_query = query.strip()
        q_lower = cleaned_query.lower()

        # 1. Privacy Boundary Check: refuse individual patient queries
        if self.is_private_record_query(cleaned_query):
            return SharedAIResponse(
                answer=PRIVATE_RECORD_REFUSAL,
                refusal=True,
                found_in_knowledge_base=False,
            )

        # 2. Query Approved Shared Knowledge Store ONLY
        items = self.store.get_all_items()
        if not items:
            return SharedAIResponse(
                answer=KNOWLEDGE_UNAVAILABLE_MESSAGE,
                found_in_knowledge_base=False,
            )

        # 3. Keyword / semantic search against approved titles and content
        best_match = None
        highest_score = 0

        # Tokenize query into meaningful search words
        query_words = set(re.findall(r"\w+", q_lower))
        stop_words = {"the", "a", "an", "is", "what", "which", "how", "for", "in", "of", "to", "and", "or", "tell", "me", "show"}
        keywords = query_words - stop_words

        for item in items:
            title_text = item.get("title", "").lower()
            content_text = item.get("content", "").lower()
            category_text = item.get("category", "").lower()
            combined = f"{title_text} {content_text} {category_text}"

            score = 0
            # Higher score for key policy phrases
            if "reschedul" in q_lower and "reschedul" in combined:
                score += 15
            if "visitor" in q_lower and "visitor" in combined:
                score += 15
            if "department" in q_lower and "department" in combined:
                score += 15
            if "telehealth" in q_lower and "telehealth" in combined:
                score += 15
            if "cohort" in q_lower and "cohort" in combined:
                score += 10

            # Match individual keywords
            for kw in keywords:
                if len(kw) > 3:
                    if kw in title_text:
                        score += 5
                    elif kw in content_text:
                        score += 2

            if score > highest_score:
                highest_score = score
                best_match = item

        if best_match and highest_score >= 5:
            hospital_tag = best_match.get("hospital_id", "Institutional Node")
            approved_title = best_match.get("title")
            approved_content = best_match.get("content")
            approved_category = best_match.get("category", "Policy")

            formatted_answer = (
                f"**{approved_title}**\n\n"
                f"*Category: {approved_category} &bull; Approved by: {hospital_tag}*\n\n"
                f"{approved_content}"
            )
            return SharedAIResponse(
                answer=formatted_answer,
                found_in_knowledge_base=True,
                matched_item=best_match,
            )

        # 4. Fallback if information is unavailable
        return SharedAIResponse(
            answer=KNOWLEDGE_UNAVAILABLE_MESSAGE,
            found_in_knowledge_base=False,
        )
