"""
Shared AI Module for AITHOS-FL
Provides controlled, approval-based knowledge sharing across the federated consortium.
"""

from shared_ai.knowledge_store import SharedKnowledgeStore, ALLOWED_CATEGORIES
from shared_ai.assistant import SharedAIAssistant, SharedAIResponse, SHARED_STATUS_LABEL, PRIVATE_RECORD_REFUSAL, KNOWLEDGE_UNAVAILABLE_MESSAGE

__all__ = [
    "SharedKnowledgeStore",
    "SharedAIAssistant",
    "SharedAIResponse",
    "SHARED_STATUS_LABEL",
    "PRIVATE_RECORD_REFUSAL",
    "KNOWLEDGE_UNAVAILABLE_MESSAGE",
    "ALLOWED_CATEGORIES",
]
