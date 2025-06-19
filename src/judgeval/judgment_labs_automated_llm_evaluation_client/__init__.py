"""A client library for accessing Judgment Labs: Automated LLM Evaluation"""

from .client import AuthenticatedClient, Client

__all__ = (
    "AuthenticatedClient",
    "Client",
)
