from enum import Enum


class SubscriptionTier(str, Enum):
    DEVELOPER = "developer"
    ENTERPRISE = "enterprise"
    PRO = "pro"
    STARTUP = "startup"

    def __str__(self) -> str:
        return str(self.value)
