from enum import Enum


class Role(str, Enum):
    ADMIN = "admin"
    DEVELOPER = "developer"
    OWNER = "owner"
    VIEWER = "viewer"

    def __str__(self) -> str:
        return str(self.value)
