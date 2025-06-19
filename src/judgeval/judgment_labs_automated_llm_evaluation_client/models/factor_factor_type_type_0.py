from enum import Enum


class FactorFactorTypeType0(str, Enum):
    PHONE = "phone"
    TOTP = "totp"

    def __str__(self) -> str:
        return str(self.value)
