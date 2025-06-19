from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

T = TypeVar("T", bound="LoginOAuthFinishRequest")


@_attrs_define
class LoginOAuthFinishRequest:
    """
    Attributes:
        auth_code (str):
        code_verifier (str):
    """

    auth_code: str
    code_verifier: str
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        auth_code = self.auth_code

        code_verifier = self.code_verifier

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "auth_code": auth_code,
                "code_verifier": code_verifier,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        auth_code = d.pop("auth_code")

        code_verifier = d.pop("code_verifier")

        login_o_auth_finish_request = cls(
            auth_code=auth_code,
            code_verifier=code_verifier,
        )

        login_o_auth_finish_request.additional_properties = d
        return login_o_auth_finish_request

    @property
    def additional_keys(self) -> list[str]:
        return list(self.additional_properties.keys())

    def __getitem__(self, key: str) -> Any:
        return self.additional_properties[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.additional_properties[key] = value

    def __delitem__(self, key: str) -> None:
        del self.additional_properties[key]

    def __contains__(self, key: str) -> bool:
        return key in self.additional_properties
