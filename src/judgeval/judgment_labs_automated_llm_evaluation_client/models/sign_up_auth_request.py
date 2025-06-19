from collections.abc import Mapping
from typing import (
    TYPE_CHECKING,
    Any,
    TypeVar,
    Union,
    cast,
)

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.sign_up_auth_request_options_type_0 import (
        SignUpAuthRequestOptionsType0,
    )


T = TypeVar("T", bound="SignUpAuthRequest")


@_attrs_define
class SignUpAuthRequest:
    """
    Attributes:
        email (str):
        password (str):
        options (Union['SignUpAuthRequestOptionsType0', None, Unset]):
    """

    email: str
    password: str
    options: Union["SignUpAuthRequestOptionsType0", None, Unset] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        from ..models.sign_up_auth_request_options_type_0 import (
            SignUpAuthRequestOptionsType0,
        )

        email = self.email

        password = self.password

        options: Union[None, Unset, dict[str, Any]]
        if isinstance(self.options, Unset):
            options = UNSET
        elif isinstance(self.options, SignUpAuthRequestOptionsType0):
            options = self.options.to_dict()
        else:
            options = self.options

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "email": email,
                "password": password,
            }
        )
        if options is not UNSET:
            field_dict["options"] = options

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.sign_up_auth_request_options_type_0 import (
            SignUpAuthRequestOptionsType0,
        )

        d = dict(src_dict)
        email = d.pop("email")

        password = d.pop("password")

        def _parse_options(
            data: object,
        ) -> Union["SignUpAuthRequestOptionsType0", None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                options_type_0 = SignUpAuthRequestOptionsType0.from_dict(data)

                return options_type_0
            except:  # noqa: E722
                pass
            return cast(Union["SignUpAuthRequestOptionsType0", None, Unset], data)

        options = _parse_options(d.pop("options", UNSET))

        sign_up_auth_request = cls(
            email=email,
            password=password,
            options=options,
        )

        sign_up_auth_request.additional_properties = d
        return sign_up_auth_request

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
