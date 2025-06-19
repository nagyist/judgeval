import datetime
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
from dateutil.parser import isoparse

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.user_identity_identity_data import UserIdentityIdentityData


T = TypeVar("T", bound="UserIdentity")


@_attrs_define
class UserIdentity:
    """
    Attributes:
        id (str):
        identity_id (str):
        user_id (str):
        identity_data (UserIdentityIdentityData):
        provider (str):
        created_at (datetime.datetime):
        last_sign_in_at (Union[None, Unset, datetime.datetime]):
        updated_at (Union[None, Unset, datetime.datetime]):
    """

    id: str
    identity_id: str
    user_id: str
    identity_data: "UserIdentityIdentityData"
    provider: str
    created_at: datetime.datetime
    last_sign_in_at: Union[None, Unset, datetime.datetime] = UNSET
    updated_at: Union[None, Unset, datetime.datetime] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        id = self.id

        identity_id = self.identity_id

        user_id = self.user_id

        identity_data = self.identity_data.to_dict()

        provider = self.provider

        created_at = self.created_at.isoformat()

        last_sign_in_at: Union[None, Unset, str]
        if isinstance(self.last_sign_in_at, Unset):
            last_sign_in_at = UNSET
        elif isinstance(self.last_sign_in_at, datetime.datetime):
            last_sign_in_at = self.last_sign_in_at.isoformat()
        else:
            last_sign_in_at = self.last_sign_in_at

        updated_at: Union[None, Unset, str]
        if isinstance(self.updated_at, Unset):
            updated_at = UNSET
        elif isinstance(self.updated_at, datetime.datetime):
            updated_at = self.updated_at.isoformat()
        else:
            updated_at = self.updated_at

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "id": id,
                "identity_id": identity_id,
                "user_id": user_id,
                "identity_data": identity_data,
                "provider": provider,
                "created_at": created_at,
            }
        )
        if last_sign_in_at is not UNSET:
            field_dict["last_sign_in_at"] = last_sign_in_at
        if updated_at is not UNSET:
            field_dict["updated_at"] = updated_at

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.user_identity_identity_data import UserIdentityIdentityData

        d = dict(src_dict)
        id = d.pop("id")

        identity_id = d.pop("identity_id")

        user_id = d.pop("user_id")

        identity_data = UserIdentityIdentityData.from_dict(d.pop("identity_data"))

        provider = d.pop("provider")

        created_at = isoparse(d.pop("created_at"))

        def _parse_last_sign_in_at(
            data: object,
        ) -> Union[None, Unset, datetime.datetime]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, str):
                    raise TypeError()
                last_sign_in_at_type_0 = isoparse(data)

                return last_sign_in_at_type_0
            except:  # noqa: E722
                pass
            return cast(Union[None, Unset, datetime.datetime], data)

        last_sign_in_at = _parse_last_sign_in_at(d.pop("last_sign_in_at", UNSET))

        def _parse_updated_at(data: object) -> Union[None, Unset, datetime.datetime]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, str):
                    raise TypeError()
                updated_at_type_0 = isoparse(data)

                return updated_at_type_0
            except:  # noqa: E722
                pass
            return cast(Union[None, Unset, datetime.datetime], data)

        updated_at = _parse_updated_at(d.pop("updated_at", UNSET))

        user_identity = cls(
            id=id,
            identity_id=identity_id,
            user_id=user_id,
            identity_data=identity_data,
            provider=provider,
            created_at=created_at,
            last_sign_in_at=last_sign_in_at,
            updated_at=updated_at,
        )

        user_identity.additional_properties = d
        return user_identity

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
