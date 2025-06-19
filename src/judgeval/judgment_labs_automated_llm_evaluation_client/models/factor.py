import datetime
from collections.abc import Mapping
from typing import (
    Any,
    TypeVar,
    Union,
    cast,
)

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from dateutil.parser import isoparse

from ..models.factor_factor_type_type_0 import FactorFactorTypeType0
from ..models.factor_status import FactorStatus
from ..types import UNSET, Unset

T = TypeVar("T", bound="Factor")


@_attrs_define
class Factor:
    """A MFA factor.

    Attributes:
        id (str):
        factor_type (Union[FactorFactorTypeType0, str]):
        status (FactorStatus):
        created_at (datetime.datetime):
        updated_at (datetime.datetime):
        friendly_name (Union[None, Unset, str]):
    """

    id: str
    factor_type: Union[FactorFactorTypeType0, str]
    status: FactorStatus
    created_at: datetime.datetime
    updated_at: datetime.datetime
    friendly_name: Union[None, Unset, str] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        id = self.id

        factor_type: str
        if isinstance(self.factor_type, FactorFactorTypeType0):
            factor_type = self.factor_type.value
        else:
            factor_type = self.factor_type

        status = self.status.value

        created_at = self.created_at.isoformat()

        updated_at = self.updated_at.isoformat()

        friendly_name: Union[None, Unset, str]
        if isinstance(self.friendly_name, Unset):
            friendly_name = UNSET
        else:
            friendly_name = self.friendly_name

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "id": id,
                "factor_type": factor_type,
                "status": status,
                "created_at": created_at,
                "updated_at": updated_at,
            }
        )
        if friendly_name is not UNSET:
            field_dict["friendly_name"] = friendly_name

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        id = d.pop("id")

        def _parse_factor_type(data: object) -> Union[FactorFactorTypeType0, str]:
            try:
                if not isinstance(data, str):
                    raise TypeError()
                factor_type_type_0 = FactorFactorTypeType0(data)

                return factor_type_type_0
            except:  # noqa: E722
                pass
            return cast(Union[FactorFactorTypeType0, str], data)

        factor_type = _parse_factor_type(d.pop("factor_type"))

        status = FactorStatus(d.pop("status"))

        created_at = isoparse(d.pop("created_at"))

        updated_at = isoparse(d.pop("updated_at"))

        def _parse_friendly_name(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        friendly_name = _parse_friendly_name(d.pop("friendly_name", UNSET))

        factor = cls(
            id=id,
            factor_type=factor_type,
            status=status,
            created_at=created_at,
            updated_at=updated_at,
            friendly_name=friendly_name,
        )

        factor.additional_properties = d
        return factor

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
